"""Demo 後端：FastAPI + ONNX Runtime（CPU），不需要 PyTorch。

    python demo/backend/server.py          # 啟動後自動開啟 http://localhost:7860

API
  GET  /api/meta                  類別、模型、demo 案例
  GET  /api/images/{category}     該類別可選的 test 影像（皆為未參與門檻決定的那一半）
  GET  /api/scores/{category}     45 組實驗的逐圖分數，前端據此即時計算漏檢率／過殺率
  POST /api/infer                 {category, image} → 三模型分數、判定、推論時間與熱圖
前端（demo/frontend 建置後的 dist/）由同一個伺服器提供。
"""
from __future__ import annotations

import base64
import io
import sys
import threading
import time
import webbrowser
from functools import lru_cache
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel
from scipy.ndimage import binary_dilation, binary_erosion

sys.path.insert(0, str(Path(__file__).parent))
from preprocess import load_input  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DATA, ONNX_DIR = ROOT / "data" / "mvtec_ad", ROOT / "weights" / "onnx"
FRONTEND = ROOT / "demo" / "frontend" / "dist"
PORT, THRESHOLD, VIEW = 7860, 0.5, 384

CATEGORIES = ["metal_nut", "hazelnut", "capsule", "carpet", "screw", "pill"]
MODELS = [("patchcore", "PatchCore"), ("fastflow", "FastFlow"), ("efficientad", "EfficientAD")]
DEMO_CASES = [  # 分數皆可在 results/per_image_predictions.csv 查到
    {"label": "三模型皆正確檢出", "category": "hazelnut", "image": "print/002.png", "kind": "ok"},
    {"label": "三模型皆漏檢", "category": "pill", "image": "crack/024.png", "kind": "escape"},
    {"label": "良品被誤判（過殺）", "category": "screw", "image": "good/000.png", "kind": "overkill"},
]

PREDICTIONS = pd.read_csv(ROOT / "results" / "per_image_predictions.csv")
PREDICTIONS = PREDICTIONS[PREDICTIONS.category.isin(CATEGORIES)]

# 熱圖配色：只替超過門檻的像素上色，黃（剛過門檻）→ 橘 → 紅（遠超門檻），與 README 圖表一致
_STOPS = np.array([[255, 225, 77], [255, 138, 31], [227, 38, 27]], dtype=np.float32) / 255


def _heat_color(strength: np.ndarray) -> np.ndarray:
    pos = strength * (len(_STOPS) - 1)
    i = np.clip(pos.astype(int), 0, len(_STOPS) - 2)
    t = (pos - i)[..., None]
    return _STOPS[i] * (1 - t) + _STOPS[i + 1] * t


def overlay(image: np.ndarray, amap: np.ndarray) -> np.ndarray:
    strength = np.clip((amap - THRESHOLD) / 0.3, 0, 1)
    alpha = np.clip((amap - THRESHOLD) / 0.04, 0, 1)[..., None] * 0.75
    return np.clip(image * (1 - alpha) + _heat_color(strength) * alpha, 0, 1)


def with_contour(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = image.copy()
    out[binary_dilation(mask & ~binary_erosion(mask, iterations=2))] = 1.0
    return out


def to_data_url(image: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray((image * 255).astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@lru_cache(maxsize=None)
def session(model: str, category: str) -> ort.InferenceSession:
    sess = ort.InferenceSession(str(ONNX_DIR / model / f"{category}.onnx"), providers=["CPUExecutionProvider"])
    sess.run(None, {"input": np.zeros((1, 3, 256, 256), np.float32)})  # 暖機
    return sess


def allowed_images(category: str) -> pd.DataFrame:
    if category not in CATEGORIES:
        raise HTTPException(404, "unknown category")
    return PREDICTIONS[(PREDICTIONS.model == "patchcore") & (PREDICTIONS.category == category)]


app = FastAPI(title="MVTec AD 工業異常偵測 Demo")


@app.get("/api/meta")
def meta():
    return {"categories": CATEGORIES, "models": [{"id": m, "name": n} for m, n in MODELS],
            "demo_cases": DEMO_CASES, "threshold": THRESHOLD}


@app.get("/api/images/{category}")
def images(category: str):
    d = allowed_images(category).sort_values(["gt_label", "image"])
    return [{"image": r.image, "is_defect": bool(r.gt_label), "defect_type": r.defect_type} for r in d.itertuples()]


@app.get("/api/scores/{category}")
def scores(category: str):
    allowed_images(category)
    d = PREDICTIONS[PREDICTIONS.category == category]
    return {m: {"defect": d[(d.model == m) & (d.gt_label == 1)].pred_score.round(4).tolist(),
                "good": d[(d.model == m) & (d.gt_label == 0)].pred_score.round(4).tolist()} for m, _ in MODELS}


class InferRequest(BaseModel):
    category: str
    image: str


@app.post("/api/infer")
def infer(req: InferRequest):
    if req.image not in set(allowed_images(req.category).image):  # 只允許清單內的影像，避免任意路徑
        raise HTTPException(404, "unknown image")
    path = DATA / req.category / "test" / req.image
    defect, file = req.image.split("/")
    mask_path = DATA / req.category / "ground_truth" / defect / file.replace(".png", "_mask.png")
    is_defect = mask_path.exists()

    x = load_input(path)
    shown = np.asarray(Image.open(path).convert("RGB").resize((VIEW, VIEW), Image.BICUBIC), np.float32) / 255
    mask = (np.asarray(Image.open(mask_path).convert("L").resize((VIEW, VIEW), Image.NEAREST)) > 127
            if is_defect else np.zeros((VIEW, VIEW), bool))
    gt = shown.copy()
    gt[mask] = gt[mask] * 0.35 + np.array([0.9, 0.2, 0.2]) * 0.65

    results = []
    for model, name in MODELS:
        sess = session(model, req.category)
        start = time.perf_counter()
        out = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, {"input": x})))
        ms = (time.perf_counter() - start) * 1000
        score = float(np.ravel(out["pred_score"])[0])
        amap = np.asarray(Image.fromarray(np.squeeze(out["anomaly_map"]).astype(np.float32))
                          .resize((VIEW, VIEW), Image.BILINEAR))
        ng = score >= THRESHOLD
        results.append({"model": model, "name": name, "score": round(score, 3), "verdict": "NG" if ng else "OK",
                        "outcome": "correct" if ng == is_defect else ("escape" if is_defect else "overkill"),
                        "ms": round(ms, 1), "heatmap": to_data_url(with_contour(overlay(shown, amap), mask))})
    return {"is_defect": is_defect, "defect_type": defect, "input": to_data_url(shown),
            "ground_truth": to_data_url(gt), "results": results}


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")


if __name__ == "__main__":
    threading.Timer(2.0, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT)
