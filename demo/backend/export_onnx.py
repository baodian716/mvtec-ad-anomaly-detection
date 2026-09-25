"""把 weights/ 內的三模型 checkpoint 匯出成 ONNX，並逐張驗證分數與實驗紀錄一致。

    python demo/backend/export_onnx.py            # 匯出 + 驗證
    python demo/backend/export_onnx.py --verify   # 只驗證既有的 ONNX

輸出：weights/onnx/<模型>/<類別>.onnx
驗證：以 onnxruntime（CPU）推論 data/mvtec_ad 中 6 類 test 影像，與 results/per_image_predictions.csv 比對。
"""
import argparse
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
from preprocess import INPUT_SIZE, load_input  # noqa: E402

warnings.filterwarnings("ignore")

CATEGORIES = ["metal_nut", "screw", "pill"]
MODELS = ["patchcore", "fastflow", "efficientad"]
WEIGHTS, DATA = ROOT / "weights", ROOT / "data" / "mvtec_ad"


def export() -> None:
    from anomalib.models import EfficientAd, Fastflow, Patchcore

    classes = {"patchcore": Patchcore, "fastflow": Fastflow, "efficientad": EfficientAd}
    for name in MODELS:
        for category in CATEGORIES:
            model = classes[name].load_from_checkpoint(WEIGHTS / name / category / "model.ckpt",
                                                       weights_only=False, map_location="cpu").eval()
            with tempfile.TemporaryDirectory() as tmp:
                # 關閉常數摺疊：否則 PatchCore 的記憶庫會多存一份轉置複本（檔案約大 60%）
                path = model.to_onnx(tmp, input_size=(INPUT_SIZE, INPUT_SIZE), do_constant_folding=False)
                target = WEIGHTS / "onnx" / name / f"{category}.onnx"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(path, target)
            print(f"exported {target.relative_to(ROOT)}  ({target.stat().st_size / 1e6:.0f} MB)")


def verify() -> None:
    expected = pd.read_csv(ROOT / "results" / "per_image_predictions.csv")
    rows = []
    for name in MODELS:
        for category in CATEGORIES:
            session = ort.InferenceSession(str(WEIGHTS / "onnx" / name / f"{category}.onnx"),
                                           providers=["CPUExecutionProvider"])
            outputs = [o.name for o in session.get_outputs()]
            ref = expected[(expected.model == name) & (expected.category == category)]
            for r in ref.itertuples():
                result = dict(zip(outputs, session.run(None, {"input": load_input(DATA / category / "test" / r.image)})))
                score = float(np.ravel(result["pred_score"])[0])
                rows.append(dict(model=name, category=category, image=r.image, expected=r.pred_score, onnx=score,
                                 same_verdict=(score >= 0.5) == (r.pred_score >= 0.5)))
            print(f"verified {name}/{category}: {len(ref)} images")
    df = pd.DataFrame(rows)
    df["abs_diff"] = (df.onnx - df.expected).abs()
    summary = df.groupby("model").agg(images=("image", "size"), max_abs_diff=("abs_diff", "max"),
                                      mean_abs_diff=("abs_diff", "mean"), verdict_mismatch=("same_verdict", lambda s: int((~s).sum())))
    print(summary.round(5).to_string())
    df.to_csv(ROOT / "results" / "onnx_parity.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verify", action="store_true", help="只驗證，不重新匯出")
    if not parser.parse_args().verify:
        export()
    verify()
