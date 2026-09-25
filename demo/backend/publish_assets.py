"""把 demo 用的 ONNX 模型與影像發布到 Hugging Face model repo，供其他電腦以 ASSETS_REPO 直接下載。

    python demo/backend/publish_assets.py            # 只整理到 outputs/hf_assets，不上傳
    python demo/backend/publish_assets.py --upload   # 整理後上傳（需先執行 hf auth login）

內容：onnx/<模型>/<類別>.onnx，以及 demo 可選的 test 影像與 mask（只含未參與門檻決定的那一半）。
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REPO_ID = "Haolian07/mvtec-ad-demo-assets"
CATEGORIES = ["metal_nut", "screw", "pill"]
OUT = ROOT / "outputs" / "hf_assets"

MODEL_CARD = """---
license: cc-by-nc-sa-4.0
tags: [anomaly-detection, mvtec-ad, onnx, patchcore, fastflow, efficientad]
---

# MVTec AD 異常偵測 Demo 資產

- `onnx/<模型>/<類別>.onnx`：以 Anomalib 2.4.2 訓練、匯出的 ONNX（輸入 1×3×256×256、數值 0～1；輸出含正規化後的 pred_score 與 anomaly_map，門檻 0.5）
- `images/<類別>/test`、`images/<類別>/ground_truth`：demo 使用的 MVTec AD test 影像與 mask

輸入需先以 antialias 雙線性縮放至 256×256（見 GitHub repo 的 `demo/backend/preprocess.py`）。
原始碼、評估與 demo 啟動方式：https://github.com/baodian716/mvtec-ad-anomaly-detection

MVTec AD 影像與衍生模型依原資料集授權 CC BY-NC-SA 4.0 提供，僅供非商業用途；
資料集出處：Bergmann et al., "The MVTec Anomaly Detection Dataset", IJCV 2021。
"""


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def prepare() -> None:
    shutil.rmtree(OUT, ignore_errors=True)
    OUT.mkdir(parents=True)
    (OUT / "README.md").write_text(MODEL_CARD, encoding="utf-8")
    for f in (ROOT / "weights" / "onnx").rglob("*.onnx"):
        copy(f, OUT / "onnx" / f.relative_to(ROOT / "weights" / "onnx"))
    preds = pd.read_csv(ROOT / "results" / "per_image_predictions.csv")
    preds = preds[(preds.model == "patchcore") & preds.category.isin(CATEGORIES)]
    data = ROOT / "data" / "mvtec_ad"
    for r in preds.itertuples():
        copy(data / r.category / "test" / r.image, OUT / "images" / r.category / "test" / r.image)
        if r.gt_label == 1:
            defect, file = r.image.split("/")
            mask = f"{r.category}/ground_truth/{defect}/{file.replace('.png', '_mask.png')}"
            copy(data / mask, OUT / "images" / mask)
    files = [f for f in OUT.rglob("*") if f.is_file()]
    print(f"{OUT.relative_to(ROOT)}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e6:.0f} MB")


def upload() -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(REPO_ID, repo_type="model", exist_ok=True)
    api.upload_large_folder(repo_id=REPO_ID, repo_type="model", folder_path=OUT)
    print(f"https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()
    prepare()
    if args.upload:
        upload()
