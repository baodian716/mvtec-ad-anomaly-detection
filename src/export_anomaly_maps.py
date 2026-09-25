"""把指定影像的三模型 anomaly map 匯出成 .npz，供 make_heatmap_figure.py 繪圖。

EfficientAD / FastFlow 直接載入已訓練的 model.ckpt 推論；PatchCore 沒有保留權重，
依正式協定（wide_resnet50_2、coreset 0.1、seed 42、from_test 0.5）重建該類別的記憶庫。
推論沿用正式實驗的 MVTecAD datamodule，因此前處理、切分與門檻都與 45 組實驗相同。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Fastflow, Patchcore
from lightning.pytorch import seed_everything

# 類別 → 要匯出的測試影像（相對於 test/）
DEFAULT_TARGETS = {
    "metal_nut": ["color/012.png", "color/004.png", "color/018.png", "color/019.png"],
    "hazelnut": ["print/002.png"],
    "capsule": ["crack/017.png"],
    "carpet": ["cut/002.png"],
    "screw": ["thread_top/011.png"],
    "pill": ["crack/024.png"],
}
MODELS = ("patchcore", "fastflow", "efficientad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--weights-root", type=Path, required=True,
                        help="含 efficientad/<類別>/model.ckpt 與 fastflow/<類別>/model.ckpt 的資料夾")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/anomaly_maps"))
    parser.add_argument("--categories", nargs="*", default=list(DEFAULT_TARGETS))
    return parser.parse_args()


def build_model(name: str, category: str, weights_root: Path):
    if name == "patchcore":
        pre_processor = Patchcore.configure_pre_processor(image_size=(256, 256))
        return Patchcore(backbone="wide_resnet50_2", layers=("layer2", "layer3"), pre_trained=True,
                         coreset_sampling_ratio=0.1, num_neighbors=9, pre_processor=pre_processor)
    cls = Fastflow if name == "fastflow" else EfficientAd
    return cls.load_from_checkpoint(weights_root / name / category / "model.ckpt",
                                    weights_only=False, map_location="cpu")


def predict(name: str, category: str, args: argparse.Namespace) -> dict[str, tuple[np.ndarray, float]]:
    seed_everything(42, workers=True)
    datamodule = MVTecAD(root=args.dataset_root, category=category, train_batch_size=8,
                         eval_batch_size=8, num_workers=0, val_split_mode="from_test",
                         val_split_ratio=0.5, seed=42)
    engine = Engine(accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1,
                    default_root_dir=args.output_dir / "_runs" / name / category,
                    logger=False, enable_progress_bar=False, max_epochs=1)
    model = build_model(name, category, args.weights_root)
    if name == "patchcore":
        engine.fit(model=model, datamodule=datamodule)
    batches = engine.predict(model=model, datamodule=datamodule)

    wanted = set(DEFAULT_TARGETS[category])
    found = {}
    for batch in batches:
        for i, path in enumerate(batch.image_path):
            rel = Path(path).as_posix().split("/test/")[-1]
            if rel in wanted:
                amap = np.squeeze(batch.anomaly_map[i].detach().cpu().float().numpy())
                found[rel] = (amap, float(batch.pred_score[i].detach().cpu()))
    return found


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for category in args.categories:
        results = {m: predict(m, category, args) for m in MODELS}
        for rel in DEFAULT_TARGETS[category]:
            defect, stem = rel.removesuffix(".png").split("/")
            out = args.output_dir / f"{category}_{defect}_{stem}.npz"
            payload = {"label": f"{category} / {defect}",
                       "image_path": str(args.dataset_root / category / "test" / rel),
                       "mask_path": str(args.dataset_root / category / "ground_truth" / defect / f"{stem}_mask.png")}
            for m in MODELS:
                amap, score = results[m][rel]
                payload[f"{m}_map"], payload[f"{m}_score"] = amap, score
            np.savez_compressed(out, **payload)
            print(out, {m: round(float(payload[f"{m}_score"]), 4) for m in MODELS})


if __name__ == "__main__":
    main()
