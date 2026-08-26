"""建立 metal_nut 的小型資料擴增 A/B 測試資料集。"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "mvtec_ad"
RERUN = PROJECT_ROOT / "outputs"
SOURCE_CATEGORY = DATASET / "metal_nut"
OUTPUT_ROOT = RERUN / "augmentation_datasets"


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            link_or_copy(path, destination / path.relative_to(source))


def build_variant(name: str) -> None:
    category_root = OUTPUT_ROOT / name / "metal_nut"
    train_out = category_root / "train" / "good"
    train_out.mkdir(parents=True, exist_ok=True)

    source_images = sorted((SOURCE_CATEGORY / "train" / "good").glob("*.png"))
    for index, source in enumerate(source_images):
        link_or_copy(source, train_out / f"{source.stem}_orig.png")

        with Image.open(source).convert("RGB") as image:
            if name == "illumination":
                brightness = (0.88, 0.94, 1.06, 1.12)[index % 4]
                contrast = (0.92, 1.08)[index % 2]
                augmented = ImageEnhance.Brightness(image).enhance(brightness)
                augmented = ImageEnhance.Contrast(augmented).enhance(contrast)
            elif name == "rotation":
                angle = (-5, -3, 3, 5)[index % 4]
                augmented = image.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0))
            else:
                raise ValueError(f"Unknown variant: {name}")

            augmented.save(train_out / f"{source.stem}_aug.png", optimize=True)

    # 測試圖片與標註保持完全相同；優先使用 hard link，避免重複占用空間。
    link_tree(SOURCE_CATEGORY / "test", category_root / "test")
    link_tree(SOURCE_CATEGORY / "ground_truth", category_root / "ground_truth")
    for metadata_name in ("license.txt", "readme.txt"):
        source = SOURCE_CATEGORY / metadata_name
        if source.exists():
            link_or_copy(source, category_root / metadata_name)

    print(f"{name}: train={len(list(train_out.glob('*.png')))}, test root ready")


def main() -> None:
    for variant in ("illumination", "rotation"):
        build_variant(variant)


if __name__ == "__main__":
    main()
