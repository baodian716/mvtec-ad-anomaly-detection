"""Compare five manual Labelme polygons with the official MVTec masks."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = PROJECT_ROOT / "data" / "labelme_sample"
ANNOTATION_DIR = SAMPLE_ROOT / "annotations"
REFERENCE_DIR = SAMPLE_ROOT / "reference_masks"
MANUAL_MASK_DIR = SAMPLE_ROOT / "manual_masks"
COMPARISON_DIR = SAMPLE_ROOT / "comparisons"


def labelme_to_mask(data: dict) -> np.ndarray:
    width = int(data["imageWidth"])
    height = int(data["imageHeight"])
    canvas = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    for shape in data.get("shapes", []):
        if shape.get("label") != "defect" or shape.get("shape_type") != "polygon":
            continue
        points = [(float(x), float(y)) for x, y in shape["points"]]
        if len(points) >= 3:
            draw.polygon(points, fill=255)
    return np.asarray(canvas) > 0


def calculate_metrics(manual: np.ndarray, official: np.ndarray) -> dict[str, float | int]:
    intersection = int(np.logical_and(manual, official).sum())
    union = int(np.logical_or(manual, official).sum())
    manual_area = int(manual.sum())
    official_area = int(official.sum())
    iou = intersection / union if union else 1.0
    dice = 2 * intersection / (manual_area + official_area) if manual_area + official_area else 1.0
    precision = intersection / manual_area if manual_area else 0.0
    recall = intersection / official_area if official_area else 0.0
    return {
        "manual_pixels": manual_area,
        "official_pixels": official_area,
        "area_ratio": manual_area / official_area if official_area else 0.0,
        "intersection_pixels": intersection,
        "iou": iou,
        "dice": dice,
        "pixel_precision": precision,
        "pixel_recall": recall,
    }


def save_comparison(image_path: Path, manual: np.ndarray, official: np.ndarray, output_path: Path) -> None:
    image = np.asarray(Image.open(image_path).convert("RGB"))
    overlap = np.logical_and(manual, official)
    manual_only = np.logical_and(manual, ~official)
    official_only = np.logical_and(official, ~manual)

    comparison = image.astype(np.float32) * 0.35
    comparison[manual_only] = (255, 60, 60)
    comparison[official_only] = (60, 220, 90)
    comparison[overlap] = (255, 220, 40)
    comparison = comparison.astype(np.uint8)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[1].imshow(manual, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Manual Labelme")
    axes[2].imshow(official, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title("Official MVTec")
    axes[3].imshow(comparison)
    axes[3].set_title("Yellow=Overlap\nRed=Manual only, Green=Official only")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    MANUAL_MASK_DIR.mkdir(parents=True, exist_ok=True)
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for annotation_path in sorted(ANNOTATION_DIR.glob("*.json")):
        image_id = annotation_path.stem
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        manual = labelme_to_mask(data)
        reference_path = REFERENCE_DIR / f"{image_id}_mask.png"
        official = np.asarray(Image.open(reference_path).convert("L")) > 0
        if official.shape != manual.shape:
            official = np.asarray(
                Image.fromarray(official.astype(np.uint8) * 255).resize(
                    (manual.shape[1], manual.shape[0]),
                    Image.Resampling.NEAREST,
                )
            ) > 0

        Image.fromarray(manual.astype(np.uint8) * 255).save(MANUAL_MASK_DIR / f"{image_id}_manual_mask.png")
        save_comparison(
            SAMPLE_ROOT / "images" / f"{image_id}.png",
            manual,
            official,
            COMPARISON_DIR / f"{image_id}_comparison.png",
        )
        metrics = calculate_metrics(manual, official)
        rows.append({"image_id": image_id, **metrics})

    csv_path = SAMPLE_ROOT / "labelme_vs_mvtec_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    averages = {
        key: sum(float(row[key]) for row in rows) / len(rows)
        for key in ("area_ratio", "iou", "dice", "pixel_precision", "pixel_recall")
    }
    report_lines = [
        "# Labelme 人工標註與 MVTec 官方 mask 比較",
        "",
        "## 評估結果",
        "",
        "| 圖片 | 人工/官方面積比 | IoU | Dice | Pixel Precision | Pixel Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        report_lines.append(
            f"| {row['image_id']} | {row['area_ratio']:.3f} | {row['iou']:.3f} | "
            f"{row['dice']:.3f} | {row['pixel_precision']:.3f} | {row['pixel_recall']:.3f} |"
        )
    report_lines.extend(
        [
            f"| **平均** | **{averages['area_ratio']:.3f}** | **{averages['iou']:.3f}** | "
            f"**{averages['dice']:.3f}** | **{averages['pixel_precision']:.3f}** | "
            f"**{averages['pixel_recall']:.3f}** |",
            "",
            "## 指標意義",
            "",
            "- IoU：人工區域與官方區域的交集除以聯集，越接近 1 越一致。",
            "- Dice：另一種區域重疊程度，越接近 1 越一致。",
            "- Pixel Precision：人工圈選的像素中，有多少落在官方 mask。",
            "- Pixel Recall：官方瑕疵像素中，有多少被人工圈到。",
            "- 面積比小於 1 代表人工圈得較窄，大於 1 代表人工圈得較寬。",
            "",
            "比較圖顏色：黃色為重疊、紅色為人工多圈、綠色為人工漏圈。",
        ]
    )
    (SAMPLE_ROOT / "Labelme標註比較報告.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Evaluated {len(rows)} annotations")
    print(f"Mean IoU={averages['iou']:.4f}, Dice={averages['dice']:.4f}")
    print(f"Report: {SAMPLE_ROOT / 'Labelme標註比較報告.md'}")


if __name__ == "__main__":
    main()

