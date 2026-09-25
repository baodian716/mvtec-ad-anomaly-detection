"""產生履歷／作品集用的 16:9 封面圖。

左側為兩張測試影像的三模型熱圖，右側為三模型的效果與訓練成本散點圖（同 c12）。
熱圖畫法沿用 make_heatmap_figure.py（只替超過像素門檻的區域上色）。

    python make_portfolio_cover.py                                  # 輪廓為官方 Ground Truth
    python make_portfolio_cover.py --manual-masks <Labelme 遮罩資料夾>  # 輪廓改用人工標註
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from make_heatmap_figure import BG, INK, MUTED, MODELS, SIZE, load, overlay

ROOT = Path(__file__).resolve().parents[1]
MAPS = ROOT / "outputs" / "anomaly_maps"
RESULTS = ROOT / "results" / "all_runs_compact.csv"
COLORS = {"patchcore": "#2F6FB0", "fastflow": "#D9731E", "efficientad": "#1E8A5A"}
GRID = "#DDE2E8"

# 輪廓來源 → (樣本, 輸出檔, 圖例文字)
VARIANTS = {
    "official": (["hazelnut_print_002.npz", "metal_nut_color_012.npz"], "cover_portfolio.png",
                 "Dashed line: ground-truth mask (MVTec AD)"),
    "manual": (["metal_nut_color_019.npz", "metal_nut_color_018.npz"], "cover_portfolio_labelme.png",
               "Dashed line: my manual annotation (Labelme)"),
}


def manual_mask(masks_dir: Path, npz: str) -> np.ndarray:
    stem = npz.removesuffix(".npz").rsplit("_", 1)[-1]
    img = Image.open(masks_dir / f"{stem}_manual_mask.png").convert("L").resize((SIZE, SIZE), Image.NEAREST)
    return np.asarray(img) > 127


def cost_scatter(fig, rect):
    """每類建模時間（對數）對 Image AUROC；小點為單一類別，菱形為 15 類平均。"""
    df = pd.read_csv(RESULTS)
    ax = fig.add_axes(rect)
    ax.set_facecolor(BG)
    for model, name in MODELS:
        d = df[df.model == model]
        c = COLORS[model]
        ax.scatter(d.fit_seconds, d.image_AUROC, s=36, color=c, alpha=0.45, lw=0)
        mx, my = d.fit_seconds.mean(), d.image_AUROC.mean()
        ax.scatter(mx, my, s=220, marker="D", color=c, edgecolor=INK, lw=1.2, zorder=3)
        dy = -0.04 if model == "fastflow" else 0.062
        ax.text(mx, my + dy, f"{name}\n{my:.3f}", ha="center", va="center", fontsize=14,
                fontweight="bold", color=c, linespacing=1.2)
    ax.set_xscale("log")
    ax.set_xlim(4, 12000)
    ax.set_ylim(0.70, 1.10)
    ax.set_xticks([10, 60, 600, 3600], ["10 s", "1 min", "10 min", "1 h"])
    ax.minorticks_off()
    ax.set_yticks([0.7, 0.8, 0.9, 1.0])
    ax.tick_params(labelsize=13, colors=MUTED, length=0)
    ax.grid(color=GRID, lw=0.8)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Training time per category (log scale)", fontsize=13, color=MUTED, labelpad=6)
    ax.set_ylabel("Image AUROC", fontsize=13, color=MUTED, labelpad=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manual-masks", type=Path, help="Labelme 匯出的 <編號>_manual_mask.png 所在資料夾")
    args = parser.parse_args()
    samples, out_name, contour_note = VARIANTS["manual" if args.manual_masks else "official"]

    fig = plt.figure(figsize=(16, 9), dpi=150)
    fig.patch.set_facecolor(BG)

    fig.text(0.04, 0.9, "MVTec AD 工業影像異常偵測", fontsize=34, fontweight="bold", color=INK, va="center")
    fig.text(0.04, 0.83, "Unsupervised defect detection & localization, trained on defect-free images only",
             fontsize=17, color=INK, va="center")
    fig.text(0.04, 0.785, "3 models × 15 categories (10 objects + 5 textures) · 45 benchmark runs",
             fontsize=15, color=MUTED, va="center")

    # 左側：熱圖（4 欄 × 2 列）
    heads = ["Input"] + [name for _, name in MODELS]
    left, top, size, gap = 0.04, 0.705, 0.142, 0.007
    panel_h = size * 16 / 9
    for r, npz in enumerate(samples):
        label, image, mask, maps, _ = load(MAPS / npz)
        if args.manual_masks:
            mask = manual_mask(args.manual_masks, npz)
        panels = [image] + [overlay(image, maps[m]) for m, _ in MODELS]
        for c, panel in enumerate(panels):
            ax = fig.add_axes([left + c * (size + gap), top - (r + 1) * panel_h - r * 0.035, size, panel_h])
            ax.imshow(panel)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c > 0:
                ax.contour(mask, levels=[0.5], colors="white", linewidths=1.1, linestyles="--")
            if r == 0:
                ax.set_title(heads[c], fontsize=15, fontweight="bold", color=INK, pad=6)
            if c == 0:
                ax.set_ylabel(label, fontsize=13, color=INK, labelpad=6)

    fig.text(0.04, 0.085, f"Heatmap: pixels above the anomaly threshold    {contour_note}",
             fontsize=12.5, color=MUTED, va="center")

    # 右側：效果與成本散點圖
    fig.text(0.655, 0.705, "Image AUROC vs. training cost", fontsize=16, fontweight="bold", color=INK,
             va="bottom")
    cost_scatter(fig, [0.7, 0.19, 0.265, 0.48])
    fig.text(0.655, 0.085, "Dots: single category    Diamonds: 15-category mean",
             fontsize=12.5, color=MUTED, va="center")

    out = ROOT / "figures" / out_name
    fig.savefig(out, facecolor=BG)
    print(out)


if __name__ == "__main__":
    main()
