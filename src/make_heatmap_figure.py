"""把 export_anomaly_maps.py 匯出的 .npz 畫成「原圖｜官方標註｜三模型熱圖」對比圖。

熱圖是 Anomalib 後處理過的 anomaly map，0.5 即該模型的像素門檻。只替超過門檻的像素上色，
也就是模型實際會判為瑕疵的範圍，三個模型用同一個標準。白色虛線為官方 mask 輪廓。
（EfficientAD 的背景值約 0.49、緊貼門檻，若用 0–1 全色階會整張泛色，因此不採用。）
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "sans-serif"]

MODELS = [("patchcore", "PatchCore"), ("fastflow", "FastFlow"), ("efficientad", "EfficientAD")]
SIZE = 512
BG, INK, MUTED, OK, NG = "#F7F8F5", "#14213D", "#5B6573", "#1E7A52", "#C0392B"
THRESHOLD = 0.5
HEAT = LinearSegmentedColormap.from_list("above_threshold", ["#FFE14D", "#FF8A1F", "#E3261B"])


def load(npz_path: Path):
    data = np.load(npz_path)
    image = np.asarray(Image.open(str(data["image_path"])).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)) / 255
    mask = np.asarray(Image.open(str(data["mask_path"])).convert("L").resize((SIZE, SIZE), Image.NEAREST)) > 127
    maps = {m: np.asarray(Image.fromarray(data[f"{m}_map"].astype(np.float32)).resize((SIZE, SIZE), Image.BILINEAR))
            for m, _ in MODELS}
    scores = {m: float(data[f"{m}_score"]) for m, _ in MODELS}
    return str(data["label"]), image, mask, maps, scores


def overlay(image, amap):
    """只替超過像素門檻（0.5）的區域上色：黃色剛過門檻，紅色遠超門檻。"""
    strength = np.clip((amap - THRESHOLD) / 0.3, 0, 1)
    alpha = np.clip((amap - THRESHOLD) / 0.04, 0, 1)[..., None] * 0.75
    return np.clip(image * (1 - alpha) + HEAT(strength)[..., :3] * alpha, 0, 1)


def overlay_relative(image, amap):
    """失敗案例用：以單張圖自身的最低～最高分縮放，顯示模型「最懷疑哪裡」，即使沒有超過門檻。"""
    v = (amap - amap.min()) / (np.ptp(amap) + 1e-8)
    alpha = np.clip((v - 0.45) / 0.25, 0, 1)[..., None] * 0.75
    return np.clip(image * (1 - alpha) + HEAT(v)[..., :3] * alpha, 0, 1)


def draw(samples: list[Path], out: Path, title: str | None, relative: bool = False) -> None:
    rows, cols = len(samples), 2 + len(MODELS)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.95 + (0.5 if title else 0)), dpi=200,
                             squeeze=False)
    fig.patch.set_facecolor(BG)
    heads = ["輸入影像", "Ground Truth"] + [name for _, name in MODELS]
    for r, npz in enumerate(samples):
        label, image, mask, maps, scores = load(npz)
        mask_fill = image.copy()
        mask_fill[mask] = mask_fill[mask] * 0.35 + np.array([0.9, 0.2, 0.2]) * 0.65
        panels = [image, mask_fill] + [(overlay_relative if relative else overlay)(image, maps[m]) for m, _ in MODELS]
        for c, (ax, panel) in enumerate(zip(axes[r], panels)):
            ax.imshow(panel)
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if c >= 2:
                ax.contour(mask, levels=[0.5], colors="white", linewidths=1.0, linestyles="--")
                m = MODELS[c - 2][0]
                hit = scores[m] >= 0.5
                ax.set_xlabel(f"score {scores[m]:.2f}  ({'TP' if hit else 'FN'})", fontsize=10,
                              color=OK if hit else NG, fontweight="bold", labelpad=5)
            if r == 0:
                ax.set_title(heads[c], fontsize=12, color=INK, fontweight="bold", pad=8)
        axes[r][0].set_ylabel(label, fontsize=11, color=INK, labelpad=6)
    if title:
        fig.suptitle(title, fontsize=14, color=INK, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out, facecolor=BG, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("samples", nargs="+", type=Path, help=".npz 檔，一個一列")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--relative", action="store_true",
                        help="改用單張圖內的相對強度上色（適合全部未過門檻的失敗案例）")
    args = parser.parse_args()
    draw(args.samples, args.out, args.title, args.relative)


if __name__ == "__main__":
    main()
