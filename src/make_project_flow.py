"""產生 README 用的專案流程圖：figures/project_flow.png

五個階段由左到右：資料 → 模型 → 評估 → 分析 → 決策。
同一階段內平行的工作（三個模型、三組單變因實驗）畫成並列方塊。
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "Noto Sans CJK TC", "PingFang TC", "sans-serif"]

OUT = Path(__file__).resolve().parents[1] / "figures" / "project_flow.png"

NAVY, BODY, MUTED = "#14213D", "#3F4A5A", "#8A94A3"
BLUE, LINE, CARD, BG, TINT = "#2F6FB0", "#C9D2DE", "#FFFFFF", "#F7F8F5", "#E8EFF7"

# 階段欄位：(標題, 左緣 x, 寬度)
STAGES = [("資料", 2, 27), ("模型", 35, 27), ("評估", 68, 29), ("分析", 107, 29), ("決策", 142, 24)]


def box(ax, x, y, w, h, title, lines=(), *, face=CARD, edge=LINE, dashed=False,
        title_c=NAVY, text_c=BODY, size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=face, ec=edge, lw=1.3, ls=(0, (4, 2)) if dashed else "-"))
    ax.text(x + 1.6, y + h - 2.8, title, va="center", fontsize=size + 1.5, fontweight="bold", color=title_c)
    for i, line in enumerate(lines):
        ax.text(x + 1.6, y + h - 6.9 - i * 3.3, line, va="center", fontsize=size, color=text_c)


def arrow(ax, p, q, *, color=MUTED, dashed=False, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14, lw=1.5, color=color,
                                 ls=(0, (4, 2)) if dashed else "-", connectionstyle=f"arc3,rad={rad}"))


def bus(ax, x_from, y_from, x_bus, targets, x_to, sources=None):
    """從來源拉到一條垂直匯流線，再分出箭頭到各目標。"""
    ys = list(targets) + (list(sources) if sources else [y_from])
    ax.plot([x_bus, x_bus], [min(ys), max(ys)], color=MUTED, lw=1.5)
    for y in sources or [y_from]:
        ax.plot([x_from, x_bus], [y, y], color=MUTED, lw=1.5)
    for y in targets:
        arrow(ax, (x_bus, y), (x_to, y))


def main():
    fig, ax = plt.subplots(figsize=(15.5, 6.6), dpi=200)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 168)
    ax.set_ylim(0, 74)
    ax.axis("off")

    # 階段標題列
    for i, (name, x, w) in enumerate(STAGES):
        ax.add_patch(FancyBboxPatch((x, 68.5), w, 4.6, boxstyle="round,pad=0,rounding_size=2.3",
                                    fc=NAVY if i == len(STAGES) - 1 else BLUE, ec="none"))
        ax.text(x + w / 2, 70.8, f"{i + 1}　{name}", ha="center", va="center",
                fontsize=12.5, fontweight="bold", color="#FFFFFF")

    # 1 資料
    box(ax, 2, 50, 27, 15, "MVTec AD", ["15 類：10 物件 + 5 紋理", "工業影像基準資料集"])
    box(ax, 2, 34, 27, 12.5, "train", ["只有良品影像", "→ 非監督式學習"])
    box(ax, 2, 4, 27, 26.5, "test（依標籤分層切半）", [])
    box(ax, 4, 15, 23, 9, "validation 50%", ["決定門檻與正規化"], face=TINT, edge=TINT, size=9.5)
    box(ax, 4, 5.5, 23, 8, "test 50%", ["只用來評分"], face=TINT, edge=TINT, size=9.5)

    # 2 模型（平行）
    models = [("PatchCore", ["特徵記憶庫＋最近鄰距離", "WRN50 · coreset 10%"]),
              ("FastFlow", ["Normalizing flow 機率", "ResNet18 · 50 epochs"]),
              ("EfficientAD", ["Teacher–student 差異", "70,000 steps"])]
    for k, (name, lines) in enumerate(models):
        box(ax, 35, 48 - k * 20.5, 27, 17, name, lines)
    ax.text(48.5, 3.2, "256 px · seed 42 · 統一協定", ha="center", va="center", fontsize=9.5, color=MUTED)

    # 3 評估
    box(ax, 68, 44, 29, 21, "偵測與定位指標",
        ["Image AUROC / F1", "Pixel AUROC / F1", "AUPRO", "FP / FN 張數"])
    box(ax, 68, 27, 29, 14, "部署成本", ["建模時間 · 推論延遲", "模型大小"])
    box(ax, 68, 4, 29, 20, "實驗產物", ["config · metrics", "逐圖預測 · checkpoint"], dashed=True)
    ax.text(69.6, 7.8, "可重新推論補算指標，不必重訓", va="center", fontsize=9.5, color=BLUE, fontweight="bold")

    # 4 分析
    box(ax, 107, 44, 29, 21, "失敗分析",
        ["漏判的分數分布", "三模型錯誤重疊", "最難瑕疵排行", "bent vs color"])
    ax.text(121.5, 40.3, "↓  選定 metal_nut / color", ha="center", va="center", fontsize=10, color=BLUE,
            fontweight="bold")
    box(ax, 107, 4, 29, 33, "單變因實驗", [])
    for k, (name, desc) in enumerate([("標註品質", "Labelme vs 官方 mask"),
                                      ("資料擴增", "照明 · 旋轉"),
                                      ("解析度", "256 → 384 · 3 seeds")]):
        box(ax, 109, 22.5 - k * 9, 25, 8, name, [desc], face=TINT, edge=TINT, size=9.5)

    # 5 決策
    box(ax, 142, 4, 24, 61, "選型結論",
        ["預設 PatchCore", "效果最佳、訓練最快", "", "精準定位用 EfficientAD", "Pixel F1 最高", "",
         "FastFlow 暫不採用", "瓶頸在 backbone", "", "門檻邊界設灰區複檢"],
        face=NAVY, edge=NAVY, title_c="#FFFFFF", text_c="#D5DEEA")

    # 階段之間的箭頭：平行分支用一條匯流線連接
    bus(ax, 29.5, 40.25, 32, [56.5, 36, 15.5], 34.5)          # train → 三個模型
    bus(ax, 62.5, None, 65, [54.5, 34, 14], 67.5, sources=[56.5, 36, 15.5])  # 模型 → 評估
    arrow(ax, (97.5, 58), (106.5, 58))
    arrow(ax, (136.5, 58), (141.5, 58))
    arrow(ax, (136.5, 20), (141.5, 20))

    fig.savefig(OUT, bbox_inches="tight", pad_inches=0.15, facecolor=BG)
    print(OUT)


if __name__ == "__main__":
    main()
