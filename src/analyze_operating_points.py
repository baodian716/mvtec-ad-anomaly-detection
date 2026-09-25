"""由 45 組實驗的逐圖預測，計算 README 引用的三組補充指標。

輸入：results/per_image_predictions.csv（15 類 test 中未參與門檻決定的那一半，分數已正規化、門檻 0.5）
輸出：results/operating_points.csv，並印出摘要

1. 漏檢率（Escape rate）／過殺率（Overkill rate）
   - 預設門檻 0.5（validation 上 F1 最大）
   - 零漏檢門檻：各類別取「最低的瑕疵分數」當門檻，所有瑕疵都判 NG 時，良品被判 NG 的比例
2. PatchCore 與 EfficientAD 的 Image AUROC 差距：配對 bootstrap 95% 信賴區間
3. 門檻若直接在 test 上挑 F1 最大（等同 validation 與 test 共用），F1 會高估多少
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["patchcore", "fastflow", "efficientad"]


def f1_errors(y: np.ndarray, s: np.ndarray, t: float) -> tuple[float, int]:
    pred = s >= t
    tp, fp, fn = (pred & y).sum(), (pred & ~y).sum(), (~pred & y).sum()
    return 2 * tp / (2 * tp + fp + fn), int(fp + fn)


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "per_image_predictions.csv")
    rows = []
    for (model, category), d in df.groupby(["model", "category"]):
        y, s = d.gt_label.values == 1, d.pred_score.values
        bad, good = s[y], s[~y]
        f1_default, err_default = f1_errors(y, s, 0.5)
        f1_oracle, err_oracle = max((f1_errors(y, s, t) for t in np.unique(s)), key=lambda r: r[0])
        rows.append(dict(model=model, category=category, n_defect=len(bad), n_good=len(good),
                         escape=int((bad < 0.5).sum()), overkill=int((good >= 0.5).sum()),
                         overkill_zero_escape=int((good >= bad.min()).sum()),
                         f1_default=f1_default, f1_test_selected=f1_oracle,
                         errors_default=err_default, errors_test_selected=err_oracle))
    per_cat = pd.DataFrame(rows)
    per_cat.to_csv(ROOT / "results" / "operating_points.csv", index=False, float_format="%.4f")

    total = per_cat.groupby("model")[["n_defect", "n_good", "escape", "overkill", "overkill_zero_escape"]].sum()
    summary = pd.DataFrame({
        "escape_rate": total.escape / total.n_defect,
        "overkill_rate": total.overkill / total.n_good,
        "overkill_at_zero_escape": total.overkill_zero_escape / total.n_good,
        "f1_inflation": per_cat.groupby("model").apply(lambda g: (g.f1_test_selected - g.f1_default).mean()),
        "errors_hidden_per_category": per_cat.groupby("model").apply(
            lambda g: (g.errors_default - g.errors_test_selected).mean()),
    }).loc[MODELS]
    print("== 15 類合計（test 半邊：瑕疵 %d 張、良品 %d 張）" % (total.n_defect.iloc[0], total.n_good.iloc[0]))
    print(summary.round(4).to_string())

    # 配對 bootstrap：每次在各類別內重抽影像（兩模型用同一組影像），重算 15 類平均 AUROC 差距
    rng = np.random.default_rng(0)
    groups = []
    for _, d in df[df.model.isin(["patchcore", "efficientad"])].groupby("category"):
        p = d.pivot_table(index="image", columns="model", values="pred_score")
        y = d[d.model == "patchcore"].set_index("image").loc[p.index].gt_label.values
        groups.append((y, p.patchcore.values, p.efficientad.values))
    diffs = []
    for _ in range(2000):
        per = []
        for y, a, b in groups:
            idx = rng.integers(0, len(y), len(y))
            if y[idx].min() == y[idx].max():
                idx = np.arange(len(y))
            per.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
        diffs.append(np.mean(per))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\n== PatchCore − EfficientAD Image AUROC 差距 95% CI（bootstrap 2000 次）：[{lo:+.4f}, {hi:+.4f}]")


if __name__ == "__main__":
    main()
