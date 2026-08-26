"""Summarize the compact results included with this repository."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "all_runs_compact.csv"
OUTPUT = ROOT / "outputs" / "published_summary"


def main() -> None:
    if not RESULTS.is_file():
        raise FileNotFoundError(f"Published result table not found: {RESULTS}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    runs = pd.read_csv(RESULTS)
    required = {
        "model",
        "category",
        "image_AUROC",
        "image_F1",
        "pixel_AUROC",
        "pixel_F1",
        "fit_seconds",
        "latency_ms",
        "model_mb",
        "FP",
        "FN",
    }
    missing = required - set(runs.columns)
    if missing:
        raise ValueError(f"Missing columns: {', '.join(sorted(missing))}")

    score_columns = ["image_AUROC", "image_F1", "pixel_AUROC", "pixel_F1"]
    resource_columns = ["fit_seconds", "latency_ms", "model_mb"]
    summary = runs.groupby("model").agg(
        **{f"mean_{column}": (column, "mean") for column in score_columns + resource_columns},
        total_FP=("FP", "sum"),
        total_FN=("FN", "sum"),
        categories=("category", "nunique"),
    )
    summary.to_csv(OUTPUT / "model_summary.csv", encoding="utf-8-sig")

    scores = summary[[f"mean_{column}" for column in score_columns]].copy()
    scores.columns = [column.replace("_", " ") for column in score_columns]
    axis = scores.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=0)
    axis.set_title("Mean performance across MVTec AD categories")
    axis.set_ylabel("Score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="lower right")
    axis.figure.tight_layout()
    axis.figure.savefig(OUTPUT / "model_scores.png", dpi=180)
    plt.close(axis.figure)

    errors = summary[["total_FP", "total_FN"]].copy()
    errors.columns = ["False positives", "False negatives"]
    axis = errors.plot(kind="bar", figsize=(8, 5), rot=0, color=["#4c78a8", "#e45756"])
    axis.set_title("Classification errors across evaluated images")
    axis.set_ylabel("Image count")
    axis.grid(axis="y", alpha=0.25)
    axis.figure.tight_layout()
    axis.figure.savefig(OUTPUT / "classification_errors.png", dpi=180)
    plt.close(axis.figure)

    print(summary.round(4).to_string())
    print(f"\nSaved summary to: {OUTPUT}")


if __name__ == "__main__":
    main()
