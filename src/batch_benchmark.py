"""Run a resumable sequence of MVTec AD experiments."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "mvtec_ad"
IMAGENETTE = PROJECT_ROOT / "data" / "imagenette2" / "train"
RERUN = PROJECT_ROOT / "outputs"
SINGLE_RUNNER = Path(__file__).with_name("run_anomaly_model.py")
DEFAULT_DATASET = DATASET
DEFAULT_IMAGENETTE = IMAGENETTE
DEFAULT_RESULTS = RERUN / "benchmark_3x15"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resumable 3x15 MVTec AD benchmark runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", choices=("patchcore", "fastflow", "efficientad"), required=True)
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--run-name", default="baseline_v1_seed42")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--imagenette-dir", type=Path, default=DEFAULT_IMAGENETTE)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--val-split-mode", choices=("from_test", "same_as_test"), default="from_test")
    parser.add_argument("--val-split-ratio", type=float, default=0.5)
    parser.add_argument("--num-visualize", type=int, default=6)
    parser.add_argument("--fastflow-epochs", type=int, default=50)
    parser.add_argument("--efficientad-steps", type=int, default=70_000)
    parser.add_argument("--max-categories", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def categories(dataset_root: Path, requested: list[str]) -> list[str]:
    available = sorted(
        item.name
        for item in dataset_root.iterdir()
        if item.is_dir() and (item / "train" / "good").is_dir() and (item / "test").is_dir()
    )
    if requested == ["all"]:
        return available
    unknown = sorted(set(requested) - set(available))
    if unknown:
        raise ValueError(f"Unknown categories: {', '.join(unknown)}")
    return requested


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def result_row(
    model: str,
    category: str,
    run_dir: Path,
    status: str,
    wall_seconds: float,
    return_code: int | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "category": category,
        "status": status,
        "wall_seconds": round(wall_seconds, 2),
        "return_code": return_code,
        "run_dir": str(run_dir),
    }
    metrics = read_json(run_dir / "metrics.json")
    if metrics:
        measured = metrics.get("metrics", {})
        row.update(
            {
                "image_AUROC": measured.get("image_AUROC"),
                "pixel_AUROC": measured.get("pixel_AUROC"),
                "image_F1": measured.get("image_F1Score"),
                "pixel_F1": measured.get("pixel_F1Score"),
                "fit_seconds": metrics.get("fit_seconds"),
                "raw_model_latency_ms": metrics.get("raw_model_latency_ms"),
                "checkpoint_mb": metrics.get("checkpoint_mb"),
                "peak_gpu_memory_mb": metrics.get("peak_gpu_memory_mb"),
                "false_positives": metrics.get("false_positives"),
                "false_negatives": metrics.get("false_negatives"),
            }
        )
    return row


def main() -> int:
    args = parse_args()
    dataset_root = args.dataset_root.absolute()
    results_root = args.results_root.absolute()
    imagenette_dir = args.imagenette_dir.absolute()
    selected = categories(dataset_root, args.categories)
    if args.max_categories is not None:
        selected = selected[: args.max_categories]

    print(f"Models: {', '.join(args.models)}")
    print(f"Categories ({len(selected)}): {', '.join(selected)}")
    print(f"Dataset: {dataset_root}")
    print(f"Results: {results_root}")
    print(
        f"Protocol: run={args.run_name}, image={args.image_size}, seed={args.seed}, "
        f"val={args.val_split_mode}/{args.val_split_ratio}, "
        f"FastFlow={args.fastflow_epochs} epochs, EfficientAD={args.efficientad_steps} steps"
    )

    summary_path = results_root / f"batch_{args.run_name}.csv"
    rows: list[dict[str, Any]] = []
    failed = 0

    for model in args.models:
        for index, category in enumerate(selected, start=1):
            run_dir = results_root / model / category / args.run_name
            status = read_json(run_dir / "status.json")
            print(f"\n[{model} {index}/{len(selected)}] {category}")
            if status and status.get("status") == "complete":
                print("Completed run already exists; skipped.")
                rows.append(result_row(model, category, run_dir, "skipped_complete", 0.0, 0))
                write_summary(rows, summary_path)
                continue

            command = [
                sys.executable,
                str(SINGLE_RUNNER),
                "--model",
                model,
                "--category",
                category,
                "--run-name",
                args.run_name,
                "--dataset-root",
                str(dataset_root),
                "--imagenette-dir",
                str(imagenette_dir),
                "--results-root",
                str(results_root),
                "--image-size",
                str(args.image_size),
                "--seed",
                str(args.seed),
                "--num-workers",
                str(args.num_workers),
                "--eval-batch-size",
                str(args.eval_batch_size),
                "--val-split-mode",
                args.val_split_mode,
                "--val-split-ratio",
                str(args.val_split_ratio),
                "--num-visualize",
                str(args.num_visualize),
                "--fastflow-epochs",
                str(args.fastflow_epochs),
                "--efficientad-steps",
                str(args.efficientad_steps),
            ]
            if args.resume:
                command.append("--resume")
            if args.dry_run:
                command.append("--dry-run")
                print(subprocess.list2cmdline(command))
                rows.append(result_row(model, category, run_dir, "dry_run", 0.0, None))
                continue

            start = time.perf_counter()
            completed = subprocess.run(command, check=False)
            wall_seconds = time.perf_counter() - start
            run_status = read_json(run_dir / "status.json") or {}
            state = run_status.get("status", "unknown")
            if completed.returncode != 0 or state != "complete":
                failed += 1
            rows.append(
                result_row(
                    model,
                    category,
                    run_dir,
                    state,
                    wall_seconds,
                    completed.returncode,
                )
            )
            write_summary(rows, summary_path)

    if args.dry_run:
        write_summary(rows, summary_path)
    print(f"\nBatch finished | runs={len(rows)} failed={failed}")
    print(f"Summary: {summary_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

