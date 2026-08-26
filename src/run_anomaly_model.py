"""Run one reproducible MVTec AD anomaly-detection experiment.

Supported models:
- PatchCore
- FastFlow
- EfficientAD

Every run writes its configuration, environment, status, metrics, predictions,
training curves, checkpoint, timing, visualizations, and failure details into
one self-contained directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import socket
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# Repository-relative defaults. Every path can still be overridden by CLI.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "data" / "mvtec_ad"
IMAGENETTE = PROJECT_ROOT / "data" / "imagenette2" / "train"
RERUN = PROJECT_ROOT / "outputs"
LOCAL_CACHE = RERUN / "_cache"
os.environ.setdefault("HF_HOME", str(LOCAL_CACHE / "huggingface"))
os.environ.setdefault("TORCH_HOME", str(LOCAL_CACHE / "torch"))

import numpy as np
import torch
from PIL import Image

from anomalib.callbacks import ModelCheckpoint
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import EfficientAd, Fastflow, Patchcore
from lightning.pytorch import seed_everything
from lightning.pytorch.loggers import CSVLogger
from sklearn.metrics import f1_score, roc_auc_score


DEFAULT_DATASET = DATASET
DEFAULT_IMAGENETTE = IMAGENETTE
DEFAULT_RESULTS = RERUN / "benchmark_3x15"
SUPPORTED_MODELS = ("patchcore", "fastflow", "efficientad")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one monitored Anomalib experiment on MVTec AD.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--category", required=True)
    parser.add_argument("--run-name", default="baseline_v1_seed42")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--imagenette-dir", type=Path, default=DEFAULT_IMAGENETTE)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument(
        "--val-split-mode",
        choices=("from_test", "same_as_test"),
        default="from_test",
    )
    parser.add_argument("--val-split-ratio", type=float, default=0.5)
    parser.add_argument("--num-visualize", type=int, default=6)
    parser.add_argument("--latency-iterations", type=int, default=30)

    parser.add_argument("--patchcore-backbone", default="wide_resnet50_2")
    parser.add_argument("--patchcore-coreset-ratio", type=float, default=0.1)
    parser.add_argument("--patchcore-neighbors", type=int, default=9)

    parser.add_argument("--fastflow-backbone", default="resnet18")
    parser.add_argument("--fastflow-epochs", type=int, default=50)
    parser.add_argument("--fastflow-flow-steps", type=int, default=8)
    parser.add_argument("--fastflow-hidden-ratio", type=float, default=1.0)

    parser.add_argument("--efficientad-model-size", choices=("small", "medium"), default="small")
    parser.add_argument("--efficientad-steps", type=int, default=70_000)
    parser.add_argument("--efficientad-teacher-channels", type=int, default=384)
    parser.add_argument("--efficientad-lr", type=float, default=1e-4)
    parser.add_argument("--efficientad-weight-decay", type=float, default=1e-5)

    parser.add_argument("--checkpoint-every-steps", type=int, default=5_000)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def configure_logging(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("nschool_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8", mode="a")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in ("anomalib", "torch", "torchvision", "lightning", "numpy", "scikit-learn"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    gpu = None
    if torch.cuda.is_available():
        gpu = {
            "name": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
        }
    return {
        "created_at": now_iso(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": packages,
        "gpu": gpu,
    }


def discover_categories(dataset_root: Path) -> list[str]:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    return sorted(
        item.name
        for item in dataset_root.iterdir()
        if item.is_dir() and (item / "train" / "good").is_dir() and (item / "test").is_dir()
    )


def experiment_config(args: argparse.Namespace) -> dict[str, Any]:
    config = {
        "protocol_version": "baseline_v1",
        "model": args.model,
        "category": args.category,
        "run_name": args.run_name,
        "dataset_root": str(args.dataset_root.absolute()),
        "imagenette_dir": str(args.imagenette_dir.absolute()),
        "results_root": str(args.results_root.absolute()),
        "image_size": args.image_size,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "eval_batch_size": args.eval_batch_size,
        "val_split_mode": args.val_split_mode,
        "val_split_ratio": args.val_split_ratio,
        "num_visualize": args.num_visualize,
        "latency_iterations": args.latency_iterations,
    }
    if args.model == "patchcore":
        config["model_parameters"] = {
            "backbone": args.patchcore_backbone,
            "layers": ["layer2", "layer3"],
            "coreset_sampling_ratio": args.patchcore_coreset_ratio,
            "num_neighbors": args.patchcore_neighbors,
            "max_epochs": 1,
        }
    elif args.model == "fastflow":
        config["model_parameters"] = {
            "backbone": args.fastflow_backbone,
            "flow_steps": args.fastflow_flow_steps,
            "hidden_ratio": args.fastflow_hidden_ratio,
            "max_epochs": args.fastflow_epochs,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
        }
    else:
        config["model_parameters"] = {
            "model_size": args.efficientad_model_size,
            "teacher_out_channels": args.efficientad_teacher_channels,
            "max_steps": args.efficientad_steps,
            "train_batch_size": 1,
            "learning_rate": args.efficientad_lr,
            "weight_decay": args.efficientad_weight_decay,
        }
    canonical = json.dumps(config, ensure_ascii=False, sort_keys=True)
    config["config_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return config


def build_model_and_engine_args(
    args: argparse.Namespace,
) -> tuple[torch.nn.Module, dict[str, Any], int]:
    size = (args.image_size, args.image_size)
    if args.model == "patchcore":
        pre_processor = Patchcore.configure_pre_processor(image_size=size)
        model = Patchcore(
            backbone=args.patchcore_backbone,
            layers=("layer2", "layer3"),
            pre_trained=True,
            coreset_sampling_ratio=args.patchcore_coreset_ratio,
            num_neighbors=args.patchcore_neighbors,
            pre_processor=pre_processor,
        )
        return model, {"max_epochs": 1}, 8

    if args.model == "fastflow":
        pre_processor = Fastflow.configure_pre_processor(image_size=size)
        model = Fastflow(
            backbone=args.fastflow_backbone,
            pre_trained=True,
            flow_steps=args.fastflow_flow_steps,
            hidden_ratio=args.fastflow_hidden_ratio,
            pre_processor=pre_processor,
        )
        return model, {"max_epochs": args.fastflow_epochs}, 8

    if not args.imagenette_dir.is_dir():
        raise FileNotFoundError(
            "EfficientAD requires an ImageNette training directory. "
            f"Not found: {args.imagenette_dir}"
        )
    # Anomalib normally stores EfficientAD teacher weights under AppData.
    # Keep the cache project-local so the experiment is portable and does not
    # require write access to a user-wide cache directory.
    import anomalib.models.image.efficient_ad.lightning_model as efficientad_lightning

    efficientad_cache = LOCAL_CACHE / "anomalib" / "pre_trained"
    efficientad_cache.mkdir(parents=True, exist_ok=True)
    efficientad_lightning.get_pretrained_weights_dir = lambda: efficientad_cache

    pre_processor = EfficientAd.configure_pre_processor(image_size=size)
    model = EfficientAd(
        imagenet_dir=args.imagenette_dir,
        teacher_out_channels=args.efficientad_teacher_channels,
        model_size=args.efficientad_model_size,
        lr=args.efficientad_lr,
        weight_decay=args.efficientad_weight_decay,
        pre_processor=pre_processor,
    )
    return model, {"max_steps": args.efficientad_steps, "max_epochs": -1}, 1


def create_checkpoint_callback(args: argparse.Namespace, run_dir: Path) -> ModelCheckpoint:
    common = {
        "dirpath": run_dir / "checkpoints",
        "filename": "{epoch:03d}-{step:07d}",
        "save_last": True,
        "save_top_k": 0,
        "enable_version_counter": True,
    }
    if args.model == "efficientad":
        return ModelCheckpoint(
            **common,
            every_n_train_steps=max(1, args.checkpoint_every_steps),
            save_on_train_epoch_end=False,
        )
    return ModelCheckpoint(
        **common,
        every_n_epochs=max(1, args.checkpoint_every_epochs),
        save_on_train_epoch_end=True,
    )


def tensor_item(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        value = value.item()
    return int(value)


def to_numpy_2d(value: torch.Tensor) -> np.ndarray:
    return np.squeeze(value.detach().cpu().float().numpy())


def save_heatmap(record: dict[str, Any], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    anomaly_map = to_numpy_2d(record["anomaly_map"])
    height, width = anomaly_map.shape[-2:]
    original = Image.open(record["image_path"]).convert("RGB").resize((width, height))
    normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    figure, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    axes[0].imshow(original)
    axes[0].set_title("Original")
    axes[1].imshow(original)
    axes[1].imshow(normalized, cmap="jet", alpha=0.5)
    axes[1].set_title("Anomaly map")
    predicted_mask = record.get("pred_mask")
    axes[2].imshow(
        to_numpy_2d(predicted_mask) if predicted_mask is not None else np.zeros_like(anomaly_map),
        cmap="gray",
    )
    axes[2].set_title("Predicted mask")
    for axis in axes:
        axis.axis("off")
    result = "OK" if record["gt_label"] == record["pred_label"] else "WRONG"
    figure.suptitle(
        f"[{result}] GT={record['gt_label']} Pred={record['pred_label']} "
        f"Score={record['pred_score']:.4f}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(figure)


def collect_predictions(prediction_batches: list[Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch in prediction_batches:
        scores = batch.pred_score
        for index in range(len(scores)):
            image_path = str(batch.image_path[index])
            anomaly_maps = getattr(batch, "anomaly_map", None)
            predicted_masks = getattr(batch, "pred_mask", None)
            records.append(
                {
                    "image_path": image_path,
                    "defect_type": Path(image_path).parent.name,
                    "gt_label": tensor_item(batch.gt_label[index]),
                    "pred_label": tensor_item(batch.pred_label[index]),
                    "pred_score": float(scores[index].detach().cpu().item()),
                    "anomaly_map": anomaly_maps[index] if anomaly_maps is not None else None,
                    "pred_mask": predicted_masks[index] if predicted_masks is not None else None,
                }
            )
    return records


def write_prediction_outputs(records: list[dict[str, Any]], run_dir: Path) -> dict[str, Any]:
    prediction_path = run_dir / "per_image_predictions.csv"
    serializable = []
    for record in records:
        serializable.append(
            {
                "image_path": record["image_path"],
                "defect_type": record["defect_type"],
                "gt_label": record["gt_label"],
                "pred_label": record["pred_label"],
                "pred_score": record["pred_score"],
                "correct": int(record["gt_label"] == record["pred_label"]),
            }
        )
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(serializable[0]))
        writer.writeheader()
        writer.writerows(serializable)

    good = [item for item in serializable if item["defect_type"] == "good"]
    defects = sorted({item["defect_type"] for item in serializable if item["defect_type"] != "good"})
    defect_rows = []
    for defect in defects:
        positives = [item for item in serializable if item["defect_type"] == defect]
        comparison = good + positives
        y_true = [item["gt_label"] for item in comparison]
        y_pred = [item["pred_label"] for item in comparison]
        y_score = [item["pred_score"] for item in comparison]
        detected = sum(item["pred_label"] == 1 for item in positives)
        defect_rows.append(
            {
                "defect_type": defect,
                "samples": len(positives),
                "image_AUROC_vs_good": roc_auc_score(y_true, y_score),
                "image_F1_vs_good": f1_score(y_true, y_pred, zero_division=0),
                "recall": detected / len(positives),
                "false_negatives": len(positives) - detected,
                "mean_anomaly_score": float(np.mean([item["pred_score"] for item in positives])),
                "min_anomaly_score": float(np.min([item["pred_score"] for item in positives])),
            }
        )
    defect_rows.sort(key=lambda item: (item["image_AUROC_vs_good"], item["recall"]))
    defect_path = run_dir / "defect_summary.csv"
    with defect_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(defect_rows[0]))
        writer.writeheader()
        writer.writerows(defect_rows)

    false_positives = sum(item["pred_label"] == 1 for item in good)
    false_negatives = sum(
        item["pred_label"] == 0 for item in serializable if item["gt_label"] == 1
    )
    return {
        "test_images": len(serializable),
        "normal_images": len(good),
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "hardest_defect": defect_rows[0] if defect_rows else None,
    }


def write_visualizations(
    records: list[dict[str, Any]],
    run_dir: Path,
    number: int,
) -> None:
    visual_dir = run_dir / "visualizations"
    failure_dir = run_dir / "failure_cases"
    visual_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    usable = [item for item in records if item["anomaly_map"] is not None]
    highest = sorted(usable, key=lambda item: item["pred_score"], reverse=True)[:number]
    for index, record in enumerate(highest):
        stem = f"{record['defect_type']}_{Path(record['image_path']).stem}"
        save_heatmap(record, visual_dir / f"{index:02d}_{stem}.png")

    mistakes = [item for item in usable if item["gt_label"] != item["pred_label"]]
    mistakes.sort(
        key=lambda item: item["pred_score"] if item["gt_label"] == 1 else -item["pred_score"]
    )
    for index, record in enumerate(mistakes[:12]):
        stem = f"{record['defect_type']}_{Path(record['image_path']).stem}"
        save_heatmap(record, failure_dir / f"{index:02d}_{stem}.png")


def model_thresholds(model: Any) -> dict[str, float | None]:
    post_processor = getattr(model, "post_processor", None)

    def read(name: str) -> float | None:
        value = getattr(post_processor, name, None)
        value = getattr(value, "value", value)
        return safe_float(value)

    return {
        "image_threshold": read("image_threshold"),
        "pixel_threshold": read("pixel_threshold"),
    }


def raw_latency(
    model: Any,
    datamodule: MVTecAD,
    iterations: int,
) -> dict[str, float | None]:
    if iterations <= 0:
        return {"raw_model_latency_ms": None, "raw_model_fps": None}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    sample = next(iter(datamodule.test_dataloader())).image[:1].to(device)
    try:
        with torch.inference_mode():
            for _ in range(5):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(iterations):
                model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - start) / iterations * 1000
        return {
            "raw_model_latency_ms": round(latency_ms, 3),
            "raw_model_fps": round(1000 / latency_ms, 2),
        }
    except Exception:
        return {"raw_model_latency_ms": None, "raw_model_fps": None}


def main() -> int:
    args = parse_args()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    args.dataset_root = args.dataset_root.absolute()
    args.imagenette_dir = args.imagenette_dir.absolute()
    args.results_root = args.results_root.absolute()
    categories = discover_categories(args.dataset_root)
    if args.category not in categories:
        raise ValueError(f"Unknown category {args.category!r}. Available: {', '.join(categories)}")

    run_dir = args.results_root / args.model / args.category / args.run_name
    config = experiment_config(args)
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"Run directory: {run_dir}")
    if args.dry_run:
        return 0

    existing_config_path = run_dir / "config.json"
    existing_status_path = run_dir / "status.json"
    if existing_config_path.exists():
        existing_config = json.loads(existing_config_path.read_text(encoding="utf-8"))
        if existing_config.get("config_sha256") != config["config_sha256"]:
            raise RuntimeError(
                f"Run directory contains a different configuration: {run_dir}. "
                "Use another --run-name."
            )
        if existing_status_path.exists():
            status = json.loads(existing_status_path.read_text(encoding="utf-8"))
            if status.get("status") == "complete":
                print("Matching completed run already exists; nothing to do.")
                return 0
            if not args.resume:
                raise RuntimeError(
                    "An incomplete run exists. Pass --resume or use another --run-name."
                )

    run_dir.mkdir(parents=True, exist_ok=True)
    logger = configure_logging(run_dir)
    atomic_write_json(existing_config_path, config)
    atomic_write_json(run_dir / "environment.json", environment_snapshot())
    started_at = now_iso()
    atomic_write_json(
        existing_status_path,
        {
            "status": "running",
            "started_at": started_at,
            "updated_at": started_at,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
        },
    )

    try:
        seed_everything(args.seed, workers=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        model, engine_arguments, train_batch_size = build_model_and_engine_args(args)
        datamodule = MVTecAD(
            root=args.dataset_root,
            category=args.category,
            train_batch_size=train_batch_size,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            val_split_mode=args.val_split_mode,
            val_split_ratio=args.val_split_ratio,
            seed=args.seed,
        )
        csv_logger = CSVLogger(
            save_dir=str(run_dir / "monitoring"),
            name="lightning",
            version=0,
        )
        checkpoint_callback = create_checkpoint_callback(args, run_dir)
        engine = Engine(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            default_root_dir=run_dir,
            logger=csv_logger,
            callbacks=[checkpoint_callback],
            log_every_n_steps=10,
            # Anomalib 2.4.2 dynamically replaces the Rich progress-bar class
            # during step-based training.  That local class cannot be pickled
            # by Lightning when EfficientAD writes a checkpoint.  Disabling
            # only the visual progress bar keeps training, CSV monitoring, and
            # resumable checkpoints intact.
            enable_progress_bar=args.model != "efficientad",
            **engine_arguments,
        )

        resume_path = run_dir / "checkpoints" / "last.ckpt"
        checkpoint = resume_path if args.resume and resume_path.exists() else None
        logger.info("Fit started | model=%s category=%s resume=%s", args.model, args.category, checkpoint)
        fit_start = time.perf_counter()
        engine.fit(model=model, datamodule=datamodule, ckpt_path=checkpoint)
        fit_seconds = time.perf_counter() - fit_start

        logger.info("Test started")
        test_start = time.perf_counter()
        test_result = engine.test(model=model, datamodule=datamodule)
        test_seconds = time.perf_counter() - test_start
        metrics = test_result[0] if test_result else {}

        logger.info("Prediction export started")
        predict_start = time.perf_counter()
        batches = engine.predict(model=model, datamodule=datamodule)
        prediction_seconds = time.perf_counter() - predict_start
        records = collect_predictions(batches or [])
        if not records:
            raise RuntimeError("Prediction produced no records.")
        error_summary = write_prediction_outputs(records, run_dir)
        write_visualizations(records, run_dir, args.num_visualize)

        final_checkpoint = run_dir / "model.ckpt"
        engine.trainer.save_checkpoint(final_checkpoint)
        checkpoint_mb = final_checkpoint.stat().st_size / (1024 * 1024)
        latency = raw_latency(model, datamodule, args.latency_iterations)
        peak_gpu_mb = None
        if torch.cuda.is_available():
            peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

        metric_values = {key: safe_float(value) for key, value in metrics.items()}
        result = {
            "protocol_version": "baseline_v1",
            "config_sha256": config["config_sha256"],
            "model": args.model,
            "category": args.category,
            "image_size": args.image_size,
            "seed": args.seed,
            "val_split_mode": args.val_split_mode,
            "val_split_ratio": args.val_split_ratio,
            "metrics": metric_values,
            "thresholds": model_thresholds(model),
            "fit_seconds": round(fit_seconds, 3),
            "test_seconds": round(test_seconds, 3),
            "prediction_seconds": round(prediction_seconds, 3),
            "test_images": len(records),
            "test_throughput_images_per_sec": round(len(records) / test_seconds, 3),
            "checkpoint_mb": round(checkpoint_mb, 3),
            "peak_gpu_memory_mb": round(peak_gpu_mb, 3) if peak_gpu_mb is not None else None,
            **latency,
            **error_summary,
            "completed_at": now_iso(),
        }
        atomic_write_json(run_dir / "metrics.json", result)
        atomic_write_json(
            existing_status_path,
            {
                "status": "complete",
                "started_at": started_at,
                "completed_at": result["completed_at"],
                "updated_at": result["completed_at"],
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "metrics_path": str(run_dir / "metrics.json"),
            },
        )
        logger.info(
            "Run complete | image_AUROC=%s pixel_AUROC=%s fit_seconds=%.1f",
            metric_values.get("image_AUROC"),
            metric_values.get("pixel_AUROC"),
            fit_seconds,
        )
        return 0
    except Exception as error:
        ended_at = now_iso()
        failure = {
            "status": "failed",
            "started_at": started_at,
            "failed_at": ended_at,
            "updated_at": ended_at,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(existing_status_path, failure)
        (run_dir / "failure_traceback.txt").write_text(failure["traceback"], encoding="utf-8")
        logger.exception("Run failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
