"""运行固定实验 c0_seed42_20261109；只使用 train 与 val。"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import sys
import tempfile
import time
import traceback
from datetime import datetime
from importlib.metadata import version
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch import nn  # noqa: E402

from src.data.dataset import load_project_config, make_loader  # noqa: E402
from src.models.c0 import C0ResNet18  # noqa: E402
from src.models.c0_training import evaluate, train_one_epoch  # noqa: E402


HISTORY_FIELDS = (
    "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy",
    "val_macro_f1", "epoch_seconds", "is_best",
)


def now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_history(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(path: Path, model: C0ResNet18, metadata: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, temporary)
    temporary.replace(path)


def save_curves(history_path: Path, image_path: Path, data_path: Path) -> None:
    with history_path.open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    epochs = [int(row["epoch"]) for row in rows]
    series = {
        key: [float(row[key]) for row in rows]
        for key in ("train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_macro_f1")
    }
    write_json(data_path, {"epochs": epochs, "series": series, "source_history": history_path.name})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].plot(epochs, series["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, series["val_loss"], marker="o", label="val")
    axes[0].set(xlabel="Epoch", ylabel="CrossEntropyLoss", title="Loss")
    axes[0].legend()
    axes[1].plot(epochs, series["train_accuracy"], marker="o", label="train")
    axes[1].plot(epochs, series["val_accuracy"], marker="o", label="val")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy", ylim=(0, 1))
    axes[1].legend()
    axes[2].plot(epochs, series["val_macro_f1"], marker="o", color="#356c9a")
    axes[2].set(xlabel="Epoch", ylabel="Macro-F1", title="Validation Macro-F1", ylim=(0, 1))
    fig.savefig(image_path, dpi=150)
    plt.close(fig)


def write_val_predictions(path: Path, rows: list[dict[str, str]], logits: torch.Tensor, labels: torch.Tensor) -> None:
    if len(rows) != logits.size(0) or len(rows) != labels.size(0):
        raise AssertionError("验证集清单与推理输出数量不一致")
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        fields = ("image_path", "class_name", "true_label", "pred_label") + tuple(f"logit_{i}" for i in range(6))
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for row, score, label in zip(rows, logits, labels):
            record = {
                "image_path": row["image_path"],
                "class_name": row["class_name"],
                "true_label": int(label),
                "pred_label": int(score.argmax()),
            }
            record.update({f"logit_{i}": float(value) for i, value in enumerate(score)})
            writer.writerow(record)


def run() -> dict[str, object]:
    config = yaml.safe_load((ROOT / "configs" / "c0.yaml").read_text(encoding="utf-8"))
    experiment_id = config["experiment_id"]
    if experiment_id != "c0_seed42_20261109":
        raise ValueError("实验编号不符合本轮约定")
    if config["seed"] != 42 or config["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("本轮要求 seed=42 且 CUDA 可用")
    if (config["optimizer"], config["loss"], config["model_selection"]) != (
        "Adam", "CrossEntropyLoss", "val_macro_f1"
    ):
        raise ValueError("C0 训练策略配置不符合约定")
    smoke = json.loads((ROOT / "results" / "metrics" / f"{experiment_id}_smoke.json").read_text(encoding="utf-8"))
    if smoke.get("passed") is not True:
        raise RuntimeError("Smoke Test 未通过，拒绝开始正式训练")

    checkpoints = ROOT / "checkpoints"
    metrics = ROOT / "results" / "metrics"
    logs = ROOT / "results" / "logs"
    figures = ROOT / "results" / "figures"
    checkpoint = checkpoints / f"{experiment_id}_best.pt"
    history_path = metrics / f"{experiment_id}_history.csv"
    run_path = metrics / f"{experiment_id}_run.json"
    status_path = metrics / f"{experiment_id}_status.json"
    summary_path = metrics / f"{experiment_id}_summary.json"
    curve_data_path = metrics / f"{experiment_id}_curve_data.json"
    predictions_path = metrics / f"{experiment_id}_val_predictions.csv"
    figure_path = figures / f"{experiment_id}_curves.png"
    log_path = logs / f"{experiment_id}_train.log"
    outputs = (
        checkpoint, history_path, run_path, status_path, summary_path,
        curve_data_path, predictions_path, figure_path, log_path,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"该实验编号已有训练文件，拒绝覆盖: {existing}")

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    device = torch.device("cuda")
    data_config, class_map = load_project_config(ROOT)
    if class_map != {"Cr": 0, "In": 1, "Pa": 2, "PS": 3, "RS": 4, "Sc": 5}:
        raise ValueError("类别映射与训练前验收不一致")
    if data_config["preprocessing"]["resnet18_weights"] != config["pretrained_weights"]:
        raise ValueError("C0 权重与数据预处理配置不一致")
    split_hashes = {
        split: sha256_file(ROOT / "data" / "splits" / f"{split}.csv")
        for split in ("train", "val", "test")
    }
    train_loader = make_loader(ROOT, "train", batch_size=config["batch_size"], num_workers=config["num_workers"], seed=42)
    val_loader = make_loader(ROOT, "val", batch_size=config["batch_size"], num_workers=config["num_workers"], seed=42)
    if len(train_loader.dataset) != 1260 or len(val_loader.dataset) != 270:
        raise ValueError("训练/验证划分数量与验收记录不一致")
    model = C0ResNet18(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.trainable_parameters, lr=config["learning_rate"])
    versions = {name: version(name) for name in ("torch", "torchvision", "numpy", "scikit-learn", "Pillow", "PyYAML")}
    metadata: dict[str, object] = {
        "experiment_id": experiment_id,
        "model_structure": model.structure_info(),
        "class_map": class_map,
        "preprocessing": data_config["preprocessing"],
        "seed": config["seed"],
        "split_sha256": split_hashes,
        "training_config": config,
        "dependency_versions": versions,
        "python_version": platform.python_version(),
        "torch_cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
    }
    write_json(run_path, {**metadata, "started_at_Asia_Shanghai": now()})
    write_json(status_path, {"experiment_id": experiment_id, "status": "running", "last_completed_epoch": 0})

    history: list[dict[str, object]] = []
    best_f1 = -1.0
    best_epoch = 0
    best_metrics: dict[str, float] = {}
    best_logits = None
    best_labels = None
    no_improvement = 0
    started = time.perf_counter()
    with log_path.open("x", encoding="utf-8") as log_file:
        def log(message: str) -> None:
            line = f"{now()} {message}"
            print(line, flush=True)
            log_file.write(line + "\n")
            log_file.flush()

        try:
            log(f"start experiment={experiment_id} device={device} train={len(train_loader.dataset)} val={len(val_loader.dataset)}")
            for epoch in range(1, config["epochs"] + 1):
                epoch_started = time.perf_counter()
                train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
                val_metrics, val_logits, val_labels = evaluate(model, val_loader, criterion, device)
                improved = val_metrics["macro_f1"] > best_f1 + 1e-12
                if improved:
                    best_f1 = val_metrics["macro_f1"]
                    best_epoch = epoch
                    best_metrics = val_metrics.copy()
                    best_logits = val_logits.clone()
                    best_labels = val_labels.clone()
                    no_improvement = 0
                    save_checkpoint(checkpoint, model, {**metadata, "best_epoch": epoch, "best_val_metrics": val_metrics})
                else:
                    no_improvement += 1
                elapsed = time.perf_counter() - epoch_started
                row: dict[str, object] = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                    "epoch_seconds": elapsed,
                    "is_best": improved,
                }
                history.append(row)
                write_history(history_path, history)
                write_json(status_path, {
                    "experiment_id": experiment_id, "status": "running", "last_completed_epoch": epoch,
                    "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
                })
                log(
                    f"epoch={epoch} train_loss={train_loss:.6f} train_accuracy={train_accuracy:.6f} "
                    f"val_loss={val_metrics['loss']:.6f} val_accuracy={val_metrics['accuracy']:.6f} "
                    f"val_macro_f1={val_metrics['macro_f1']:.6f} seconds={elapsed:.3f} best={improved}"
                )
                if no_improvement >= config["early_stopping_patience"]:
                    log(f"early_stopping epoch={epoch} patience={config['early_stopping_patience']}")
                    break
            total_seconds = time.perf_counter() - started
            if best_logits is None or best_labels is None:
                raise RuntimeError("没有可用的最佳模型")
            save_curves(history_path, figure_path, curve_data_path)

            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if payload["metadata"]["split_sha256"] != split_hashes:
                raise AssertionError("最佳权重的划分版本与本次运行不一致")
            reloaded = C0ResNet18(pretrained=False).to(device)
            reloaded.load_state_dict(payload["model_state_dict"], strict=True)
            reloaded_metrics, reloaded_logits, reloaded_labels = evaluate(reloaded, val_loader, criterion, device)
            max_abs_diff = float((reloaded_logits - best_logits).abs().max().item())
            consistent = (
                torch.equal(reloaded_labels, best_labels)
                and torch.allclose(reloaded_logits, best_logits, rtol=1e-5, atol=1e-6)
                and all(abs(reloaded_metrics[key] - best_metrics[key]) < 1e-6 for key in best_metrics)
            )
            if not consistent:
                raise AssertionError(f"最佳权重重新加载后的验证输出不一致: max_diff={max_abs_diff}")
            write_val_predictions(predictions_path, val_loader.dataset.rows, reloaded_logits, reloaded_labels)
            summary = {
                "experiment_id": experiment_id,
                "status": "complete",
                "epochs_completed": len(history),
                "early_stopped": len(history) < config["epochs"],
                "best_epoch": best_epoch,
                "best_val_metrics": best_metrics,
                "training_total_seconds": total_seconds,
                "checkpoint_reload_consistent": True,
                "reload_max_abs_logit_difference": max_abs_diff,
                "reloaded_val_metrics": reloaded_metrics,
                "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
                "history": history_path.relative_to(ROOT).as_posix(),
                "curves": figure_path.relative_to(ROOT).as_posix(),
            }
            write_json(summary_path, summary)
            write_json(status_path, {"experiment_id": experiment_id, "status": "complete", "last_completed_epoch": len(history)})
            log(
                f"complete best_epoch={best_epoch} best_val_accuracy={best_metrics['accuracy']:.6f} "
                f"best_val_macro_f1={best_metrics['macro_f1']:.6f} total_seconds={total_seconds:.3f}"
            )
            return summary
        except BaseException:
            failure = traceback.format_exc()
            write_json(status_path, {
                "experiment_id": experiment_id, "status": "failed", "last_completed_epoch": len(history),
                "best_epoch": best_epoch, "traceback": failure,
            })
            log(f"failed last_completed_epoch={len(history)}\n{failure}")
            raise


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
