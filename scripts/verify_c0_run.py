"""独立核对 C0 训练产物与保存的验证集预测。"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import f1_score
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.c0 import C0ResNet18  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def run() -> dict:
    experiment = "c0_seed42_20261109"
    metrics = ROOT / "results" / "metrics"
    history = read_csv(metrics / f"{experiment}_history.csv")
    curves = read_json(metrics / f"{experiment}_curve_data.json")
    summary = read_json(metrics / f"{experiment}_summary.json")
    run_info = read_json(metrics / f"{experiment}_run.json")
    predictions = read_csv(metrics / f"{experiment}_val_predictions.csv")
    log_lines = (ROOT / "results" / "logs" / f"{experiment}_train.log").read_text(encoding="utf-8").splitlines()
    assert summary["status"] == "complete" and len(history) == summary["epochs_completed"]
    assert [int(row["epoch"]) for row in history] == curves["epochs"] == list(range(1, len(history) + 1))
    assert sum(row["is_best"] == "True" for row in history) >= 1
    best = max(history, key=lambda row: float(row["val_macro_f1"]))
    assert int(best["epoch"]) == summary["best_epoch"]
    assert float(best["val_macro_f1"]) == summary["best_val_metrics"]["macro_f1"]
    for field in ("train_loss", "val_loss", "train_accuracy", "val_accuracy", "val_macro_f1"):
        assert [float(row[field]) for row in history] == curves["series"][field]
    assert sum(" train_loss=" in line and " epoch=" in line for line in log_lines) == len(history)
    assert any("complete best_epoch=" in line for line in log_lines)
    for split, expected in run_info["split_sha256"].items():
        assert sha256(ROOT / "data" / "splits" / f"{split}.csv") == expected

    checkpoint = ROOT / "checkpoints" / f"{experiment}_best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert payload["metadata"]["split_sha256"] == run_info["split_sha256"]
    assert payload["metadata"]["class_map"] == run_info["class_map"]
    assert payload["metadata"]["best_epoch"] == summary["best_epoch"]
    model = C0ResNet18(pretrained=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    assert model.structure_info()["trainable_parameters"] == 3078

    assert len(predictions) == 270
    scores = torch.tensor([[float(row[f"logit_{i}"]) for i in range(6)] for row in predictions])
    labels = torch.tensor([int(row["true_label"]) for row in predictions])
    predicted = scores.argmax(dim=1)
    assert all(int(row["pred_label"]) == int(label) for row, label in zip(predictions, predicted))
    loss = float(nn.CrossEntropyLoss()(scores, labels))
    accuracy = float((predicted == labels).float().mean())
    macro_f1 = float(f1_score(labels.numpy(), predicted.numpy(), labels=list(range(6)), average="macro", zero_division=0))
    expected = summary["best_val_metrics"]
    assert abs(loss - expected["loss"]) < 1e-6
    assert abs(accuracy - expected["accuracy"]) < 1e-6
    assert abs(macro_f1 - expected["macro_f1"]) < 1e-6
    figure = ROOT / "results" / "figures" / f"{experiment}_curves.png"
    with Image.open(figure) as image:
        image.verify()
    return {
        "passed": True,
        "epochs": len(history),
        "best_epoch": summary["best_epoch"],
        "checkpoint_loaded": True,
        "split_hashes_unchanged": True,
        "history_curve_log_consistent": True,
        "validation_predictions": len(predictions),
        "recomputed_best_val_metrics": {"loss": loss, "accuracy": accuracy, "macro_f1": macro_f1},
        "figure_valid": True,
    }


if __name__ == "__main__":
    result = run()
    path = ROOT / "results" / "metrics" / "c0_seed42_20261109_verification.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
