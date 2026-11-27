"""只使用固定 train/val，先真实批次验证，再运行 C1。"""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import traceback
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import yaml
from torch import nn

from scripts.train_c0 import now, save_checkpoint, save_curves, sha256_file, write_history, write_json, write_val_predictions
from src.data.dataset import load_project_config, make_loader
from src.models.c0_training import evaluate, train_one_epoch
from src.models.c1 import C1ResNet18


def smoke_test(model, loader, device, initial_state):
    model.train()
    assert all(not module.training for module in model.modules() if isinstance(module, nn.BatchNorm2d))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable and all(name.startswith(("network.layer4.", "network.fc.")) for name in trainable)
    assert all(torch.equal(value.cpu(), initial_state[name]) for name, value in model.state_dict().items())
    images, labels = next(iter(loader))
    optimizer = torch.optim.Adam(model.trainable_parameters, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    logits = model(images.to(device))
    loss = nn.CrossEntropyLoss()(logits, labels.to(device))
    loss.backward()
    assert logits.shape == (len(labels), 6)
    assert all(parameter.grad is None for parameter in model.parameters() if not parameter.requires_grad)
    assert all(parameter.grad is not None for parameter in model.trainable_parameters)
    optimizer.step()
    updated = [name for name, value in model.state_dict().items() if not torch.equal(value.cpu(), initial_state[name])]
    assert any(name.startswith("network.layer4.") for name in updated)
    assert any(name.startswith("network.fc.") for name in updated)
    assert all(name in trainable for name in updated)
    model.eval()
    with torch.inference_mode():
        expected = model(images.to(device)).cpu()
    clone = C1ResNet18().to(device).eval()
    clone.load_state_dict(model.state_dict())
    with torch.inference_mode():
        actual = clone(images.to(device)).cpu()
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    model.load_state_dict(initial_state)
    return {"passed": True, "batch_shape": list(images.shape), "loss": loss.item(),
            "trainable_names": trainable, "updated_names": updated,
            "reload_max_abs_difference": (actual - expected).abs().max().item()}


def run(config_path):
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = config["experiment_id"]
    if not experiment or Path(experiment).name != experiment or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in experiment):
        raise ValueError("实验编号只能使用字母、数字、下划线和连字符")
    if (config["optimizer"], config["loss"], config["model_selection"], config["batchnorm_running_stats"]) != ("Adam", "CrossEntropyLoss", "val_macro_f1", "frozen"):
        raise ValueError("C1 配置不符合训练协议")
    if config["epochs"] < 1 or config["learning_rate"] <= 0 or config["early_stopping_patience"] < 1:
        raise ValueError("训练轮数、学习率和耐心值必须为正")
    directories = {key: ROOT / "results" / key for key in ("metrics", "logs", "figures")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)
    metrics = directories["metrics"]
    run_path = metrics / f"{experiment}_run.json"
    status_path = metrics / f"{experiment}_status.json"
    checkpoint = ROOT / "checkpoints" / f"{experiment}_best.pt"
    existing = list(metrics.glob(f"{experiment}_*")) + list(directories["logs"].glob(f"{experiment}_*")) + list(directories["figures"].glob(f"{experiment}_*"))
    if checkpoint.exists() or existing:
        raise FileExistsError("实验已有产物，请使用新的实验编号")
    # 排他创建用于并发互斥。失败保留状态，重试使用新实验编号。
    with run_path.open("x", encoding="utf-8") as output:
        json.dump({"status": "initializing", "experiment_id": experiment}, output)
    log_path = directories["logs"] / f"{experiment}_train.log"
    with log_path.open("x", encoding="utf-8") as log:
        def record(message):
            line = f"{now()} {message}"
            print(line, flush=True)
            log.write(line + "\n")
            log.flush()

        try:
            torch.set_num_threads(min(8, os.cpu_count() or 1))
            seed = config["seed"]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if config["device"] == "auto" else torch.device(config["device"])
            initial_path = ROOT / config["initial_checkpoint"]
            payload = torch.load(initial_path, map_location="cpu", weights_only=True)
            data_config, class_map = load_project_config(ROOT)
            split_hashes = {split: sha256_file(ROOT / "data/splits" / f"{split}.csv") for split in ("train", "val", "test")}
            initial_metadata = payload["metadata"]
            if initial_metadata["class_map"] != class_map or initial_metadata["preprocessing"] != data_config["preprocessing"] or initial_metadata["split_sha256"] != split_hashes:
                raise ValueError("C0 元数据与当前数据配置或划分不一致")
            model = C1ResNet18().to(device)
            model.load_state_dict(payload["model_state_dict"], strict=True)
            loader_args = dict(batch_size=config["batch_size"], num_workers=config["num_workers"], seed=seed)
            smoke_loader = make_loader(ROOT, "train", **loader_args)
            smoke = smoke_test(model, smoke_loader, device, payload["model_state_dict"])
            write_json(metrics / f"{experiment}_smoke.json", smoke)
            # Smoke Test 消耗过随机增强状态，正式训练重新固定随机种子和加载器。
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            train = make_loader(ROOT, "train", **loader_args)
            val = make_loader(ROOT, "val", **loader_args)
            metadata = {
                "experiment_id": experiment, "model_kind": "C1", "training_config": config,
                "class_map": class_map, "preprocessing": data_config["preprocessing"],
                "split_sha256": split_hashes, "initial_checkpoint_sha256": sha256_file(initial_path),
                "model_structure": model.structure_info(), "device": str(device),
                "python_version": platform.python_version(),
                "dependency_versions": {name: version(name) for name in ("torch", "torchvision", "numpy", "scikit-learn", "Pillow", "PyYAML")},
                "started_at": now(),
            }
            write_json(run_path, metadata)
            record(f"start device={device} train={len(train.dataset)} val={len(val.dataset)} smoke=passed")
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.trainable_parameters, lr=config["learning_rate"])
            history = []
            best_f1 = -1.0
            stale = 0
            started = time.perf_counter()
            for epoch in range(1, config["epochs"] + 1):
                epoch_started = time.perf_counter()
                train_loss, train_accuracy = train_one_epoch(model, train, criterion, optimizer, device)
                scores, logits, labels = evaluate(model, val, criterion, device)
                improved = scores["macro_f1"] > best_f1 + 1e-12
                if improved:
                    best_f1, best_epoch, best_metrics = scores["macro_f1"], epoch, scores.copy()
                    best_logits, best_labels = logits.clone(), labels.clone()
                    save_checkpoint(checkpoint, model, {**metadata, "best_epoch": epoch, "best_val_metrics": scores})
                    stale = 0
                else:
                    stale += 1
                row = {"epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
                       "val_loss": scores["loss"], "val_accuracy": scores["accuracy"], "val_macro_f1": scores["macro_f1"],
                       "epoch_seconds": time.perf_counter() - epoch_started, "is_best": improved}
                history.append(row)
                write_history(metrics / f"{experiment}_history.csv", history)
                write_json(status_path, {"status": "running", "last_completed_epoch": epoch, "best_epoch": best_epoch})
                record(json.dumps(row))
                if stale >= config["early_stopping_patience"]:
                    break
            elapsed = time.perf_counter() - started
            reloaded = C1ResNet18().to(device)
            reloaded.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
            scores, logits, labels = evaluate(reloaded, val, criterion, device)
            assert torch.equal(labels, best_labels) and torch.allclose(logits, best_logits, atol=1e-6, rtol=1e-5)
            assert split_hashes == {split: sha256_file(ROOT / "data/splits" / f"{split}.csv") for split in split_hashes}
            write_val_predictions(metrics / f"{experiment}_val_predictions.csv", val.dataset.rows, logits, labels)
            save_curves(metrics / f"{experiment}_history.csv", directories["figures"] / f"{experiment}_curves.png", metrics / f"{experiment}_curve_data.json")
            summary = {"experiment_id": experiment, "status": "complete", "best_epoch": best_epoch,
                       "epochs_completed": len(history), "best_val_metrics": best_metrics,
                       "training_total_seconds": elapsed, "device": str(device),
                       "checkpoint_reload_consistent": True, "reload_max_abs_logit_difference": (logits-best_logits).abs().max().item(),
                       "checkpoint": checkpoint.relative_to(ROOT).as_posix(), "test_evaluated": False}
            write_json(metrics / f"{experiment}_summary.json", summary)
            write_json(status_path, {"status": "complete", **summary})
            record("completed")
        except BaseException as error:
            write_json(status_path, {"status": "failed", "error": str(error), "traceback": traceback.format_exc()})
            record(traceback.format_exc())
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/c1.yaml")
    run(parser.parse_args().config)
