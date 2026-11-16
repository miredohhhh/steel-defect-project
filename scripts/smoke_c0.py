"""在真实训练 batch 上验证 C0 的梯度、冻结状态和权重保存/加载。"""

from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
sys.path.insert(0, str(ROOT))

from src.data.dataset import make_loader  # noqa: E402
from src.models.c0 import C0ResNet18  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run() -> dict[str, object]:
    config = yaml.safe_load((ROOT / "configs" / "c0.yaml").read_text(encoding="utf-8"))
    experiment_id = config["experiment_id"]
    if config["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C0 Smoke Test 要求可用的 CUDA GPU")
    if config["batch_size"] != 32 or config["seed"] != 42:
        raise ValueError("Smoke Test 应使用固定的 32 张训练图片和 seed=42")
    checkpoint = ROOT / "checkpoints" / f"{experiment_id}_smoke.pt"
    report_path = ROOT / "results" / "metrics" / f"{experiment_id}_smoke.json"
    if checkpoint.exists() or report_path.exists():
        raise FileExistsError("该实验编号已有 Smoke Test 记录，拒绝覆盖")

    set_seed(config["seed"])
    torch.set_num_threads(min(8, os.cpu_count() or 1))
    loader = make_loader(ROOT, "train", batch_size=32, num_workers=0, seed=config["seed"])
    images, labels = next(iter(loader))
    if images.shape != (32, 3, 224, 224):
        raise AssertionError(f"训练 batch 尺寸错误: {tuple(images.shape)}")
    images = images.to("cuda", non_blocking=True)
    labels = labels.to("cuda", non_blocking=True)
    model = C0ResNet18(pretrained=True).to("cuda")
    optimizer = torch.optim.Adam(model.trainable_parameters, lr=config["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    frozen_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters() if not name.startswith("network.fc.")
    }
    fc_before = [parameter.detach().clone() for parameter in model.trainable_parameters]
    bn_before = {
        name: buffer.detach().clone()
        for name, buffer in model.named_buffers()
        if name.endswith(("running_mean", "running_var", "num_batches_tracked"))
    }
    model.train()
    bn_layers = [module for module in model.modules() if isinstance(module, nn.BatchNorm2d)]
    if not bn_layers or any(layer.training for layer in bn_layers):
        raise AssertionError("冻结主干的 BatchNorm 未保持 eval 模式")
    optimizer.zero_grad(set_to_none=True)
    logits = model(images)
    if tuple(logits.shape) != (32, 6):
        raise AssertionError(f"分类输出尺寸错误: {tuple(logits.shape)}")
    loss = criterion(logits, labels)
    if not torch.isfinite(loss):
        raise AssertionError("Smoke Test 损失不是有限数")
    loss.backward()
    gradients = [parameter.grad for parameter in model.trainable_parameters]
    if any(gradient is None or not torch.isfinite(gradient).all() for gradient in gradients):
        raise AssertionError("fc 梯度缺失或非有限数")
    gradient_norm = sum(float(gradient.norm().item()) for gradient in gradients if gradient is not None)
    if gradient_norm <= 0:
        raise AssertionError("fc 梯度全为零")
    if any(parameter.grad is not None for name, parameter in model.named_parameters() if name in frozen_before):
        raise AssertionError("主干参数出现梯度")
    optimizer.step()
    if not any(not torch.equal(before, after) for before, after in zip(fc_before, model.trainable_parameters)):
        raise AssertionError("optimizer.step() 未更新 fc 参数")
    if any(not torch.equal(before, dict(model.named_parameters())[name]) for name, before in frozen_before.items()):
        raise AssertionError("冻结的主干参数发生变化")
    if any(not torch.equal(before, dict(model.named_buffers())[name]) for name, before in bn_before.items()):
        raise AssertionError("BatchNorm 运行统计量发生变化")

    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".pt", dir=ROOT / ".cache", delete=False) as handle:
        temporary = Path(handle.name)
    torch.save({"model_state_dict": model.state_dict(), "experiment_id": experiment_id}, temporary)
    payload = torch.load(temporary, map_location="cpu", weights_only=True)
    reloaded = C0ResNet18(pretrained=False).to("cuda")
    reloaded.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    reloaded.eval()
    with torch.inference_mode():
        expected = model(images)
        actual = reloaded(images)
    max_abs_difference = float((expected - actual).abs().max().item())
    if not torch.allclose(expected, actual, rtol=1e-5, atol=1e-6):
        raise AssertionError(f"保存/加载后输出不一致: max abs diff={max_abs_difference}")
    temporary.replace(checkpoint)

    report: dict[str, object] = {
        "experiment_id": experiment_id,
        "checked_at_Asia_Shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "passed": True,
        "device": str(images.device),
        "gpu_name": torch.cuda.get_device_name(0),
        "batch_shape": list(images.shape),
        "output_shape": list(logits.shape),
        "cross_entropy_loss": float(loss.item()),
        "fc_gradient_norm_sum": gradient_norm,
        "fc_updated": True,
        "backbone_unchanged": True,
        "batchnorm_stats_unchanged": True,
        "checkpoint_reload_consistent": True,
        "reload_max_abs_logit_difference": max_abs_difference,
        "model_structure": model.structure_info(),
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    try:
        result = run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except BaseException:
        failure = ROOT / "results" / "metrics" / "c0_seed42_20261109_smoke_failure.json"
        if not failure.exists():
            failure.write_text(json.dumps({"passed": False, "traceback": traceback.format_exc()}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise
