"""C0 的训练与验证计算；验证函数只接收验证集 DataLoader。"""

from __future__ import annotations

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.models.c0 import C0ResNet18


def train_one_epoch(
    model: C0ResNet18,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_count += batch_size
    return total_loss / total_count, total_correct / total_count


def evaluate(
    model: C0ResNet18,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_batches = []
    label_batches = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += float(loss.item()) * labels.size(0)
            total_count += labels.size(0)
            logits_batches.append(logits.cpu())
            label_batches.append(labels.cpu())
    all_logits = torch.cat(logits_batches)
    all_labels = torch.cat(label_batches)
    predictions = all_logits.argmax(dim=1)
    accuracy = float((predictions == all_labels).float().mean().item())
    macro_f1 = float(
        f1_score(all_labels.numpy(), predictions.numpy(), labels=list(range(6)), average="macro", zero_division=0)
    )
    return {"loss": total_loss / total_count, "accuracy": accuracy, "macro_f1": macro_f1}, all_logits, all_labels
