"""冻结 ImageNet ResNet18 主干，仅训练六类全连接分类头。"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


class C0ResNet18(nn.Module):
    def __init__(self, num_classes: int = 6, pretrained: bool = True) -> None:
        super().__init__()
        if num_classes != 6:
            raise ValueError("C0 的类别数固定为 6")
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.network = resnet18(weights=weights)
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        self.in_features = self.network.fc.in_features
        self.network.fc = nn.Linear(self.in_features, num_classes)
        self._freeze_batchnorm_stats()

    def _freeze_batchnorm_stats(self) -> None:
        for module in self.network.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def train(self, mode: bool = True) -> C0ResNet18:
        super().train(mode)
        self._freeze_batchnorm_stats()
        return self

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.network(images)

    @property
    def trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.network.fc.parameters())

    def structure_info(self) -> dict[str, object]:
        trainable = sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in self.parameters())
        batchnorm_layers = sum(
            isinstance(module, nn.BatchNorm2d) for module in self.network.modules()
        )
        return {
            "architecture": "torchvision.models.resnet18",
            "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
            "classifier": f"Linear({self.in_features}, 6)",
            "total_parameters": total,
            "trainable_parameters": trainable,
            "frozen_parameters": total - trainable,
            "batchnorm_layers_kept_in_eval": batchnorm_layers,
        }
