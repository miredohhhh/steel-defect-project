"""NEU-CLS Dataset、预处理和 DataLoader。"""

from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath
from typing import Literal

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights


SplitName = Literal["train", "val", "test"]


def load_project_config(project_root: Path) -> tuple[dict, dict[str, int]]:
    config = yaml.safe_load((project_root / "configs" / "data.yaml").read_text(encoding="utf-8"))
    class_map = json.loads((project_root / "configs" / "class_map.json").read_text(encoding="utf-8"))
    return config, class_map


def build_transforms(split: SplitName, config: dict):
    if split not in ("train", "val", "test"):
        raise ValueError(f"未知划分: {split}")
    settings = config["preprocessing"]
    if settings["channels"] != 3 or settings["image_size"] != 224:
        raise ValueError("当前 ResNet18 数据配置要求 3 通道、224×224")
    weights = ResNet18_Weights[settings["resnet18_weights"]]
    preset = weights.transforms()
    mean, std = list(preset.mean), list(preset.std)
    if mean != settings["mean"] or std != settings["std"]:
        raise ValueError("data.yaml 的归一化参数与 ResNet18 权重不一致")
    operations = [transforms.Resize((224, 224))]
    if split == "train":
        operations.append(transforms.RandomHorizontalFlip(p=settings["train_horizontal_flip_probability"]))
    operations.extend((transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)))
    return transforms.Compose(operations)


class NEUCLSDataset(Dataset):
    def __init__(self, project_root: Path, split: SplitName) -> None:
        self.project_root = project_root.resolve()
        self.split = split
        config, self.class_map = load_project_config(self.project_root)
        self.data_root = (self.project_root / config["data_root"]).resolve()
        self.transform = build_transforms(split, config)
        manifest = self.data_root / "splits" / f"{split}.csv"
        with manifest.open(newline="", encoding="utf-8-sig") as source:
            self.rows = list(csv.DictReader(source))
        if not self.rows:
            raise ValueError(f"空数据划分: {manifest}")
        for row in self.rows:
            relative = PurePosixPath(row["image_path"])
            if relative.is_absolute() or ".." in relative.parts or "\\" in row["image_path"]:
                raise ValueError(f"非法图像路径: {row['image_path']}")
            image_path = (self.data_root / Path(*relative.parts)).resolve()
            if not image_path.is_relative_to(self.data_root) or not image_path.is_file():
                raise FileNotFoundError(image_path)
            if row["class_name"] not in self.class_map or int(row["label"]) != self.class_map[row["class_name"]]:
                raise ValueError(f"类别映射不一致: {row}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.rows[index]
        path = self.data_root / Path(*PurePosixPath(row["image_path"]).parts)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            tensor = self.transform(rgb)
        return tensor, int(row["label"])


def make_loader(
    project_root: Path,
    split: SplitName,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
) -> DataLoader:
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size 必须为正，num_workers 不能为负")
    dataset = NEUCLSDataset(project_root, split)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )
