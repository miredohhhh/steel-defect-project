import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from PIL import Image
from torchvision.transforms import RandomHorizontalFlip

from src.data.dataset import NEUCLSDataset, make_loader


CLASS_MAP = {"Cr": 0, "In": 1, "Pa": 2, "PS": 3, "RS": 4, "Sc": 5}


@pytest.fixture
def project(tmp_path: Path) -> Path:
    configs = tmp_path / "configs"
    images = tmp_path / "data" / "raw" / "NEU-CLS"
    splits = tmp_path / "data" / "splits"
    configs.mkdir()
    images.mkdir(parents=True)
    splits.mkdir()
    config = {
        "data_root": "data",
        "preprocessing": {
            "channels": 3,
            "image_size": 224,
            "resnet18_weights": "IMAGENET1K_V1",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "train_horizontal_flip_probability": 0.5,
        },
    }
    (configs / "data.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (configs / "class_map.json").write_text(json.dumps(CLASS_MAP), encoding="utf-8")
    rows = []
    for class_name, label in CLASS_MAP.items():
        name = f"{class_name}_1.bmp"
        pixels = np.full((20, 20), label * 30, dtype=np.uint8)
        Image.fromarray(pixels, mode="L").save(images / name)
        rows.append({"image_path": f"raw/NEU-CLS/{name}", "class_name": class_name, "label": label})
    for split in ("train", "val", "test"):
        with (splits / f"{split}.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=("image_path", "class_name", "label"))
            writer.writeheader()
            writer.writerows(rows)
    return tmp_path


def test_dataset_shapes_labels_and_deterministic_eval(project: Path) -> None:
    train = NEUCLSDataset(project, "train")
    val = NEUCLSDataset(project, "val")
    test = NEUCLSDataset(project, "test")
    assert any(isinstance(op, RandomHorizontalFlip) for op in train.transform.transforms)
    assert not any(isinstance(op, RandomHorizontalFlip) for op in val.transform.transforms)
    assert not any(isinstance(op, RandomHorizontalFlip) for op in test.transform.transforms)
    image, label = train[3]
    assert image.shape == (3, 224, 224)
    assert image.dtype == torch.float32
    assert label == 3
    assert torch.equal(val[2][0], val[2][0])
    assert torch.equal(test[2][0], test[2][0])
    images, labels = next(iter(make_loader(project, "val", batch_size=4)))
    assert images.shape == (4, 3, 224, 224)
    assert labels.tolist() == [0, 1, 2, 3]


def test_dataset_rejects_path_traversal(project: Path) -> None:
    manifest = project / "data" / "splits" / "val.csv"
    manifest.write_text("image_path,class_name,label\n../outside.bmp,Cr,0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="非法图像路径"):
        NEUCLSDataset(project, "val")
