"""验证 NEU-CLS 三套 DataLoader，并保存训练增强示例。"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
(ROOT / ".cache" / "matplotlib").mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from src.data.dataset import NEUCLSDataset, build_transforms, load_project_config, make_loader  # noqa: E402


def save_augmentation_examples() -> None:
    config, class_map = load_project_config(ROOT)
    train = NEUCLSDataset(ROOT, "train")
    baseline_transform = build_transforms("val", config)
    mean = np.asarray(config["preprocessing"]["mean"], dtype=np.float32)[:, None, None]
    std = np.asarray(config["preprocessing"]["std"], dtype=np.float32)[:, None, None]
    fig, axes = plt.subplots(6, 4, figsize=(10, 14), constrained_layout=True)
    torch.manual_seed(config["seed"])
    first_by_class = {}
    for index, row in enumerate(train.rows):
        first_by_class.setdefault(row["class_name"], index)
    for row_index, class_name in enumerate(class_map):
        index = first_by_class[class_name]
        for column_index in range(4):
            if column_index == 0:
                path = ROOT / config["data_root"] / train.rows[index]["image_path"]
                with Image.open(path) as image:
                    tensor = baseline_transform(image.convert("RGB"))
            else:
                tensor, _ = train[index]
            pixels = np.clip(tensor.numpy() * std + mean, 0, 1).transpose(1, 2, 0)
            axes[row_index, column_index].imshow(pixels)
            axes[row_index, column_index].axis("off")
            axes[row_index, column_index].set_title(
                f"{class_name}: sample" if column_index == 0 else f"{class_name}: train view {column_index}",
                fontsize=9,
            )
    output = ROOT / "results" / "figures" / "neu_cls_train_augmentation.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main() -> None:
    report = {
        "generated_at_Asia_Shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "splits": {},
        "validation_and_test_are_deterministic": False,
    }
    for split in ("train", "val", "test"):
        loader = make_loader(ROOT, split, batch_size=32, num_workers=0, seed=42)
        images, labels = next(iter(loader))
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 224, 224):
            raise AssertionError(f"{split}: 张量尺寸异常 {tuple(images.shape)}")
        if labels.min().item() < 0 or labels.max().item() > 5:
            raise AssertionError(f"{split}: 标签越界")
        if not torch.isfinite(images).all():
            raise AssertionError(f"{split}: 张量含 NaN 或无穷值")
        all_labels = [int(row["label"]) for row in loader.dataset.rows]
        if min(all_labels) < 0 or max(all_labels) > 5 or set(all_labels) != set(range(6)):
            raise AssertionError(f"{split}: 全部标签不覆盖 0..5")
        report["splits"][split] = {
            "dataset_size": len(loader.dataset),
            "batch_shape": list(images.shape),
            "label_min_in_batch": int(labels.min()),
            "label_max_in_batch": int(labels.max()),
            "label_min_all": min(all_labels),
            "label_max_all": max(all_labels),
            "image_dtype": str(images.dtype),
        }
    val = NEUCLSDataset(ROOT, "val")
    test = NEUCLSDataset(ROOT, "test")
    deterministic = torch.equal(val[0][0], val[0][0]) and torch.equal(test[0][0], test[0][0])
    if not deterministic:
        raise AssertionError("验证或测试预处理不是确定性的")
    report["validation_and_test_are_deterministic"] = True
    save_augmentation_examples()
    report["augmentation_example"] = "results/figures/neu_cls_train_augmentation.png"
    output = ROOT / "results" / "metrics" / "neu_cls_loading_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
