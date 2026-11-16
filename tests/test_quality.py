import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.quality import run_quality


CLASS_MAP = {"Cr": 0, "In": 1, "Pa": 2, "PS": 3, "RS": 4, "Sc": 5}


def test_quality_reports_corruption_and_exact_duplicates(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_root = data_root / "raw" / "NEU-CLS"
    images_root.mkdir(parents=True)
    pixels = np.arange(400, dtype=np.uint8).reshape(20, 20)
    Image.fromarray(pixels, mode="L").save(images_root / "Cr_1.bmp")
    (images_root / "Cr_2.bmp").write_bytes((images_root / "Cr_1.bmp").read_bytes())
    Image.fromarray(np.flipud(pixels), mode="L").save(images_root / "In_1.bmp")
    (images_root / "Pa_1.bmp").write_bytes(b"not an image")
    (images_root / "Thumbs.db").write_bytes(b"ignored")

    metrics = tmp_path / "metrics"
    figures = tmp_path / "figures"
    summary = run_quality(data_root, images_root, metrics, figures, CLASS_MAP)

    assert summary["total_image_candidates"] == 4
    assert summary["valid_images"] == 3
    assert summary["invalid_images"] == 1
    assert summary["class_counts"]["Cr"] == 2
    assert summary["exact_duplicate_groups"] == 1
    assert summary["exact_duplicate_images"] == 2
    with (metrics / "neu_cls_invalid_images.csv").open(encoding="utf-8-sig", newline="") as source:
        invalid = list(csv.DictReader(source))
    assert invalid[0]["image_path"] == "raw/NEU-CLS/Pa_1.bmp"
    assert (figures / "neu_cls_class_distribution.png").is_file()
    assert (images_root / "Cr_2.bmp").is_file()  # 检查不删除样本
