import csv
import json
from pathlib import Path

from src.data.split import make_splits


CLASS_MAP = {"Cr": 0, "In": 1, "Pa": 2, "PS": 3, "RS": 4, "Sc": 5}


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_split_keeps_duplicate_candidates_together(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    images_root = data_root / "raw" / "NEU-CLS"
    metrics = tmp_path / "metrics"
    images_root.mkdir(parents=True)
    metrics.mkdir()
    inventory = []
    for class_name, label in CLASS_MAP.items():
        for number in range(1, 11):
            name = f"{class_name}_{number}.bmp"
            (images_root / name).write_bytes(b"fixture")
            inventory.append({
                "image_path": f"raw/NEU-CLS/{name}",
                "class_name": class_name,
                "label": label,
                "sha256": "same-Cr" if class_name == "Cr" and number in (1, 2) else name,
            })
    write_csv(metrics / "neu_cls_inventory.csv", list(inventory[0]), inventory)
    exact = [
        {"sha256": "same-Cr", "image_path": f"raw/NEU-CLS/Cr_{number}.bmp", "class_name": "Cr"}
        for number in (1, 2)
    ]
    write_csv(metrics / "neu_cls_exact_duplicates.csv", list(exact[0]), exact)
    near = [{"image_path_a": "raw/NEU-CLS/In_1.bmp", "image_path_b": "raw/NEU-CLS/In_2.bmp"}]
    write_csv(metrics / "neu_cls_near_duplicates.csv", list(near[0]), near)
    (metrics / "neu_cls_quality.json").write_text(
        json.dumps({"valid_images": 60, "invalid_images": 0, "exact_duplicate_images": 2, "near_duplicate_pairs": 1}),
        encoding="utf-8",
    )

    report = make_splits(
        data_root, metrics, data_root / "splits", CLASS_MAP,
        {"train": 0.7, "val": 0.15, "test": 0.15}, seed=42,
    )

    membership = {}
    for split in ("train", "val", "test"):
        with (data_root / "splits" / f"{split}.csv").open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                membership[row["image_path"]] = split
    assert len(membership) == 60
    assert membership["raw/NEU-CLS/Cr_1.bmp"] == membership["raw/NEU-CLS/Cr_2.bmp"]
    assert membership["raw/NEU-CLS/In_1.bmp"] == membership["raw/NEU-CLS/In_2.bmp"]
    assert report["near_candidate_cross_split_count"] == 0
    assert report["exact_duplicate_group_leak_count"] == 0
