"""按类别目标和重复候选组生成 NEU-CLS 固定划分。"""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np


SPLITS = ("train", "val", "test")
SPLIT_FIELDS = ("image_path", "class_name", "label")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def write_split_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=SPLIT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


class UnionFind:
    def __init__(self, items: list[str]) -> None:
        self.parent = {item: item for item in items}

    def find(self, item: str) -> str:
        if item not in self.parent:
            raise ValueError(f"重复候选引用未知图像: {item}")
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, first: str, second: str) -> None:
        a, b = self.find(first), self.find(second)
        if a != b:
            self.parent[b] = a


def make_splits(
    data_root: Path,
    metrics_dir: Path,
    splits_dir: Path,
    class_map: dict[str, int],
    ratios: dict[str, float],
    seed: int = 42,
) -> dict[str, object]:
    if set(ratios) != set(SPLITS) or abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ValueError("train/val/test 比例不完整或总和不为 1")
    quality = json.loads((metrics_dir / "neu_cls_quality.json").read_text(encoding="utf-8"))
    if quality["invalid_images"] != 0:
        raise RuntimeError("存在损坏或无法分类图像；划分前需人工处理")
    inventory = read_csv(metrics_dir / "neu_cls_inventory.csv")
    if len(inventory) != quality["valid_images"] or not inventory:
        raise RuntimeError("图像清单与质量报告不一致或为空")
    paths = [row["image_path"] for row in inventory]
    if len(paths) != len(set(paths)):
        raise RuntimeError("图像清单存在重复路径")
    for row in inventory:
        if row["class_name"] not in class_map or int(row["label"]) != class_map[row["class_name"]]:
            raise RuntimeError(f"类别映射不一致: {row['image_path']}")
        if not (data_root / row["image_path"]).is_file():
            raise FileNotFoundError(data_root / row["image_path"])

    exact = read_csv(metrics_dir / "neu_cls_exact_duplicates.csv")
    near = read_csv(metrics_dir / "neu_cls_near_duplicates.csv")
    if len(near) != quality["near_duplicate_pairs"] or len(exact) != quality["exact_duplicate_images"]:
        raise RuntimeError("重复候选清单与质量报告不一致")
    grouped = UnionFind(paths)
    first_by_sha: dict[str, str] = {}
    for row in inventory:
        sha = row["sha256"]
        if sha in first_by_sha:
            grouped.union(first_by_sha[sha], row["image_path"])
        else:
            first_by_sha[sha] = row["image_path"]
    for pair in near:
        grouped.union(pair["image_path_a"], pair["image_path_b"])

    members: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in inventory:
        members[grouped.find(row["image_path"])].append(row)
    groups = list(members.values())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)
    class_names = list(class_map)
    class_indexes = {name: index for index, name in enumerate(class_names)}
    total_by_class = Counter(row["class_name"] for row in inventory)
    target = np.array(
        [[total_by_class[name] * ratios[split] for name in class_names] for split in SPLITS],
        dtype=np.float64,
    )
    assigned = np.zeros_like(target)
    assigned_rows: dict[str, list[dict[str, str]]] = {split: [] for split in SPLITS}
    for group in groups:
        vector = np.zeros(len(class_names), dtype=np.float64)
        for row in group:
            vector[class_indexes[row["class_name"]]] += 1
        scores = []
        for split_index in range(3):
            candidate = assigned.copy()
            candidate[split_index] += vector
            class_error = np.sum((candidate - target) ** 2 / (target + 1))
            total_error = np.sum((candidate.sum(axis=1) - target.sum(axis=1)) ** 2 / (target.sum(axis=1) + 1))
            scores.append(float(class_error + total_error))
        best = min(scores)
        tied = [index for index, score in enumerate(scores) if abs(score - best) < 1e-10]
        chosen = rng.choice(tied)
        assigned[chosen] += vector
        assigned_rows[SPLITS[chosen]].extend(group)

    path_to_split = {
        row["image_path"]: split for split in SPLITS for row in assigned_rows[split]
    }
    if len(path_to_split) != len(inventory) or any(not assigned_rows[split] for split in SPLITS):
        raise RuntimeError("划分未覆盖所有图像或存在空集合")
    sha_splits: dict[str, set[str]] = defaultdict(set)
    for row in inventory:
        sha_splits[row["sha256"]].add(path_to_split[row["image_path"]])
    exact_leaks = sum(len(splits) > 1 for splits in sha_splits.values())
    near_leaks = sum(path_to_split[pair["image_path_a"]] != path_to_split[pair["image_path_b"]] for pair in near)
    if exact_leaks or near_leaks:
        raise RuntimeError(f"重复候选跨划分: exact={exact_leaks}, near={near_leaks}")

    splits_dir.mkdir(parents=True, exist_ok=True)
    split_counts = {}
    for split in SPLITS:
        selected = sorted(assigned_rows[split], key=lambda row: row["image_path"])
        write_split_csv(
            splits_dir / f"{split}.csv",
            [{field: row[field] for field in SPLIT_FIELDS} for row in selected],
        )
        split_counts[split] = {
            "total": len(selected),
            "by_class": {name: sum(row["class_name"] == name for row in selected) for name in class_names},
        }
    report: dict[str, object] = {
        "generated_at_Asia_Shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "seed": seed,
        "requested_ratios": ratios,
        "policy": "完全重复和所有 dHash 初筛近重复候选均分组，不删除图像；候选仍待人工复核",
        "group_count": len(groups),
        "largest_group_size": max(map(len, groups)),
        "split_counts": split_counts,
        "actual_ratios": {split: len(assigned_rows[split]) / len(inventory) for split in SPLITS},
        "path_overlap_count": 0,
        "exact_duplicate_group_leak_count": exact_leaks,
        "near_candidate_cross_split_count": near_leaks,
        "test_set_usage": "仅用于未来最终评价，不参与调参",
    }
    (metrics_dir / "neu_cls_split_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report
