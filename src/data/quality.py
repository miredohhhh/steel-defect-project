"""NEU-CLS 图像质量检查。只生成记录，不修改或删除原始图像。"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"))
(Path(os.environ["MPLCONFIGDIR"])).mkdir(parents=True, exist_ok=True)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
INVENTORY_FIELDS = (
    "image_path", "class_name", "label", "width", "height", "format", "mode",
    "channels", "sha256", "dhash64",
)


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def dhash64(image: Image.Image) -> int:
    pixels = np.asarray(
        ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.flat:
        value = (value << 1) | int(bit)
    return value


def class_from_path(path: Path, class_map: dict[str, int]) -> str | None:
    if path.parent.name in class_map:
        return path.parent.name
    for class_name in class_map:
        if path.stem.startswith(class_name + "_"):
            return class_name
    return None


def inspect_image(path: Path, data_root: Path, class_map: dict[str, int]) -> tuple[dict[str, object], np.ndarray]:
    class_name = class_from_path(path, class_map)
    if class_name is None:
        raise ValueError("无法根据目录或文件名前缀确定类别")
    with Image.open(path) as opened:
        opened.verify()
    with Image.open(path) as opened:
        opened.load()
        width, height = opened.size
        image_format = opened.format or "unknown"
        mode = opened.mode
        channels = len(opened.getbands())
        perceptual_hash = dhash64(opened)
        histogram = np.bincount(
            np.asarray(ImageOps.exif_transpose(opened).convert("L"), dtype=np.uint8).ravel(),
            minlength=256,
        )
    row = {
        "image_path": path.relative_to(data_root).as_posix(),
        "class_name": class_name,
        "label": class_map[class_name],
        "width": width,
        "height": height,
        "format": image_format,
        "mode": mode,
        "channels": channels,
        "sha256": sha256_file(path),
        "dhash64": f"{perceptual_hash:016x}",
    }
    return row, histogram


def make_figures(
    rows: list[dict[str, object]],
    data_root: Path,
    class_map: dict[str, int],
    histogram: np.ndarray,
    figures_dir: Path,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    class_names = list(class_map)
    counts = Counter(str(row["class_name"]) for row in rows)

    fig, axes = plt.subplots(6, 4, figsize=(11, 15), constrained_layout=True)
    for row_index, class_name in enumerate(class_names):
        selected = [row for row in rows if row["class_name"] == class_name][:4]
        for column_index, axis in enumerate(axes[row_index]):
            axis.axis("off")
            if column_index < len(selected):
                relative = str(selected[column_index]["image_path"])
                with Image.open(data_root / relative) as image:
                    axis.imshow(image.convert("L"), cmap="gray", vmin=0, vmax=255)
                axis.set_title(f"{class_name} · {Path(relative).name}", fontsize=9)
    fig.savefig(figures_dir / "neu_cls_examples.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    bars = axis.bar(class_names, [counts[name] for name in class_names], color="#356c9a")
    axis.bar_label(bars)
    axis.set(xlabel="Class", ylabel="Images", title="NEU-CLS class distribution")
    fig.savefig(figures_dir / "neu_cls_class_distribution.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    axis.plot(np.arange(256), histogram, color="#293845", linewidth=1.4)
    axis.set(xlabel="Grayscale value", ylabel="Pixel count", title="NEU-CLS grayscale histogram", xlim=(0, 255))
    fig.savefig(figures_dir / "neu_cls_grayscale_histogram.png", dpi=150)
    plt.close(fig)


def run_quality(
    data_root: Path,
    images_root: Path,
    metrics_dir: Path,
    figures_dir: Path,
    class_map: dict[str, int],
    max_hamming_distance: int = 4,
) -> dict[str, object]:
    data_root = data_root.resolve()
    images_root = images_root.resolve()
    if not images_root.is_dir() or not images_root.is_relative_to(data_root):
        raise FileNotFoundError(f"NEU-CLS 图像目录不存在或不在数据根目录下: {images_root}")
    if not 0 <= max_hamming_distance <= 64:
        raise ValueError("感知哈希距离必须在 0 到 64 之间")

    rows: list[dict[str, object]] = []
    invalid: list[dict[str, object]] = []
    ignored: list[str] = []
    histogram = np.zeros(256, dtype=np.int64)
    for path in sorted(images_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(data_root).as_posix()
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            ignored.append(relative)
            continue
        try:
            row, image_histogram = inspect_image(path, data_root, class_map)
            rows.append(row)
            histogram += image_histogram
        except (OSError, ValueError, UnidentifiedImageError) as error:
            invalid.append({"image_path": relative, "error": f"{type(error).__name__}: {error}"})

    by_sha: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_sha[str(row["sha256"])].append(row)
    exact_groups = [group for group in by_sha.values() if len(group) > 1]
    exact_rows = [
        {"sha256": str(group[0]["sha256"]), "image_path": row["image_path"], "class_name": row["class_name"]}
        for group in exact_groups for row in group
    ]
    near_rows: list[dict[str, object]] = []
    hashes = [int(str(row["dhash64"]), 16) for row in rows]
    for a, b in itertools.combinations(range(len(rows)), 2):
        if rows[a]["sha256"] == rows[b]["sha256"]:
            continue
        distance = (hashes[a] ^ hashes[b]).bit_count()
        if distance <= max_hamming_distance:
            near_rows.append({
                "image_path_a": rows[a]["image_path"],
                "class_a": rows[a]["class_name"],
                "image_path_b": rows[b]["image_path"],
                "class_b": rows[b]["class_name"],
                "hamming_distance": distance,
            })

    # dHash 对整体明暗梯度敏感。额外计算高通纹理相关性，帮助人工优先复核，
    # 但不据此自动删除或排除任何候选。
    involved_paths = {
        str(pair[key]) for pair in near_rows for key in ("image_path_a", "image_path_b")
    }
    textures: dict[str, tuple[np.ndarray, float]] = {}
    for relative in involved_paths:
        with Image.open(data_root / relative) as image:
            gray = image.convert("L")
            pixels = np.asarray(gray, dtype=np.float32)
            low_frequency = np.asarray(gray.filter(ImageFilter.GaussianBlur(radius=4)), dtype=np.float32)
            high_frequency = pixels - low_frequency
            high_frequency -= high_frequency.mean()
            textures[relative] = high_frequency, float(np.linalg.norm(high_frequency))
    for pair in near_rows:
        a, norm_a = textures[str(pair["image_path_a"])]
        b, norm_b = textures[str(pair["image_path_b"])]
        correlation = float(np.sum(a * b) / (norm_a * norm_b)) if norm_a and norm_b else 0.0
        pair["highpass_correlation"] = round(correlation, 6)
        pair["review_priority"] = "high" if correlation >= 0.4 else "standard"
    near_rows.sort(key=lambda pair: (-float(pair["highpass_correlation"]), int(pair["hamming_distance"])))

    metrics_dir.mkdir(parents=True, exist_ok=True)
    write_csv(metrics_dir / "neu_cls_inventory.csv", INVENTORY_FIELDS, rows)
    write_csv(metrics_dir / "neu_cls_invalid_images.csv", ("image_path", "error"), invalid)
    write_csv(metrics_dir / "neu_cls_exact_duplicates.csv", ("sha256", "image_path", "class_name"), exact_rows)
    write_csv(
        metrics_dir / "neu_cls_near_duplicates.csv",
        ("image_path_a", "class_a", "image_path_b", "class_b", "hamming_distance", "highpass_correlation", "review_priority"),
        near_rows,
    )
    class_counts = Counter(str(row["class_name"]) for row in rows)
    write_csv(
        metrics_dir / "neu_cls_class_counts.csv",
        ("class_name", "label", "count"),
        [{"class_name": name, "label": label, "count": class_counts[name]} for name, label in class_map.items()],
    )
    make_figures(rows, data_root, class_map, histogram, figures_dir)
    summary: dict[str, object] = {
        "generated_at_Asia_Shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "dataset": "NEU-CLS",
        "images_root": images_root.relative_to(data_root).as_posix(),
        "total_image_candidates": len(rows) + len(invalid),
        "valid_images": len(rows),
        "invalid_images": len(invalid),
        "ignored_non_image_files": ignored,
        "class_counts": {name: class_counts[name] for name in class_map},
        "dimensions": dict(Counter(f"{row['width']}x{row['height']}" for row in rows)),
        "formats": dict(Counter(str(row["format"]) for row in rows)),
        "modes": dict(Counter(str(row["mode"]) for row in rows)),
        "channel_counts": dict(Counter(str(row["channels"]) for row in rows)),
        "exact_duplicate_groups": len(exact_groups),
        "exact_duplicate_images": len(exact_rows),
        "near_duplicate_pairs": len(near_rows),
        "near_duplicate_cross_class_pairs": sum(pair["class_a"] != pair["class_b"] for pair in near_rows),
        "near_duplicate_distance_counts": dict(sorted(Counter(str(pair["hamming_distance"]) for pair in near_rows).items())),
        "high_priority_near_duplicate_pairs": sum(pair["review_priority"] == "high" for pair in near_rows),
        "near_duplicate_max_hamming_distance": max_hamming_distance,
        "near_duplicates_require_review": bool(near_rows),
    }
    (metrics_dir / "neu_cls_quality.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
