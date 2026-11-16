"""从 torchvision 指定的官方 URL 分段下载并校验 ResNet18 ImageNet 权重。"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from torchvision.models import ResNet18_Weights


ROOT = Path(__file__).resolve().parents[1]
URL = ResNet18_Weights.IMAGENET1K_V1.url
FILENAME = Path(URL).name
EXPECTED_HASH_PREFIX = FILENAME.rsplit("-", 1)[1].split(".")[0]
TARGET = ROOT / ".cache" / "torch" / "hub" / "checkpoints" / FILENAME
PARTS_DIR = ROOT / ".cache" / "torch" / "parts"
RECORD = ROOT / "results" / "metrics" / "c0_seed42_20261109_pretrained_weight.json"
PART_COUNT = 8


def open_range(start: int, end: int):
    return urlopen(Request(URL, headers={"Range": f"bytes={start}-{end}"}), timeout=90)


def get_size() -> int:
    with open_range(0, 0) as response:
        header = response.headers.get("Content-Range", "")
        if response.status != 206 or not header.startswith("bytes 0-0/"):
            raise RuntimeError(f"官方服务器不支持分段续传: {response.status} {header}")
        return int(header.rsplit("/", 1)[1])


def fetch_part(index: int, start: int, end: int, total_size: int) -> Path:
    path = PARTS_DIR / f"{FILENAME}.part{index}"
    expected = end - start + 1
    for attempt in range(1, 31):
        current = path.stat().st_size if path.exists() else 0
        if current == expected:
            return path
        if current > expected:
            raise RuntimeError(f"分段超出预期长度: {path}")
        offset = start + current
        try:
            with open_range(offset, end) as response:
                header = response.headers.get("Content-Range", "")
                if response.status != 206 or header != f"bytes {offset}-{end}/{total_size}":
                    raise RuntimeError(f"错误的分段响应: {response.status} {header}")
                with path.open("ab") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                if path.stat().st_size == expected:
                    return path
        except (OSError, TimeoutError) as error:
            print(f"part {index} retry {attempt}: {error}", flush=True)
    raise RuntimeError(f"分段 {index} 在 30 次尝试后仍未完成")


def main() -> None:
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    total_size = get_size()
    if TARGET.exists():
        digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
        if TARGET.stat().st_size != total_size or not digest.startswith(EXPECTED_HASH_PREFIX):
            raise RuntimeError(f"已有权重文件哈希不正确，拒绝覆盖: {TARGET}")
    else:
        chunk_size = (total_size + PART_COUNT - 1) // PART_COUNT
        with ThreadPoolExecutor(max_workers=PART_COUNT) as pool:
            futures = [
                pool.submit(fetch_part, i, i * chunk_size, min((i + 1) * chunk_size, total_size) - 1, total_size)
                for i in range(PART_COUNT)
            ]
            for future in as_completed(futures):
                future.result()
        temporary = TARGET.with_name(TARGET.name + ".assembling")
        digestor = hashlib.sha256()
        with temporary.open("wb") as output:
            for index in range(PART_COUNT):
                part = PARTS_DIR / f"{FILENAME}.part{index}"
                with part.open("rb") as source:
                    while chunk := source.read(1024 * 1024):
                        output.write(chunk)
                        digestor.update(chunk)
        digest = digestor.hexdigest()
        if temporary.stat().st_size != total_size or not digest.startswith(EXPECTED_HASH_PREFIX):
            raise RuntimeError(f"权重完整性校验失败: size={temporary.stat().st_size}, sha256={digest}")
        temporary.replace(TARGET)
    record = {
        "source_url": URL,
        "torchvision_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "cached_path": TARGET.relative_to(ROOT).as_posix(),
        "size_bytes": total_size,
        "sha256": digest,
        "verified_prefix": EXPECTED_HASH_PREFIX,
        "verified_at_Asia_Shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    if RECORD.exists():
        raise FileExistsError(f"预训练权重记录已存在，拒绝覆盖: {RECORD}")
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
