"""在质量检查之后生成固定 seed=42 的分层目标、分组约束划分。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.split import make_splits  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    class_map = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
    report = make_splits(
        data_root=ROOT / config["data_root"],
        metrics_dir=ROOT / "results" / "metrics",
        splits_dir=ROOT / "data" / "splits",
        class_map=class_map,
        ratios=config["split"],
        seed=config["seed"],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
