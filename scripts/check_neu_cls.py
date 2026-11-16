"""执行 NEU-CLS 数据质量检查。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.quality import run_quality  # noqa: E402


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    class_map = json.loads((ROOT / "configs" / "class_map.json").read_text(encoding="utf-8"))
    summary = run_quality(
        data_root=ROOT / config["data_root"],
        images_root=ROOT / config["images_root"],
        metrics_dir=ROOT / "results" / "metrics",
        figures_dir=ROOT / "results" / "figures",
        class_map=class_map,
        max_hamming_distance=config["quality"]["near_duplicate_max_hamming_distance"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
