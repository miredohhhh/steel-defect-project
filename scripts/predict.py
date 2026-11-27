"""在项目根目录运行 python scripts/predict.py IMAGE。"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.inference import Predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/c0_seed42_20261109_best.pt")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    predictor = Predictor(ROOT, args.checkpoint, args.device)
    with Image.open(args.image) as image:
        print(json.dumps(predictor.predict(image), ensure_ascii=False, indent=2))
