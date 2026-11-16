"""检查项目依赖和 GPU，并保存实际验证结果。"""

from __future__ import annotations

import importlib
import json
import os
import platform
import sys
import traceback
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "results" / "logs" / "environment_validation.json"
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))

PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "pandas": "pandas",
    "Pillow": "PIL",
    "scikit-learn": "sklearn",
    "matplotlib": "matplotlib",
    "PyYAML": "yaml",
    "tqdm": "tqdm",
    "pytest": "pytest",
}


def main() -> int:
    versions: dict[str, str] = {}
    errors: dict[str, str] = {}
    report: dict[str, object] = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "versions": versions,
        "errors": errors,
    }

    if sys.version_info[:2] != (3, 11):
        errors["python"] = "需要 Python 3.11"
    expected_python = ROOT / ".venv" / "Scripts" / "python.exe"
    if Path(sys.executable).resolve() != expected_python.resolve():
        errors["interpreter"] = "未使用项目 .venv 解释器"

    for distribution, module_name in PACKAGES.items():
        try:
            versions[distribution] = version(distribution)
            importlib.import_module(module_name)
        except Exception:
            errors[distribution] = traceback.format_exc()

    if "torch" not in errors:
        try:
            import torch

            report["torch_cuda_runtime"] = torch.version.cuda
            report["cuda_available"] = torch.cuda.is_available()
            if not torch.cuda.is_available():
                raise RuntimeError("torch.cuda.is_available() 为 False")
            report["gpu_name"] = torch.cuda.get_device_name(0)
            report["gpu_compute_capability"] = torch.cuda.get_device_capability(0)
            left = torch.tensor([1.0, 2.0, 3.0], device="cuda")
            right = torch.tensor([4.0, 5.0, 6.0], device="cuda")
            result = (left + right).cpu().tolist()
            report["gpu_tensor_add_result"] = result
            if result != [5.0, 7.0, 9.0]:
                raise AssertionError(f"GPU 张量运算结果异常: {result}")
        except Exception:
            errors["cuda"] = traceback.format_exc()

    report["ok"] = not errors
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
