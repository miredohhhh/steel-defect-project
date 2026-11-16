"""检查官方 NEU-CLS RAR 条目后，解压到 data/raw/NEU-CLS。"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data" / "raw" / "NEU-CLS.rar"
DESTINATION = ROOT / "data" / "raw" / "NEU-CLS"
IMAGE_NAME = re.compile(r"NEU-CLS/(Cr|In|Pa|PS|RS|Sc)_[0-9]+\.bmp$")


def run_tar(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(["tar", *args], capture_output=True, text=True)
    if result.returncode:
        excerpt = "\n".join(result.stderr.splitlines()[:8])
        raise RuntimeError(f"tar {' '.join(args)} 失败（前 8 行）:\n{excerpt}")
    return result


def main() -> None:
    if not ARCHIVE.is_file():
        raise FileNotFoundError(ARCHIVE)
    names = run_tar("-tf", str(ARCHIVE)).stdout.splitlines()
    details = run_tar("-tvf", str(ARCHIVE)).stdout.splitlines()
    if len(names) != len(details):
        raise RuntimeError("RAR 条目列表与类型列表数量不一致")
    images = []
    for name, detail in zip(names, details):
        if IMAGE_NAME.fullmatch(name):
            if not detail.startswith("-"):
                raise RuntimeError(f"图像条目不是普通文件: {name}")
            images.append(name)
        elif name == "NEU-CLS":
            if not detail.startswith("d"):
                raise RuntimeError("NEU-CLS 根条目不是目录")
        elif name == "NEU-CLS/Thumbs.db":
            if not detail.startswith("-"):
                raise RuntimeError("Thumbs.db 不是普通文件")
        else:
            raise RuntimeError(f"RAR 中有意外路径，拒绝解压: {name}")
    if len(images) != 1800 or len(set(images)) != 1800:
        raise RuntimeError(f"RAR 图像条目数量异常: {len(images)}")

    if DESTINATION.exists():
        existing = {"NEU-CLS/" + item.name for item in DESTINATION.glob("*.bmp")}
        if existing == set(images):
            print(f"已存在全部 {len(existing)} 张 BMP；不会覆盖。请继续运行逐张质量检查。")
            return
        raise FileExistsError(f"目标目录已存在但内容不完整，拒绝覆盖: {DESTINATION}")

    run_tar("-xmf", str(ARCHIVE), "-C", str(ARCHIVE.parent))
    extracted = list(DESTINATION.glob("*.bmp"))
    if len(extracted) != 1800:
        raise RuntimeError(f"实际解压图像数异常: {len(extracted)}")
    print(f"已解压 {len(extracted)} 张 BMP 至 {DESTINATION}")


if __name__ == "__main__":
    main()
