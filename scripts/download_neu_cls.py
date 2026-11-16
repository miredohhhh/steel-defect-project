"""从东北大学教师主页所列的 NEU-CLS Google Drive 链接下载原始压缩包。"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
SOURCE_PAGE = "https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/"
DRIVE_FILE_ID = "1NGlXT9sIaQpyxUoT6MLKm1Pr6x8oxOvc"
DRIVE_PAGE = f"https://drive.google.com/file/d/{DRIVE_FILE_ID}/view"
ARCHIVE = ROOT / "data" / "raw" / "NEU-CLS.rar"
RECORD = ROOT / "results" / "metrics" / "neu_cls_download.json"
USER_AGENT = "Mozilla/5.0 NEU-CLS-data-engineering/1.0"


class DownloadForm(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_form = False
        self.action = ""
        self.fields: dict[str, str] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if tag == "form" and values.get("id") == "download-form":
            self.in_form = True
            self.action = values.get("action") or ""
        elif self.in_form and tag == "input" and values.get("name"):
            self.fields[values["name"]] = values.get("value") or ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "form":
            self.in_form = False


def request(url: str):
    return urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=120)


def download_url() -> str:
    with request(f"https://drive.google.com/uc?export=download&id={DRIVE_FILE_ID}") as response:
        page = response.read().decode("utf-8")
    form = DownloadForm()
    form.feed(page)
    if not form.action.startswith("https://drive.usercontent.google.com/download"):
        raise RuntimeError("Google Drive 未返回可确认的官方文件下载表单")
    if form.fields.get("id") != DRIVE_FILE_ID or form.fields.get("confirm") != "t":
        raise RuntimeError("Google Drive 下载表单的文件 ID 或确认字段不匹配")
    return f"{form.action}?{urlencode(form.fields)}"


def main() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    temporary = ARCHIVE.with_suffix(".rar.part")
    resume = "--resume" in sys.argv[1:]
    if ARCHIVE.exists():
        if not resume or temporary.exists():
            raise FileExistsError(f"原始包已存在，拒绝覆盖: {ARCHIVE}")
        ARCHIVE.replace(temporary)
    elif temporary.exists() and not resume:
        raise FileExistsError(f"发现未完成的下载，使用 --resume 继续: {temporary}")

    with request(download_url()) as response:
        expected_size = int(response.headers["Content-Length"])
        disposition = response.headers.get("Content-Disposition", "")
        if "NEU-CLS.rar" not in disposition:
            raise RuntimeError(f"响应并非预期的 NEU-CLS.rar: {disposition!r}")

    try:
        for attempt in range(20):
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset == expected_size:
                break
            if offset > expected_size:
                raise RuntimeError(f"下载文件大于服务器报告的长度: {offset} > {expected_size}")
            headers = {"User-Agent": USER_AGENT, "Range": f"bytes={offset}-"}
            with urlopen(Request(download_url(), headers=headers), timeout=120) as response:
                content_range = response.headers.get("Content-Range", "")
                if response.status != 206 or not content_range.startswith(f"bytes {offset}-"):
                    raise RuntimeError(f"服务器未按要求续传: {response.status} {content_range}")
                if not content_range.endswith(f"/{expected_size}"):
                    raise RuntimeError(f"续传文件长度变化: {content_range}")
                with temporary.open("ab") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
            new_size = temporary.stat().st_size
            print(f"downloaded {new_size}/{expected_size} bytes (attempt {attempt + 1})", flush=True)
            if new_size <= offset:
                raise RuntimeError("续传没有增加任何数据")
        if temporary.stat().st_size != expected_size:
            raise RuntimeError(f"下载不完整: {temporary.stat().st_size}/{expected_size} bytes")
        with temporary.open("rb") as source:
            if not source.read(7).startswith(b"Rar!\x1a\x07"):
                raise RuntimeError("下载文件没有 RAR 文件头")
        listing = subprocess.run(["tar", "-tf", str(temporary)], capture_output=True, text=True)
        if listing.returncode != 0:
            raise RuntimeError(f"RAR 列表校验失败: {listing.stderr.strip()}")
        entries = [line for line in listing.stdout.splitlines() if line.lower().endswith(".bmp")]
        if len(entries) != 1800:
            raise RuntimeError(f"RAR 内 BMP 数量与官方说明不符: {len(entries)}")
        sha256 = hashlib.sha256()
        with temporary.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                sha256.update(chunk)
        temporary.replace(ARCHIVE)
    except Exception:
        if temporary.exists():
            print(f"未完成文件保留于 {temporary}")
        raise

    record = {
        "dataset": "NEU-CLS",
        "source_page": SOURCE_PAGE,
        "official_page_google_drive_link": DRIVE_PAGE,
        "downloaded_at_Asia_Shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "archive_path": ARCHIVE.relative_to(ROOT).as_posix(),
        "archive_size_bytes": expected_size,
        "archive_sha256": sha256.hexdigest(),
        "archive_bmp_entries": len(entries),
        "license_status": "官方页面未给出明确许可证或使用条款；只列出引用建议，需进一步确认",
    }
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    RECORD.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
