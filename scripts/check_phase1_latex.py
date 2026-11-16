"""不依赖 TeX 发行版的第一阶段报告静态核验。"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATCHES = list((ROOT / "report_overleaf").rglob("MLPTA.tex"))
if len(MATCHES) != 1:
    raise RuntimeError(f"预期找到一份 MLPTA.tex, 实际为 {len(MATCHES)} 份")
MAIN = MATCHES[0]
TEXT = MAIN.read_text(encoding="utf-8")
BACKUP = ROOT / "docs" / "phase1" / "MLPTA.teacher-original.tex"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check() -> dict:
    assert BACKUP.is_file() and sha256(BACKUP) != sha256(MAIN)
    assert r"\documentclass[letterpaper, 10 pt, conference]{ieeeconf}" in TEXT
    assert (MAIN.parent / "ieeeconf.cls").is_file()
    assert all(f"\\section{{{name}}}" in TEXT for name in (
        "引言", "相关研究与需求分析", "数据集与质量保障", "冻结主干模型与实验方法",
        "验证结果与误差分析", "工艺辅助分析方案", "讨论", "结论与展望",
    ))
    assert r"\begin{abstract}" in TEXT and r"\begin{eabstract}" in TEXT
    assert not re.search(r"(?<![A-Za-z])[A-Za-z]:[/\\]", TEXT)
    assert "smooth.eps" not in TEXT and "机器学习高级技术及应用作业模板" not in TEXT
    assert "202421315022" in TEXT and "陈家浩" in TEXT
    assert "组员二" in TEXT and "组员三" in TEXT

    levels = []
    for action, name in re.findall(r"\\(begin|end)\{([A-Za-z*]+)\}", TEXT):
        if action == "begin":
            levels.append(name)
        else:
            assert levels and levels.pop() == name, f"环境不匹配: {name}"
    assert not levels
    brace_level = 0
    for match in re.finditer(r"(?<!\\)[{}]", TEXT):
        brace_level += 1 if match.group() == "{" else -1
        assert brace_level >= 0
    assert brace_level == 0

    labels = re.findall(r"\\label\{([^}]+)\}", TEXT)
    refs = re.findall(r"\\ref\{([^}]+)\}", TEXT)
    bibliography = re.findall(r"\\bibitem\{([^}]+)\}", TEXT)
    citations = [key.strip() for group in re.findall(r"\\cite\{([^}]+)\}", TEXT) for key in group.split(",")]
    assert len(labels) == len(set(labels)) and set(refs) <= set(labels)
    assert len(bibliography) == len(set(bibliography)) and set(citations) <= set(bibliography)
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", TEXT)
    assert len(figures) == 5
    for figure in figures:
        copied = MAIN.parent / "figures" / figure
        assert copied.is_file()
        if figure == "c0_seed42_20261109_curves.png":
            source = ROOT / "results" / "figures" / figure
            assert source.is_file() and sha256(copied) == sha256(source)

    summary = json.loads((ROOT / "results" / "metrics" / "c0_seed42_20261109_summary.json").read_text(encoding="utf-8"))
    quality = json.loads((ROOT / "results" / "metrics" / "neu_cls_quality.json").read_text(encoding="utf-8"))
    split = json.loads((ROOT / "results" / "metrics" / "neu_cls_split_report.json").read_text(encoding="utf-8"))
    assert summary["status"] == "complete" and summary["best_epoch"] == 8
    assert f'{summary["best_val_metrics"]["accuracy"]:.6f}' in TEXT
    assert f'{summary["best_val_metrics"]["macro_f1"]:.6f}' in TEXT
    assert quality["valid_images"] == 1800 and quality["near_duplicate_pairs"] == 4418
    assert split["group_count"] == 1511 and split["largest_group_size"] == 207
    assert all(str(n) in TEXT for n in (4418, 1511, 207, 1800))

    result = {
        "passed": True,
        "main_tex": MAIN.relative_to(ROOT).as_posix(),
        "template_backup": BACKUP.relative_to(ROOT).as_posix(),
        "figures": figures,
        "labels": labels,
        "references": refs,
        "bibliography_keys": bibliography,
        "citations": citations,
        "xelatex_available": shutil.which("xelatex") is not None,
        "figure_files_exist": True,
        "training_curve_matches_source": True,
        "artifact_metrics_match": True,
    }
    return result


if __name__ == "__main__":
    report = check()
    output = ROOT / "docs" / "phase1" / "report_latex_static_check.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
