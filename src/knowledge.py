"""本地六类工艺条目校验与确认后查询。"""

import json
from datetime import date
from pathlib import Path

from src.inference import CLASS_NAMES

BOUNDARY = "仅供工艺核查参考，不构成根因诊断或生产参数调整指令。"
FIELDS = {"id", "class_code", "process", "mechanism", "checks", "conditions", "evidence_location",
          "source_url", "source_title", "evidence_type", "limitations", "entered_by", "reviewer",
          "review_date", "review_status", "source_status", "source_checked_at"}


def load_knowledge(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if set(data["coverage"]) != set(CLASS_NAMES):
        raise ValueError("知识覆盖状态必须包含六类")
    identifiers = set()
    for record in data["records"]:
        if FIELDS - record.keys() or record["id"] in identifiers or record["class_code"] not in CLASS_NAMES:
            raise ValueError("知识条目字段、编号或类别无效")
        identifiers.add(record["id"])
        if record["review_status"] not in {"pending", "approved", "rejected"}:
            raise ValueError("知识审核状态无效")
        if record["review_status"] == "approved":
            required = ("reviewer", "review_date", "evidence_location", "source_url", "conditions", "limitations")
            if any(not record[key] for key in required) or record["reviewer"] == record["entered_by"]:
                raise ValueError("已审核条目需要独立复核人、日期和完整来源")
            date.fromisoformat(record["review_date"])
    return data


def query_knowledge(data: dict, confirmed_class: str | None, conditions_ok: bool) -> dict:
    if confirmed_class is None:
        return {"records": [], "reason": "请先确认或修正缺陷类别。"}
    if confirmed_class not in CLASS_NAMES:
        raise ValueError("类别编码无效")
    candidates = [record for record in data["records"] if record["class_code"] == confirmed_class]
    approved = [record for record in candidates if record["review_status"] == "approved"]
    valid = [record for record in approved if record["source_status"] == "verified" and record["source_checked_at"]]
    if not candidates:
        reason = "暂无可靠工艺关联，等待资料补充。"
    elif not approved:
        reason = "该类别候选条目尚待独立成员审核。"
    elif not valid:
        reason = "来源失效或待核查，暂缓展示具体工艺建议。"
    elif not conditions_ok:
        reason = "请核对资料适用条件。条件未知或不适用时，保留分类结果供人工核查。"
    else:
        return {"records": valid, "reason": ""}
    return {"records": [], "reason": reason}


def coverage_rows(data: dict) -> list[dict]:
    rows = []
    for code, name in CLASS_NAMES.items():
        candidates = [record for record in data["records"] if record["class_code"] == code]
        approved = sum(record["review_status"] == "approved" for record in candidates)
        rows.append({"类别": f"{code} {name}", "覆盖状态": "有审核通过的知识" if approved else data["coverage"][code],
                     "候选条目": len(candidates), "已审核条目": approved})
    return rows
