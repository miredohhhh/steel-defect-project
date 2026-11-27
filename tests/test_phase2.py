"""第二阶段输入和知识门控回归测试。"""

import copy
import io
from pathlib import Path

import pytest
from PIL import Image

from src.inference import MAX_UPLOAD_BYTES, decode_upload, reset_confirmation
from src.knowledge import coverage_rows, load_knowledge, query_knowledge

ROOT = Path(__file__).resolve().parents[1]


def test_upload_decode_and_rejections():
    buffer = io.BytesIO()
    Image.new("L", (200, 200)).save(buffer, format="PNG")
    assert decode_upload(buffer.getvalue(), "sample.png").mode == "RGB"
    for content, name in [(b"bad", "broken.png"), (buffer.getvalue(), "file.txt"),
                          (b"x" * (MAX_UPLOAD_BYTES + 1), "large.png")]:
        with pytest.raises(ValueError):
            decode_upload(content, name)


def test_new_input_clears_confirmation():
    state = {"input_identity": "old", "confirmed_class": "In", "prediction": {}, "conditions_ok": True, "class_choice": "In"}
    reset_confirmation(state, "new")
    assert state == {"input_identity": "new"}


def test_knowledge_requires_confirmation_review_and_conditions():
    data = load_knowledge(ROOT / "data/knowledge/process_knowledge.json")
    assert len(coverage_rows(data)) == 6
    assert query_knowledge(data, None, True)["records"] == []
    assert query_knowledge(data, "RS", True)["records"] == []
    assert query_knowledge(data, "Pa", True)["records"] == []
    # 仅测试夹具模拟复核，生产知识表保持真实 pending 状态。
    fixture = copy.deepcopy(data)
    record = fixture["records"][0]
    record.update(review_status="approved", reviewer="test reviewer", review_date="2026-11-23")
    assert query_knowledge(fixture, "RS", False)["records"] == []
    assert [r["id"] for r in query_knowledge(fixture, "RS", True)["records"]] == [record["id"]]
    assert query_knowledge(fixture, "Sc", True)["records"] == []
    record["source_status"] = "unavailable"
    assert query_knowledge(fixture, "RS", True)["records"] == []


def test_approved_knowledge_needs_independent_reviewer(tmp_path):
    data = load_knowledge(ROOT / "data/knowledge/process_knowledge.json")
    record = data["records"][0]
    record.update(review_status="approved", reviewer=record["entered_by"], review_date="2026-11-23")
    import json
    path = tmp_path / "knowledge.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load_knowledge(path)
