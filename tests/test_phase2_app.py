"""运行真实 Streamlit 脚本，上传数据通过控件边界注入。"""

import io
from pathlib import Path
from unittest.mock import patch

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def upload(name):
    result = io.BytesIO()
    with Image.open(ROOT / "data/raw/NEU-CLS" / name) as image:
        image.save(result, format="PNG")
    result.name = Path(name).with_suffix(".png").name
    return result


@pytest.mark.parametrize("filename", ["Cr_1.bmp", "In_240.bmp", "Sc_141.bmp"])
def test_three_real_images_and_class_correction(filename):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40)
    with patch("streamlit.file_uploader", return_value=upload(filename)):
        at.run()
        assert not at.exception
        assert at.session_state["prediction"]["predicted_class"] in {"Cr", "In", "Pa", "PS", "RS", "Sc"}
        scores = at.session_state["prediction"]["scores"]
        assert abs(sum(scores.values()) - 1) < 1e-4
        at.selectbox[0].select("RS").run()
        at.button[0].click().run()
        assert at.session_state["confirmed_class"] == "RS"
        assert any("尚待独立成员审核" in item.value for item in at.info)
        at.selectbox[0].select("Pa").run()
        assert "confirmed_class" not in at.session_state
        at.button[0].click().run()
        assert at.session_state["confirmed_class"] == "Pa"
        assert any("暂无可靠" in item.value for item in at.info)
        assert not at.exception
    with patch("streamlit.file_uploader", return_value=upload("PS_1.bmp")):
        at.run()
        assert "confirmed_class" not in at.session_state
        assert not at.exception


def test_bad_image_recovers_and_missing_checkpoint(monkeypatch):
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40)
    invalid = io.BytesIO(b"broken")
    invalid.name = "broken.png"
    with patch("streamlit.file_uploader", return_value=invalid):
        at.run()
        assert at.error and not at.exception
    with patch("streamlit.file_uploader", return_value=upload("Cr_1.bmp")):
        at.run()
        assert not at.error and not at.exception
    monkeypatch.setenv("STEEL_CHECKPOINT", "checkpoints/missing_test.pt")
    at.run()
    assert any("模型暂时无法加载" in item.value for item in at.error)
    assert not at.exception
    monkeypatch.delenv("STEEL_CHECKPOINT")
    with patch("streamlit.file_uploader", return_value=None):
        at.run()
        assert not at.error and not at.exception


def test_experiment_and_coverage_pages():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=40).run()
    at.radio[0].set_value("实验分析").run()
    assert len(at.dataframe) >= 1
    assert not at.exception
    at.radio[0].set_value("知识覆盖").run()
    assert len(at.dataframe[0].value) == 6
    assert not at.exception
