"""Streamlit 本地单图分类与确认后知识查询。"""

import hashlib
import json
import os
import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
import torch

from src.inference import CLASS_NAMES, Predictor, decode_upload, reset_confirmation
from src.knowledge import BOUNDARY, coverage_rows, load_knowledge, query_knowledge

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(min(8, os.cpu_count() or 1))
st.set_page_config(page_title="钢材表面缺陷分类", layout="wide")


@st.cache_resource
def load_predictor(checkpoint, signature):
    return Predictor(ROOT, Path(checkpoint))


def classification():
    st.subheader("单图识别")
    st.write("上传一张已有缺陷的钢材表面图像，查看六类预测，并确认最终类别。")
    st.caption("适用于 NEU-CLS 六类已知缺陷。正常表面、未知缺陷和不同成像条件下的表现待验证。")
    default = "checkpoints/c0_seed42_20261109_best.pt"
    comparison_path = ROOT / "results/metrics/phase2_comparison.json"
    if comparison_path.exists():
        default = f"checkpoints/{json.loads(comparison_path.read_text(encoding='utf-8'))['selected_model']}_best.pt"
    default = os.environ.get("STEEL_CHECKPOINT", default)
    checkpoint = ROOT / default
    try:
        signature = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        predictor = load_predictor(str(checkpoint), signature)
    except (OSError, ValueError, KeyError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
        st.error(f"模型暂时无法加载，请检查权重和配置后重试。{error}")
        return
    st.caption(f"当前模型 {predictor.model_id}")
    uploaded = st.file_uploader("选择图片（JPEG/PNG，最大 10 MB）", type=["jpg", "jpeg", "png"])
    if uploaded is None:
        reset_confirmation(st.session_state, None)
        return
    content = uploaded.getvalue()
    identity = hashlib.sha256(content).hexdigest() + signature
    reset_confirmation(st.session_state, identity)
    try:
        image = decode_upload(content, uploaded.name)
        if "prediction" not in st.session_state:
            st.session_state.prediction = predictor.predict(image)
    except (OSError, ValueError, RuntimeError) as error:
        st.error(f"图片识别失败，请重新上传。{error}")
        return
    result = st.session_state.prediction
    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption=uploaded.name, width=400)
    with right:
        st.markdown(f"**原始预测　{result['predicted_class']} · {result['predicted_name']}**")
        st.dataframe(pd.DataFrame([{"类别": f"{code} {CLASS_NAMES[code]}", "Softmax 分数": value}
                                   for code, value in result["scores"].items()]), hide_index=True, width="stretch")
        st.caption("Softmax 分数反映模型的六类输出，正确率校准仍待验证。")
    options = list(result["scores"])
    choice = st.selectbox("确认或修正缺陷类别", options, index=options.index(result["predicted_class"]),
                          format_func=lambda code: f"{code} · {CLASS_NAMES[code]}", key="class_choice")
    if st.session_state.get("confirmed_class") != choice:
        st.session_state.pop("confirmed_class", None)
        st.session_state.pop("conditions_ok", None)
    if st.button("确认该类别", type="primary"):
        st.session_state.confirmed_class = choice
    st.subheader("工艺核查资料")
    st.caption(BOUNDARY)
    try:
        data = load_knowledge(ROOT / "data/knowledge/process_knowledge.json")
    except (OSError, ValueError, KeyError) as error:
        st.warning(f"知识资料暂不可用。{error}")
        return
    confirmed = st.session_state.get("confirmed_class")
    if confirmed:
        st.write(f"最终确认类别　{confirmed} · {CLASS_NAMES[confirmed]}")
        applicable = [record for record in data["records"] if record["class_code"] == confirmed and record["review_status"] == "approved"]
        for record in applicable:
            st.write(f"资料适用条件　{record['conditions']}")
        conditions_ok = st.checkbox("已核对上述资料适用条件，当前样本满足这些条件", key="conditions_ok", disabled=not applicable)
    else:
        conditions_ok = False
    response = query_knowledge(data, confirmed, conditions_ok)
    if response["reason"]:
        st.info(response["reason"])
    for record in response["records"]:
        st.markdown(f"**{record['id']}　{record['process']}**")
        st.write(record["mechanism"])
        st.write("建议核查对象　" + "、".join(record["checks"]))
        st.write("限制与反例　" + record["limitations"])
        st.write(f"审核人 {record['reviewer']}，审核日期 {record['review_date']}，状态 approved")
        st.write(f"原文位置 {record['evidence_location']}")
        st.link_button(record["source_title"], record["source_url"])


def experiments():
    st.subheader("验证集实验")
    st.caption("270 张固定验证集图像用于模型选择。保留测试集尚待最终评价。")
    comparison_path = ROOT / "results/metrics/phase2_comparison.json"
    if not comparison_path.exists():
        st.info("实验汇总尚待生成。")
        return
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    st.dataframe(pd.DataFrame(comparison["models"]), hide_index=True, width="stretch")
    st.write(comparison["cost_note"])
    for model in comparison["models"]:
        experiment = model["experiment_id"]
        with st.expander(experiment, expanded=True):
            for suffix in ("curves", "val_confusion_matrix"):
                figure = ROOT / "results/figures" / f"{experiment}_{suffix}.png"
                if figure.exists():
                    st.image(str(figure), width="stretch")
            report = ROOT / "results/metrics" / f"{experiment}_val_report.json"
            if report.exists():
                st.dataframe(pd.DataFrame(json.loads(report.read_text(encoding="utf-8"))["per_class"]).T, width="stretch")


st.title("钢材表面缺陷分类与辅助分析")
page = st.radio("查看内容", ["单图识别", "实验分析", "知识覆盖"], horizontal=True, label_visibility="collapsed")
if page == "单图识别":
    classification()
elif page == "实验分析":
    experiments()
else:
    st.subheader("工艺知识覆盖")
    st.caption(BOUNDARY)
    try:
        st.dataframe(coverage_rows(load_knowledge(ROOT / "data/knowledge/process_knowledge.json")), hide_index=True, width="stretch")
    except (OSError, ValueError, KeyError) as error:
        st.error(f"知识表读取失败。{error}")
