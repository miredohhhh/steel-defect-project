"""从验证预测与真实训练产物汇总实验、图表和第二阶段报告。"""

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.inference import CLASS_NAMES
from src.knowledge import coverage_rows, load_knowledge


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    metrics = ROOT / "results/metrics"
    models = []
    names = list(CLASS_NAMES)
    for experiment in ("c0_seed42_20261109", "c1_seed42_20261123"):
        summary = read_json(metrics / f"{experiment}_summary.json")
        run = read_json(metrics / f"{experiment}_run.json")
        if summary["status"] != "complete":
            raise ValueError("实验尚未完成")
        with (metrics / f"{experiment}_val_predictions.csv").open(encoding="utf-8-sig", newline="") as source:
            rows = list(csv.DictReader(source))
        actual = [int(row["true_label"]) for row in rows]
        predicted = [int(row["pred_label"]) for row in rows]
        report = classification_report(actual, predicted, labels=list(range(6)), target_names=names, output_dict=True, zero_division=0)
        accuracy = accuracy_score(actual, predicted)
        assert len(rows) == 270
        assert abs(accuracy - summary["best_val_metrics"]["accuracy"]) < 1e-6
        assert abs(report["macro avg"]["f1-score"] - summary["best_val_metrics"]["macro_f1"]) < 1e-6
        matrix = confusion_matrix(actual, predicted, labels=list(range(6)))
        details = {"split": "val", "sample_count": len(rows), "accuracy": accuracy,
                   "macro_precision": report["macro avg"]["precision"], "macro_recall": report["macro avg"]["recall"],
                   "macro_f1": report["macro avg"]["f1-score"], "per_class": {name: report[name] for name in names},
                   "confusion_matrix": matrix.tolist(), "errors": [row for row in rows if row["true_label"] != row["pred_label"]]}
        write_json(metrics / f"{experiment}_val_report.json", details)
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        ax.imshow(matrix, cmap="Blues")
        ax.set(xticks=range(6), yticks=range(6), xticklabels=names, yticklabels=names,
               xlabel="Predicted class", ylabel="True class", title=f"{experiment} / val (270)")
        for row in range(6):
            for col in range(6):
                ax.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > 22 else "black")
        fig.savefig(ROOT / "results/figures" / f"{experiment}_val_confusion_matrix.png", dpi=150)
        plt.close(fig)
        models.append({"experiment_id": experiment, "split": "val", "accuracy": accuracy,
                       "macro_f1": details["macro_f1"], "best_epoch": summary["best_epoch"],
                       "epochs": summary["epochs_completed"], "train_seconds": summary["training_total_seconds"],
                       "trainable_parameters": run["model_structure"]["trainable_parameters"],
                       "device": run.get("device", run.get("gpu_name", "待核实")),
                       "errors": len(details["errors"])})
    selected = models[1] if models[1]["macro_f1"] > models[0]["macro_f1"] + 1e-12 else models[0]
    note = "C1 从 C0 继续训练，增加了训练预算。本轮 C1 与历史 C0 的设备环境不同，耗时仅作记录，不能作同硬件速度对照。"
    comparison = {"models": models, "selected_model": selected["experiment_id"], "selection_rule": "验证 Macro-F1 优先，持平选 C0", "cost_note": note, "test_evaluated": False}
    write_json(metrics / "phase2_comparison.json", comparison)
    knowledge = load_knowledge(ROOT / "data/knowledge/process_knowledge.json")
    coverage = coverage_rows(knowledge)
    write_json(metrics / "phase2_knowledge_coverage.json", {"coverage": coverage, "approved_count": sum(row["已审核条目"] for row in coverage)})
    table = "\n".join(f"| {m['experiment_id']} | {m['accuracy']:.6f} | {m['macro_f1']:.6f} | {m['best_epoch']} | {m['epochs']} | {m['train_seconds']:.3f} | {m['trainable_parameters']} | {m['errors']} |" for m in models)
    report = f"""# 第二阶段项目进展报告

记录日期为 2026-11-23。范围为本机实现及固定验证集实验。外部成员复核和用户验收待完成。

## 1. 承接与实现

承接第一阶段 1800 张 NEU-CLS 图像及 1260/270/270 固定划分。单图接口复用数据配置和验证预处理，页面支持 JPEG/PNG 上传、预览、六类分数、人工确认与修正。新图和权重身份变化时重置确认状态。

新增 C1 只训练 layer4 和 fc，全部 BatchNorm 运行统计量固定。真实批次 Smoke Test 验证初始化、梯度与更新范围及重载一致性，正式训练后再次验证最佳权重重载。配置和依赖环境见实际 run.json。

## 2. 验证集实验结果

| 实验 | Accuracy | Macro-F1 | 最佳轮次 | 总轮次 | 训练秒数 | 可训练参数 | 错例数 |
|---|---|---|---|---|---|---|---|
{table}

{note}

阶段选择结果为 {selected['experiment_id']}，按预定验证 Macro-F1 规则决定。页面默认加载该阶段选择模型，可通过 STEEL_CHECKPOINT 显式指定其他已有权重。

逐类指标、错误样本和混淆矩阵由真实验证预测重算。实验数据见 [比较结果](../../results/metrics/phase2_comparison.json)。每个实验的 val_report.json 包含逐类 Precision/Recall/F1 与错例，results/figures 中保存曲线及混淆矩阵。

## 3. 工艺知识原型

本地 JSON 精确按最终确认类别查询。已定位两条 AIST 候选资料，分别涉及 RS 除鳞记录和 Sc 接触路径，原文位置与适用条件随条目保存。六类覆盖状态均已记录。正式展示条目须经录入者之外的成员复核，当前审核通过条目为 0。页面显示待审核和无可靠知识状态，具体工艺内容等待真实复核。

审核操作见 [知识复核说明](knowledge_review.md)。该依赖来自现有需求 V0.3 的审核规则。

## 4. 验证与限制

实际测试和本机验证结果见 [验证记录](validation.md)，用户验收状态单独记录。验证分数只覆盖当前同源六类图像。已有近重复保留在验证和测试集合内部，数据代表性和未知输入表现仍有验证缺口。

本阶段未执行保留测试集模型评价。AC-04 及正式测试结果展示留待最终协议锁定后完成。

## 5. 后续工作

请成员复核候选资料和适用条件，填写真实身份、日期和结论。之后验证真实审核条目展示，并由其他成员按 README 独立复现。第三阶段进行深入错误分析、来源交叉复核和模型锁定后的最终评价。课程 LaTeX 正式排版与提交由团队按教师模板安排。
"""
    target = ROOT / "docs/phase2"
    target.mkdir(parents=True, exist_ok=True)
    (target / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
