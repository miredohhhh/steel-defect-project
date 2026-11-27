# 第二阶段验证记录

验证日期为 2026-11-23。范围为本机 CPU 环境的 C1 实验、推理、Streamlit 脚本交互及浏览器检查。成员复核和用户验收待完成。

## 1. 环境与实验

本机为 Python 3.13.7、CPU 版 torch 2.12.0、torchvision 0.27.0。创建项目 .venv，复用系统已安装库并补装 Streamlit、pytest、scikit-learn。pip check 返回 No broken requirements found。精确依赖见 [环境 JSON](../../results/logs/phase2_environment.json)，独立安装参考 requirements-phase2.txt，干净环境重新安装待跨成员验证。

C1 的真实批次 Smoke Test、正式 train/val 训练、最佳权重重载及划分哈希核对均通过。C1 训练 6 轮，第 1 轮最佳，第 6 轮早停，验证 Accuracy 和 Macro-F1 均为 1.0。原始记录见 [实验摘要](../../results/metrics/c1_seed42_20261123_summary.json)与 [Smoke Test](../../results/metrics/c1_seed42_20261123_smoke.json)。

## 2. 自动测试

执行 python -m pytest -q，最终结果为 15 passed，1 warning（12.31 秒）。警告来自已有 anyio pytest 插件对 CallSpec2 的弃用引用。完整记录见 [测试日志](../../results/logs/phase2_tests.log)与 [JUnit XML](../../results/logs/phase2_tests.xml)。后续修复如产生新测试结果，以最新日志为准。

- 既有数据质量、划分、Dataset 和 C0 回归测试通过。
- 新增上传解码、格式与大小检查、知识审核及条件门控、输入变化重置确认测试通过。
- AppTest 执行真实 app.py，三张真实图像在上传控件边界注入 PNG 字节，验证原始预测和六类分数。
- AppTest 验证类别改为 RS 后查询待审核状态，改为 Pa 后等待重新确认并显示暂无可靠知识，新图片清除旧确认。
- AppTest 验证坏图、缺失权重的错误提示与恢复，实验和知识覆盖页面渲染。

AppTest 使用测试夹具验证已审核条目的正确类别查询及来源失效门控。生产条目维持真实待审状态。

## 3. 演示输入

三张演示图来自已有数据中的 Cr_1.bmp、In_240.bmp 和 Sc_141.bmp。演示副本保存成 PNG，位于 .cache/phase2-demo。原始数据和划分保持原版本。真实分数与预测见 [演示输出](../../results/metrics/phase2_demo.json)。

浏览器服务仅监听 127.0.0.1。实际浏览器已验证 10 MB 上传限制、In_240.png 上传并预测为 In、人工修正为 RS 并确认后显示待审核、实验分析及六类知识覆盖页面。截图见 [单图预测](screenshots/prediction.jpg)、[确认后查询](screenshots/confirmed.jpg)、[实验分析](screenshots/experiments.jpg)和[知识覆盖](screenshots/knowledge.jpg)。用户验收、独立成员复现和真实已审核知识展示仍待完成。

## 4. 验收对应

| 验收范围 | 当前证据与状态 |
|---|---|
| AC-01～AC-03 | 第一阶段已有数据与 C0 产物，本次回归测试通过 |
| AC-04 | 固定验证集汇总已完成，独立测试集评价留待最终协议锁定 |
| AC-05～AC-06 | 三张有效图片及坏图、格式、大小、权重异常的自动测试通过，浏览器演示另存 |
| AC-07 | 本机 .venv 可运行网页，干净安装与其他成员复现待完成 |
| AC-08 | C1 配置、权重、日志、曲线、对比已形成 |
| AC-09 | C0 两张错例保留，C1 验证错例为 0，深入分析安排第三阶段 |
| AC-10 | 第二阶段报告正文已生成，正式课程 LaTeX 与提交待安排 |
| AC-11 | 自动测试与本机验证已有记录，外部复现待完成 |
| AC-K01～AC-K02 | 未确认限制、确认查询、修改类别及新图重置的测试通过 |
| AC-K03 | 两条候选定位到原文，独立成员复核待完成 |
| AC-K04～AC-K06 | 降级逻辑、六类覆盖和边界提示已实现，门控自动测试通过 |
