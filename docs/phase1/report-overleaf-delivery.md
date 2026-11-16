# 第一阶段 Overleaf 报告交付说明

日期：2026-11-09（Asia/Shanghai）。本文件记录本地提交前核验；Overleaf 端尚需组员自行同步和编译。

## 上传文件

- 使用 `report_overleaf/phase1_overleaf_upload.zip`；内含 `MLPTA.tex`、老师原始 `ieeeconf.cls` 和正文引用的 5 张 PNG。主文件为 `MLPTA.tex`，编译器选 XeLaTeX。
- 本地预览为 `report_overleaf/build-final/MLPTA.pdf`，7 页。压缩包不含原始训练数据、模型权重、虚拟环境或本地构建产物。
- 原始模板备份在 `docs/phase1/MLPTA.teacher-original.tex`，不需上传。

## 事实与排版核验

- 数据划分代码和日志确认 4418 对 dHash 候选形成 1511 个连通组，最大组 207 张、跨六类，整体在训练集；三集合为 1260/270/270。候选是保守分组约束，不能视为全部真实重复。
- C0 记录确认最佳第 8 轮，验证 Accuracy 0.9925925732、Macro-F1 0.9925921353；270 张中两张错误，分别为 In→PS 和 Sc→In。正文和图表使用四舍五入值，不含测试集结果。
- `scripts/check_phase1_latex.py` 已通过，输出 `docs/phase1/report_latex_static_check.json`。本地 Tectonic 编译成功，已渲染检查标题、六类表、样例图、实验图和末页。
- 参考文献共 13 条；Pimenov 作者顺序按期刊卷期目录列示，AIST 的 57、65 为幻灯片编号。不同索引对 Pimenov 末两位作者顺序不一致，最好再看原文首页。

## 提交前人工事项

- 组员二、组员三的真实姓名和学号仍待填写；不要将占位符作为正式署名提交。已填陈家浩 / 202421315022。页眉课程和页脚李乡儒、华南师范大学沿用老师原始模板。
- 在 Overleaf 用 XeLaTeX 再编译，复查字体、参考文献和 7 页分页。Windows 便携 Tectonic 对 Fandol/Latin Modern 报字体文件名警告；预览可见，未发现缺字，但本地不等于 Overleaf 环境。
- 本地日志无 Overfull hbox，有约 1.49 pt 的 Overfull vbox；Underfull 主要出现在长英文网址和六类表的一行，未见文字越界。第 7 页参考文献后留白较多，保留内容与原模板双栏排版，不人为填充。
- NEU-CLS 官方页面未给出明确再分发许可；如要公开发布含原始样例的论文，应先确认图像再发布许可。
