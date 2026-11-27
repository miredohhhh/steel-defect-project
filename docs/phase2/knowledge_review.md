# 工艺知识候选条目复核

核阅日期为 2026-11-23。本次访问 AIST 原始培训 PDF，定位到 RS 与 Sc 的相关幻灯片。生产知识表中两条记录均为 pending，当前正式审核通过数量为 0。

## 1. 原文及候选条目

来源为 AIST 的 Principles of Hot Rolling，2020 年培训资料。资料按每页两张幻灯片排版，幻灯片编号与 PDF 页码分别保存。

- RS-AIST-057 对应 PDF 第 29 页上半页、幻灯片 57 的 IMPROPER DESCALING。资料讨论除鳞喷嘴堵塞或损伤与氧化铁皮随后压入表面的联系。候选核查对象包括喷嘴检查与除鳞记录。
- Sc-AIST-065 对应 PDF 第 33 页上半页、幻灯片 65 的 HIGH TEMPERATURE SCRATCHES。资料讨论侵入轧制路径的接触物与高温划痕。候选核查对象包括接触路径和形貌观察。

[AIST 原始培训资料](https://www.aist.org/AIST/aist/AIST/Conferences_Exhibitions/Training_Seminars/Training%20Presentations/10-Hot-Rolling-Processes-msts-20.pdf)

这些表述仅用于候选核查，适用材料与工况需要独立成员复核。Cr、In、PS 延续第一阶段条件性资料线索，全文或可比性待核实，Pa 保留证据缺口。六类状态见[知识表](../../data/knowledge/process_knowledge.json)。

## 2. 成员复核操作

请录入者之外的一名成员逐条查看原文，检查类别定义、资料解释、工况适用性和限制，并在知识表中填写真实身份及日期。

1. 核对 id、class_code、process、mechanism 和 checks 与原文关系。
2. 补充 entered_by 为真实录入或整理负责人，reviewer 为独立复核人，两者须不同。
3. 只有通过审核才将 review_status 改为 approved，填写 ISO 日期 review_date。未通过使用 rejected，未完成保持 pending。
4. 同时复查 source_status、source_checked_at 和 evidence_location。来源失效时设置 unavailable，并记录原因。
5. 运行知识测试及 scripts/build_phase2_report.py，更新审核条目计数与阶段报告，再验证网页展示。报告中审核状态须与真实记录一致。

网页读取 JSON 时校验审核人及日期，正式查询还要求来源状态有效、类别已确认、用户已核对适用条件。测试夹具中的虚构审核人仅用于验证查询门控，生产记录保持待审。

## 3. 复核清单

| 条目 | 原文位置已定位 | 独立成员审核 | 真实页面正式展示 |
|---|---|---|---|
| RS-AIST-057 | 是 | 待完成 | 待审核后验证 |
| Sc-AIST-065 | 是 | 待完成 | 待审核后验证 |
