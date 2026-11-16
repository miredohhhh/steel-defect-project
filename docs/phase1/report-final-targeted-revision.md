# 第一阶段论文最后一轮针对性修订

日期：2026-11-09。修订前主文件保存为 MLPTA.before-final-targeted-revision.tex，旧上传包保存为 report_overleaf/phase1_overleaf_upload_before_final_revision.zip。原始数据、三份划分、训练代码与模型权重均未修改。

## 已核实事实与改动

划分程序先将 SHA256 完全重复图像和 4418 对 dHash 初筛候选加入并查集，再以连通组为单位分配集合。报告记录 1511 组、最大组 207 张。复算发现最大组含 Cr 34、In 62、Pa 3、PS 67、RS 1、Sc 40，207 张全部在训练集；候选边跨集合数为 0。旧稿“没有强制把所有候选合并”的说法已更正。分组是防止候选跨集合的保守约束，不代表组内样本均经人工确认重复。

六类工艺表逐条按来源强度修订：Cr 使用热轧钢板裂纹案例并限定材料差异；In 使用连铸坯夹杂表征研究；Pa 去除无直接依据的“碎屑压入”成因；PS 使用热轧机组麻点案例并标明可比性待核；RS 保留氧化层/除鳞案例；Sc 使用 AIST 热轧工艺资料。分类识别论文仅保留为视觉研究背景。Tang 等综述作者已改为 Bo Tang、Li Chen、Wei Sun、Zhong-kang Lin。逐条来源与待人工核查事项见 reference-and-process-audit.md。

工艺知识表方案补充了类别、工序、机制、材料和工况、原文定位、DOI/链接、证据类型、反例、核查建议、审核人及状态；提出双人复核、六类覆盖状态、来源可追溯、类别关联及异常路径检查。模块仍是计划。架构图改为系统功能与数据流，图注明确第一阶段完成范围。

## 编译与版式

使用官方 Tectonic 0.17.0 便携包在项目缓存中编译，下载包 SHA256 与发布清单一致。构建命令：在报告主文件目录运行 tectonic --keep-logs -o report_overleaf/build-final MLPTA.tex（TECTONIC_CACHE_DIR 指向项目 .cache）。退出码 0，生成 7 页 PDF。渲染检查了全部页面；中文、表格、样例图、曲线、混淆矩阵、误分类图和架构图可见。balance 宏包使末页参考文献分列，页底仍有留白；没有 Overfull hbox，存在小于 1.5 pt 的 Overfull vbox。Tectonic 对 Fandol/Latin Modern 字体文件名报告警告，视觉渲染正常，但 PDF 文本提取中的中文字形映射不完整；正式 Overleaf XeLaTeX 版本仍应再编译与检查。

## 人工检查事项

1. 核阅 Pimenov 等论文全文及其与 NEU-CLS PS 的缺陷定义可比性，继续寻找 Pa 的直接工艺证据。
2. 请有工艺背景的组员复核表中候选环节、工况适用范围和知识表审核规则。
3. 在 Overleaf 使用 XeLaTeX 复编，查看末页留白、字号、页眉页脚和文献链接断行；填写两名组员身份。
4. 公开发布报告样例图前核对 NEU-CLS 图片再发布许可。
