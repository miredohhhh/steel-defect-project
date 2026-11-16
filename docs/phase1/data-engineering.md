# 第一阶段：NEU-CLS 数据工程进展

记录日期：2026-11-09（Asia/Shanghai）。本阶段没有训练模型，也没有开发 Streamlit 界面。

## 来源、完整性与许可

- 数据集：**NEU-CLS**，东北大学[宋克臣教师主页数据集页面](https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/)列出的分类用 image-only 文件，通过该页面的 Google Drive 链接获取。未使用 NEU-DET、NEU-CLS-64 或镜像。
- 原始包：`data/raw/NEU-CLS.rar`，下载完成于 2026-11-09 14:59:41 +08:00，大小 **50,733,617 字节**，SHA256：`a85bcc6bfa1a40e76f28079e135694aaf5f6a4ee01203f7eaacff0ebafaa96f8`。机器可读来源记录：`results/metrics/neu_cls_download.json`。
- RAR 可完整列出 1800 张 BMP；另有 `Thumbs.db` 和根目录条目。解压后的 1800 张图像均已逐张成功读取。移动盘不支持恢复包内旧时间戳，首次 `tar` 解压对此报错；文件写出后经逐张质量检查确认。解压脚本已改为新解压时不恢复时间戳。
- **使用许可：尚未确认。** 官方页面只给出论文引用建议，没有明确许可证或使用、再分发条款；压缩包也没有许可文件。当前仅做本地数据工程，公开发布原图或再分发前应向数据提供者确认授权。

## 实际质量检查

| 检查项 | 结果 |
| --- | ---: |
| 候选图像 / 可读取图像 | 1800 / 1800 |
| 损坏或不可读取图像 | 0 |
| 图像尺寸 | 全部 200×200 |
| 图像格式 | 全部 BMP |
| 图像模式 / 通道 | 全部 L / 1 通道 |
| Cr / In / Pa / PS / RS / Sc | 各 300 张 |
| SHA256 完全重复 | 1 组、2 张：`Pa_101.bmp` 与 `Pa_105.bmp` |
| dHash 距离 ≤4 的疑似近重复 | 4418 对，其中跨类别 2904 对 |
| 高通纹理相关性 ≥0.4 的优先复核候选 | 2 对 |

dHash 在有相近整体明暗梯度的钢材纹理上会产生较多候选，**4418 对不等于 4418 对已确认重复**。初筛时列为优先人工复核的两对图像是 `In_188.bmp` / `In_210.bmp`、`PS_171.bmp` / `PS_292.bmp`；实际人工结论见下文验收记录。原图均保留，没有自动删除或改标签。全部候选及优先级见 `results/metrics/neu_cls_near_duplicates.csv`；完全重复和异常记录分别见 `neu_cls_exact_duplicates.csv`、`neu_cls_invalid_images.csv`。

## 划分决策与实际数量

固定 seed=42，以每类 70%/15%/15% 为目标。为防止未确认候选跨集合泄漏，将 SHA256 完全重复和所有 dHash 初筛候选连成组，同组只进入一个集合；共 **1511 组**，最大组 **207 张**。这是一项保守的分组决定，不代表组内图像都已确认为重复。

| 类别 | train | val | test |
| --- | ---: | ---: | ---: |
| Cr | 210 | 45 | 45 |
| In | 210 | 45 | 45 |
| Pa | 210 | 45 | 45 |
| PS | 210 | 45 | 45 |
| RS | 210 | 45 | 45 |
| Sc | 210 | 45 | 45 |
| **合计** | **1260** | **270** | **270** |

三个 CSV 的路径相对于 `data/`，列为 `image_path,class_name,label`。路径交叉数、完全重复组跨集合数、疑似近重复候选跨集合数均为 **0**。划分见 `data/splits/{train,val,test}.csv`，报告见 `results/metrics/neu_cls_split_report.json`。测试集只用于未来最终评价，不参与调参。

## 训练前人工复核验收（2026-11-09）

读取 `docs/phase1/duplicate_manual_review.csv` 的两条已填写记录：**duplicate 2 对、not_duplicate 0 对、uncertain 0 对**。两对 `duplicate` 均属于原高优先级候选，且没有跨划分。

| 人工确认的近重复图片对 | 判定 | 所在集合 |
| --- | --- | --- |
| `In_188.bmp` / `In_210.bmp` | duplicate | test / test |
| `PS_171.bmp` / `PS_292.bmp` | duplicate | val / val |

原有完全重复组 `Pa_101.bmp` / `Pa_105.bmp` 同在 **train**，两张原图的 SHA256 已重新计算并与完全重复清单一致。三个划分合计 **1800 个唯一图片路径**，train/val/test 分别为 **1260/270/270**；每类分别为 **210/45/45**。类别映射经核对仍为 `Cr=0, In=1, Pa=2, PS=3, RS=4, Sc=5`，所有清单行的类别名、数字标签和实际图片路径均有效。

**验收结论：通过，可基于现有划分继续本地 C0 模型训练。** 本次没有删除原图或调整划分。已确认的近重复虽未跨集合，但分别保留在 val 和 test 内部；未来解释验证和最终测试指标时应披露这一点，测试集仍只用于最终评价。

## 后续进展：C0 首轮训练（2026-11-09）

基于本次验收通过的固定划分，已完成 ResNet18 C0 的 Smoke Test 和首轮训练。实验 `c0_seed42_20261109` 在第 8 轮取得最佳验证 Accuracy 0.9925925732、Macro-F1 0.9925921353；第 13 轮按 patience=5 提前停止。最佳权重重新加载及产物一致性核验通过，原始数据、类别映射和三份划分文件未修改。逐轮指标、运行环境、权重及问题记录见 [C0 训练记录](c0-training.md)。测试集仍未用于训练或评价。

## 加载与验证

图像转换成 RGB，缩放到 224×224，并使用 `ResNet18_Weights.IMAGENET1K_V1` 的均值 `[0.485, 0.456, 0.406]` 和标准差 `[0.229, 0.224, 0.225]` 归一化。训练增强仅为概率 0.5 的水平翻转；验证和测试没有随机增强。三套 DataLoader 均成功读取 `[32, 3, 224, 224]` 的 `float32` 批次，完整清单标签覆盖 0–5，验证和测试重复读取结果一致。加载报告见 `results/metrics/neu_cls_loading_report.json`。

## 产物与命令

- 图：`results/figures/neu_cls_examples.png`、`neu_cls_class_distribution.png`、`neu_cls_grayscale_histogram.png`、`neu_cls_train_augmentation.png`。图中数值与样本均来自本次实际运行。
- 指标：`results/metrics/neu_cls_quality.json`、图像清单和类别统计 CSV、异常/重复候选 CSV、下载与划分/加载 JSON。
- 终端日志：`results/logs/neu_cls_quality.log`、`neu_cls_split.log`、`neu_cls_loading.log`、`pytest_phase1.log`。
- 命令：`scripts/check_neu_cls.py` → `scripts/split_neu_cls.py` → `scripts/validate_data_loading.py` → `python -m pytest -q`，均通过 `.venv/Scripts/python.exe` 执行。具体 PowerShell 命令见 README。
- pytest：**4 passed in 10.55s**（最终复跑）。

## 待人工复核

1. 向数据提供者核实许可证与是否允许公开再分发原图。
2. 两对高优先级近重复候选已由人工判为 duplicate，且全部已确认重复均未跨集合。其余 dHash 候选尚未逐对人工确认，保存在 CSV 中，可按高通纹理相关性排序复核。若将来决定排除或重新标注样本，必须重新运行质量检查与划分并记录新数量。
3. 当前工作区没有单独的“项目需求说明”或“三阶段实施计划”文件；本阶段依据现有 README 和本次任务条目实施。
