# 项目交接说明：第二阶段接手指南

更新日期：2026-11-16。本文给第一次接触项目的同学阅读。项目目前完成第一阶段，下面提到的第二、三阶段功能是计划，不能当成现有功能。GitHub 仓库计划包含原始 NEU-CLS 数据与 C0 最佳权重；`report_overleaf/` 由项目负责人单独发送。

## 1. 一句话认识项目

输入一张**已经有钢材表面缺陷**的图片，用图像分类模型判断它最像六种已知缺陷中的哪一种，并在后续做一个便于演示的网页。项目研究的是“缺陷属于哪一类”，不是判断钢材是否合格，也不定位缺陷在图片中的位置。

NEU-CLS 数据集只有缺陷图片，没有正常钢材样本，也没有真实生产参数或经核实的缺陷根因。因此模型不能用于正常/异常筛查，更不能根据单张图片自动给出确定的工艺调整指令。

## 2. 术语速查

| 项目称呼 | 含义 |
| --- | --- |
| NEU-CLS | 本项目使用的钢材表面缺陷**图像分类**数据集；六类共 1800 张灰度图，每类 300 张。不要与用于其他任务的 NEU-DET、NEU-CLS-64 混用。 |
| 六类标签 | `Cr` = Crazing（裂纹状缺陷），`In` = Inclusion（夹杂），`Pa` = Patches（斑块），`PS` = Pitted Surface（麻点表面），`RS` = Rolled-in Scale（轧入氧化皮），`Sc` = Scratches（划痕）。数字编号依次是 0～5，以 `configs/class_map.json` 为准。 |
| ResNet18 | 用来提取图片特征的卷积神经网络；本项目从 ImageNet 预训练参数开始。 |
| C0 / C1 | 同一 ResNet18 的两种训练方案。C0 冻结原有主干，仅训练最后的六分类层 `fc`；计划中的 C1 从最佳 C0 权重出发，再解冻 `layer4` 和 `fc` 做部分微调。C1 目前尚未实现。 |
| 迁移学习 | 利用模型先前从 ImageNet 图片学到的通用特征，再适配本项目的六种缺陷。 |
| train / val / test | 训练集用于更新模型参数；验证集用于选择模型和比较方案；测试集留到最终方案固定后评价。当前分别有 1260 / 270 / 270 张。 |
| Macro-F1 | 先分别计算六类的 F1，再取平均；每一类权重相同，用于避免只看总体正确率。 |
| checkpoint / 最佳权重 | 训练后保存的模型参数文件。本项目已保存 `checkpoints/c0_seed42_20261109_best.pt`，克隆包含该文件的仓库后可直接加载 C0。 |
| Smoke Test | 用少量真实数据检查前向计算、梯度和权重保存/加载是否正常；不是正式训练结果。 |
| Streamlit / MVP | Streamlit 是计划用于本地网页演示的 Python 工具；MVP 指能上传单图、看到预览、六类输出和提示的最小可用版本。目前没有网页。 |
| 工艺知识关联 | V0.3 / V1.1 **建议稿**提出的扩展：按预测类别查询人工审核的资料，展示待核查的工艺线索及来源。不是已实现功能，也不是根因诊断；正式范围待团队与教师确认。 |

## 3. 项目怎样解决问题

目前的实际流程是：获取 NEU-CLS 图片 → 检查图片与重复情况 → 固定 train/val/test 清单 → 把灰度图转成 RGB、缩放为 224×224 并归一化 → 用训练集训练 C0 → 在验证集选最佳权重并保存指标。训练集另有随机水平翻转；验证和测试集没有随机增强。`configs/data.yaml` 保存数据与预处理设置，`configs/c0.yaml` 保存 C0 训练设置。

下一步要在这个基础上做单图推理接口和网页：上传图片 → 使用与验证集相同的预处理 → 加载已保存的 C0/C1 权重 → 输出六类分数及预测标签。网页每次上传只做推理，不启动训练。Softmax 输出可展示为六类模型概率，但它不是经过校准的“预测正确概率”。

## 4. 已完成的第一阶段与证据

- 数据检查：1800 张图片均可读取；六类各 300 张。固定划分为每类 train 210、val 45、test 45，划分文件在 `data/splits/`。完全重复和 dHash 初筛候选没有跨集合。两对人工确认的近重复图片分别位于 val 内部和 test 内部，解释指标时应披露。细节见 [数据工程记录](phase1/data-engineering.md)。
- C0：实验编号 `c0_seed42_20261109`，训练 13 轮，第 8 轮为最佳；验证集 Accuracy 约 **0.992593**，Macro-F1 约 **0.992592**。这只是该固定验证集上的结果，不代表真实生产线准确率，更不是测试集结果。细节见 [C0 训练记录](phase1/c0-training.md)。
- 已保存最佳权重、逐轮指标、训练曲线、日志和验证集预测。权重重新加载的一致性已经在第一阶段核对。最佳权重的 SHA256 为 `3BBA7E881F2C523A192F4BC5CEA4A91A981071E5C7E2702C4CAC4A2DEDCCCDA5`。
- **尚未完成**独立的单图推理脚本、Streamlit 页面、C1 训练、C0/C1 对比、独立测试集最终评价和工艺知识模块。C0 已可按第 5 节示例直接加载并预测；Streamlit 网页是第二阶段接手同学的开发任务。不要把 `results/metrics/*val*` 中的验证结果写成测试结果。

## 5. 克隆后怎么启动和使用现有项目

以下命令在 **Windows PowerShell 的项目根目录**运行。建议 Python 3.11。`requirements.txt` 固定了 PyTorch CUDA 13.0 构建；原环境是 Windows 11 + NVIDIA RTX 4060 Laptop GPU。其他显卡驱动或无 NVIDIA GPU 的电脑，应先按设备情况安装兼容的 PyTorch，再安装其余依赖；现有 `scripts/verify_environment.py` 包含 CUDA 检查，因此 CPU 环境不能以该检查通过作为启动条件。环境细节见 [环境记录](phase1/environment-setup.md)。

```powershell
git clone <GitHub 仓库地址>
cd <克隆后的项目目录>
python --version                         # 确认 Python 3.11
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip check
& .\.venv\Scripts\python.exe scripts\verify_environment.py
```

克隆后应已有 `checkpoints/c0_seed42_20261109_best.pt`。先检查文件完整性；输出哈希须与第 4 节相同：

```powershell
Get-FileHash checkpoints\c0_seed42_20261109_best.pt -Algorithm SHA256
```

**不必重新训练 C0。** 目前没有 `app.py` 或 `predict.py`，所以不能执行 `streamlit run app.py`；Streamlit 页面将由第二阶段开发。下面的 PowerShell 示例可以直接对仓库内的 NEU-CLS 图片或你提供的一张 JPEG/PNG/BMP 图片运行现有 C0 模型；把最后的路径换成真实图片路径。模型权重已包含主干与分类层，`pretrained=False` 不会另行下载 ImageNet 权重。

```powershell
@'
import sys
from pathlib import Path
import torch
from PIL import Image
from src.data.dataset import build_transforms, load_project_config
from src.models.c0 import C0ResNet18

root = Path.cwd()
config, class_map = load_project_config(root)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
payload = torch.load(root / "checkpoints/c0_seed42_20261109_best.pt",
                     map_location="cpu", weights_only=True)
model = C0ResNet18(pretrained=False)
model.load_state_dict(payload["model_state_dict"], strict=True)
model = model.to(device).eval()
with Image.open(sys.argv[1]) as image:
    tensor = build_transforms("test", config)(image.convert("RGB")).unsqueeze(0).to(device)
with torch.inference_mode():
    probabilities = model(tensor).softmax(dim=1)[0].cpu()
names = {index: name for name, index in class_map.items()}
for index in range(6):
    print(f"{names[index]}: {probabilities[index].item():.4f}")
print("预测类别:", names[int(probabilities.argmax())])
'@ | & .\.venv\Scripts\python.exe - "data\raw\NEU-CLS\Cr_1.bmp"
```

仓库还包含 `data/raw/NEU-CLS.rar`、解压图片和固定划分清单，克隆后可直接进行数据检查、测试和第二阶段开发。以下检查可在环境装好后运行；原始数据的来源和校验信息见 [数据工程记录](phase1/data-engineering.md)：

```powershell
& .\.venv\Scripts\python.exe scripts\check_neu_cls.py
& .\.venv\Scripts\python.exe scripts\validate_data_loading.py
& .\.venv\Scripts\python.exe -m pytest -q
```

已有 `data/splits/*.csv` 是第一阶段固定的划分；通常无需重跑 `split_neu_cls.py`。如果确需修改划分，先与团队确认，并同步更新实验记录，否则新旧指标不可直接比较。仅当克隆后发现数据缺失时，才按 [README](../README.md) 的说明运行下载与解压脚本。NEU-CLS 来源页未给出明确的再分发许可；公开仓库前应确认授权。

## 6. 第二阶段你要做什么

原计划第二阶段为 **2026-11-10 至 2026-12-01 08:00**，以教师最新通知为准。第一阶段的 C0 训练已经完成，接手后先使用现有最佳权重贯通推理和界面，再开展 C1；不要为了启动网页重新训练 C0。

| 顺序与计划节点 | 任务 | 要交出的可检查成果 |
| --- | --- | --- |
| 接手后先做；原计划 11/22 内部演示 | 核对仓库权重、环境和数据；封装单图预测接口，复用 `data.yaml` 预处理与固定类别映射；实现 Streamlit 上传、预览、预测标签和六类输出；处理坏图、格式不支持、权重缺失。 | 可运行的单图页面、启动命令、至少 3 张有效图片演示记录与异常输入记录。 |
| 原计划 11/23～11/27 | 从最佳 C0 权重开始实现 C1，只解冻 `layer4` 和 `fc`，用较低学习率微调；沿用相同划分与指标，在验证集比较 C0/C1。 | C1 配置、唯一实验编号、最佳权重、逐轮日志、验证指标和有依据的对比。C1 是计划中的**新训练**，不等于重训 C0。 |
| 原计划 11/28～11/30 | 整合页面、真实结果、图表、异常测试和第二阶段报告；核对文案与代码一致。 | 页面截图、运行说明、结果表、问题与未完成事项、第二阶段报告。 |
| 若团队确认 V0.3/V1.1 建议 | 增加结构化工艺知识表，按类别展示已复核的资料及其适用工况、来源和“仅供核查”提示。 | 可追溯条目、审核状态、来源定位及异常/无可靠关联时的提示；不要宣称根因诊断准确率。 |

### 模型对比到底比较什么

**计划必做的比较是 C0 与 C1**：两者使用相同的 ResNet18、六类标签、固定数据划分和评价指标，比较仅训练分类层与进一步微调 `layer4` 的结果、耗时和可训练参数。由于 C1 从 C0 接着训练，它还获得了额外训练时间；报告不能把差异完全归因于“解冻更多层”。

**交接安排：第二阶段不增加第三个模型；先交付 C1、C0/C1 对比及网页。第三阶段预留实验 B 作为优先的额外对照**，由模型负责人在开始第三阶段时确认算力、时间和教师新要求。选择 B 是为了在相同 ResNet18 结构下研究预训练的作用；这是一项团队计划，并非教师规定必须有第三个模型。若 C1、网页或基本验收未完成，优先补齐主线，并在阶段记录中明确写出 B 未开展的原因，不能把计划写成实验结果。

原计划的可选方向如下；A 和 D 作为备选，不要同时开展多个扩展：

| 可选方案 | 来源与对照问题 |
| --- | --- |
| 实验 A：简单 CNN | 自行搭建较简单的卷积网络，从随机参数开始训练，比较基础 CNN 与迁移学习方案的表现。网络结构也不同，不能把性能差异全部归因于预训练。 |
| 实验 B：随机初始化 ResNet18 | 使用与 C0 同类的 ResNet18 结构，但不加载 ImageNet 预训练参数，比较预训练是否有帮助；应尽量控制训练预算等条件。 |
| 实验 D：数据增强消融 | 在固定模型方案上改变一项训练增强，比较该增强是否有帮助；这是训练策略对照，不是新网络结构。 |

若第三阶段启动 B，模型负责人应先写下与哪一方案对比、两边哪些参数可训练、训练轮数/时间预算和评价方式，再开跑。不能把“随机初始化后训练整个 ResNet18”与“冻结主干的 C0”之间的差异全解释为预训练效果，因为可训练参数也改变了。建议沿用固定划分、预处理和验证指标，保存 B 的配置、权重、逐轮日志及与 C0/C1 的对比表。所有模型确定后再用保留测试集作最终评价。实验 A/B/D 的原始定义见 [三阶段计划 V1.0](plans/钢材表面缺陷分类项目_三阶段实施与报告提交计划_V1.0.md)。

任务安排可由团队分为数据与来源、模型与接口、界面与报告三块；具体负责人应由成员确定。先约定输入预处理、类别映射、权重格式和接口返回值，再并行开发，避免页面与模型输出对不上。

## 7. 怎样判断第二阶段做对了

下面是把原 [需求与验收标准 V0.2](requirements/钢材表面缺陷分类项目_需求说明与验收标准_V0.2.md) 和 [三阶段计划 V1.0](plans/钢材表面缺陷分类项目_三阶段实施与报告提交计划_V1.0.md) 转为接手时可操作的检查项；知识关联的要求来自待确认的 [调整建议 V1.1](plans/三阶段计划调整建议_V1.1.md)。

| 检查项 | 通过表现与证据 |
| --- | --- |
| 数据与模型能接上 | `data/splits/` 和类别映射未被无记录地改动；现有 C0 最佳权重可加载；新同学能说明训练、验证、测试各自用途。 |
| 单图预测 | 至少 3 张有效 JPEG/PNG 图片均有预览、六类分数和预测类别；六个 Softmax 值在 0～1，和约为 1（误差不超过 `1e-4`），预测类别对应最大值。 |
| 网页和异常处理 | 本地 Streamlit 能启动；不支持格式、损坏图片、权重缺失有明确提示，失败后页面仍可继续使用；上传不会触发训练。 |
| 实验评价 | C0/C1 对比使用相同划分、预处理及指标定义；保存配置、权重、日志、验证集逐类指标、曲线和混淆矩阵。报告说清 C1 增加了训练预算，不能把差异只归因于解冻层。若 C1 下降，也如实记录。 |
| 测试集纪律 | 目前测试集未评价。按现有阶段计划，应在模型和评价协议确定后、第三阶段进行最终测试；第二阶段报告使用验证集结果时明确标注 `val`，不要用 `test` 反复挑选方案。 |
| 交付可复现 | README 写出适用环境、安装/启动命令、模型文件位置；其他同学按步骤能启动页面并得到预测，报告中的数字能在仓库产物中找到。 |
| 知识关联（仅在确认纳入范围后） | 六类均有覆盖状态；展示条目有原文位置、来源、适用条件、审核人和状态；只展示已复核条目，缺少证据时明确提示“暂无可靠工艺关联”。 |

完整课程验收编号是 AC-01～AC-11；其中 AC-04 的**独立测试**和更深入的错误分析、最终报告属于后续最终验收。第二阶段须如实标注已完成与待完成内容，具体提交形式与截止日期以教师最新通知为准。

## 8. 文件从哪里找、遇到问题看哪里

| 位置 | 用途 |
| --- | --- |
| `README.md` | 当前项目运行命令与第一阶段概览。 |
| `configs/class_map.json`、`configs/data.yaml`、`configs/c0.yaml` | 类别编号、图片预处理、C0 参数。 |
| `src/data/dataset.py`、`src/models/c0.py` | 图片读取/转换与 C0 网络结构；第二阶段接口应复用这里的约定。 |
| `scripts/train_c0.py`、`scripts/verify_c0_run.py` | 已有 C0 训练和产物核验实现；训练脚本固定实验编号且拒绝覆盖旧产物。 |
| `checkpoints/c0_seed42_20261109_best.pt` | 可直接推理的 C0 最佳权重。`*_smoke.pt` 不用于正式预测。 |
| `data/raw/NEU-CLS.rar`、`data/raw/NEU-CLS/` | 随仓库交付的数据压缩包和解压图片；`data/splits/` 为固定划分。 |
| `results/metrics/`、`results/figures/`、`results/logs/` | 实际指标、曲线和运行记录；报告应引用这些真实产物。 |
| `docs/phase1/` | 环境、数据工程、C0 训练和第一阶段报告记录。 |

开始工作前，先确认 GitHub 仓库确实包含最佳权重和 `data/raw/NEU-CLS/`，校验权重哈希，再用第 5 节的样本图片跑通预测。若依赖安装失败，优先核对 Python、CUDA 构建与显卡驱动；若图片找不到，检查原始数据是否随仓库完整克隆；若标签或分数异常，核对 `class_map.json` 和推理预处理是否与 `data.yaml` 一致。
