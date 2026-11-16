# 钢材表面缺陷分类识别与辅助分析系统

基于 CNN 与迁移学习的钢材表面六类缺陷分类项目。第一阶段的数据工程和 ResNet18 C0 训练已完成。仓库包含 NEU-CLS 原始数据、固定划分和 C0 最佳权重；克隆后无需重新下载数据或训练 C0。单图推理接口、Streamlit 网页、C1 微调以及“用户确认类别后按确认类别查询已审核工艺知识”属于第二、第三阶段待完成工作。初次接手请先读 [第二阶段交接说明](docs/HANDOVER.md)。

“基于确认缺陷类别的工艺核查建议”已于 2026-11-16 正式纳入[需求基线 V0.3](docs/requirements/钢材表面缺陷分类项目_需求说明与验收标准_V0.3.md)和[三阶段计划 V1.1](docs/plans/钢材表面缺陷分类项目_三阶段实施与报告提交计划_V1.1.md)。该功能当前尚未实现：现有 C0 只能完成六类预测，直接加载或运行 C0 不代表已经具备人工确认、知识查询或工艺建议功能。

## 环境

- Windows 11、Python 3.11、PyTorch + torchvision
- 项目解释器：`.venv\Scripts\python.exe`
- VSCode 已配置上述解释器，并把 pip、Torch 缓存放在项目 `.cache` 目录

克隆仓库后，在项目根目录的 PowerShell 中创建虚拟环境并安装依赖，无需激活虚拟环境：

```powershell
python -m venv .venv
$env:PIP_CACHE_DIR = (Join-Path (Get-Location) '.cache\pip')
$env:TORCH_HOME = (Join-Path (Get-Location) '.cache\torch')
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe scripts\verify_environment.py
& .\.venv\Scripts\python.exe -m pip check
```

创建前先确认使用 Python 3.11；如 `py -3.11` 能识别 Python，也可执行 `py -3.11 -m venv .venv`。依赖清单固定了 Windows CUDA 13.0 构建，原环境使用 NVIDIA GPU；其他设备需要按实际驱动选择兼容的 PyTorch 安装方式。现有 `verify_environment.py` 要求 CUDA 可用，CPU 环境无法通过其中的 GPU 检查。可参考 [PyTorch 官方安装页](https://pytorch.org/get-started/locally/)及[版本命令](https://pytorch.org/get-started/previous-versions/)。

现阶段可直接加载 `checkpoints/c0_seed42_20261109_best.pt` 对单图分类。完整的 PowerShell 预测示例在[交接说明第 5 节](docs/HANDOVER.md#5-克隆后怎么启动和使用现有项目)。仓库目前没有 `app.py` 或 `predict.py`，因此暂时没有可启动的 Streamlit 页面。最佳权重可用 `Get-FileHash checkpoints\c0_seed42_20261109_best.pt -Algorithm SHA256` 校验，预期 SHA256 为 `3BBA7E881F2C523A192F4BC5CEA4A91A981071E5C7E2702C4CAC4A2DEDCCCDA5`。

## 目录

| 路径 | 用途 |
| --- | --- |
| `configs/` | 实验配置 |
| `data/raw/` | NEU-CLS 原始压缩包及解压图片，纳入版本库 |
| `data/splits/` | 固定的数据划分清单，纳入版本库 |
| `src/data/`、`src/models/` | 数据处理和 C0 模型代码 |
| `tests/` | 数据工程和 C0 测试 |
| `checkpoints/` | 最佳 C0 权重纳入版本库；Smoke Test 权重及其他权重忽略 |
| `results/figures/`、`results/metrics/`、`results/logs/` | 图、指标、日志与环境验证报告 |
| `docs/phase1/` | 第一阶段文档 |
| `docs/requirements/钢材表面缺陷分类项目_需求说明与验收标准_V0.3.md` | 当前正式需求与 AC-K01～AC-K06 验收基线 |
| `docs/plans/钢材表面缺陷分类项目_三阶段实施与报告提交计划_V1.1.md` | 当前正式三阶段实施计划 |
| `docs/HANDOVER.md` | 第二阶段交接与验收说明 |
| `scripts/` | 环境验证和后续辅助脚本 |

`report_overleaf/` 的报告由项目负责人单独交付，不纳入 GitHub 仓库。

验证脚本会检查 Python 及各依赖的导入、CUDA 可用性，并在 GPU 上执行张量加法。真实结果（包括失败堆栈）写入 `results/logs/environment_validation.json`。

本机实际版本、GPU 验证结果和安装期间的问题记录见 `docs/phase1/environment-setup.md`。

## 第一阶段：NEU-CLS 数据工程

数据来自[东北大学宋克臣教师主页](https://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/)明确列出的 **NEU-CLS image-only** Google Drive 文件；没有使用 NEU-DET 或 NEU-CLS-64。原始压缩包和解压图像均位于 `data/raw/`，随仓库上传。数据来源页面未载明明确的再分发许可；公开仓库前应向提供者确认授权，仓库访问范围也应由团队决定。

仓库已执行 `git init`。此移动盘文件系统不记录所有权，若 Git 报 `dubious ownership`，可对单次命令使用 `git -c safe.directory=E:/steel_defect_project status`；本项目没有修改全局 Git 配置。

类别映射在 `configs/class_map.json`，数据与预处理配置在 `configs/data.yaml`。克隆后已有原始数据及 `data/splits/*.csv`，可在项目根目录运行以下检查：

```powershell
$env:MPLCONFIGDIR = (Join-Path (Get-Location) '.cache\matplotlib')
& .\.venv\Scripts\python.exe scripts\check_neu_cls.py
& .\.venv\Scripts\python.exe scripts\validate_data_loading.py
& .\.venv\Scripts\python.exe -m pytest -q
```

仅在仓库数据缺失时才使用 `scripts/download_neu_cls.py` 和 `scripts/extract_neu_cls.py`；下载中断可用 `scripts/download_neu_cls.py --resume`。现有划分已固定，不应在接手时重跑 `split_neu_cls.py`。质量检查只生成记录，不删除疑似重复样本。划分使用 seed=42，六类均按 70%/15%/15%；测试集保留给未来最终评价。

Dataset 和 DataLoader 位于 `src/data/dataset.py`。灰度原图被转换为 RGB，再缩放为 224×224，并使用 `ResNet18_Weights.IMAGENET1K_V1` 对应的均值与标准差归一化。训练集仅使用随机水平翻转；验证、测试集没有随机增强。实际统计、划分数量、图表、复核事项及运行日志见 `docs/phase1/data-engineering.md`。

## 第一阶段：ResNet18 C0

C0 冻结预训练 ResNet18 主干及其 BatchNorm 运行统计量，仅训练输出 6 类的 `fc`（3078 个可训练参数）。配置见 `configs/c0.yaml`。固定实验编号 `c0_seed42_20261109` 已完成 13 轮训练，在第 8 轮得到最佳验证 Macro-F1，随后按 patience=5 提前停止。完整指标与限制见 `docs/phase1/c0-training.md`。

下列脚本是第一阶段形成 C0 权重时使用的流程，**不是克隆后启动或预测所需命令**。训练脚本固定实验编号，并拒绝覆盖已有产物；当前仓库已经包含该编号的权重与指标。`fetch_resnet18_weights.py` 用于从官方来源获取并校验预训练权重，现有完整 C0 checkpoint 的加载不需要执行它。

```powershell
& .\.venv\Scripts\python.exe scripts\fetch_resnet18_weights.py
& .\.venv\Scripts\python.exe scripts\smoke_c0.py
& .\.venv\Scripts\python.exe -m pytest -q
& .\.venv\Scripts\python.exe scripts\train_c0.py
& .\.venv\Scripts\python.exe scripts\verify_c0_run.py
```

最佳权重：`checkpoints/c0_seed42_20261109_best.pt`；逐轮指标：`results/metrics/c0_seed42_20261109_history.csv`；曲线：`results/figures/c0_seed42_20261109_curves.png`；训练日志：`results/logs/c0_seed42_20261109_train.log`。运行元数据记录实际依赖版本、类别映射、预处理、随机种子和划分 SHA256。C0 只使用 train 与 val；test 留待最终评价。

## 后续功能：基于确认类别的工艺核查建议

正式流程为：上传单张缺陷图片 → 六分类推理并展示原始预测和六类输出 → 用户确认或修正类别 → 按最终确认类别精确查询结构化知识 → 展示审核通过的候选工序、候选机制、核查事项、来源与适用条件。用户确认前不展示具体工艺机制；修改类别后查询结果必须同步更新。

该模块是 F1 后续交付内容，不阻塞现有 C0 与单图分类 M0。它仅提供可追溯的核查线索，不进行根因诊断、生产参数调整、批次分析，也不使用数据库、知识图谱、大模型或 RAG。截至 2026-11-16，人工确认交互、知识表、查询和展示均尚未实现。
