# 第一阶段环境配置记录

日期：2026-11-09（Asia/Shanghai）

## 主机与安装选择

- 项目路径：`E:\steel_defect_project`（移动 SSD）
- Python：3.11.9，64 位；`py` 启动器未识别已安装 Python，因此使用 `python -m venv .venv`
- Git：2.52.0.windows.1；项目目录起初不是 Git 仓库，本阶段未执行 `git init`
- GPU：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB，计算能力 8.9
- NVIDIA 驱动：581.29；`nvidia-smi` 显示 CUDA 13.0
- PyTorch：官方 Windows/Pip CUDA 13.0 构建，torch 2.12.0+cu130、torchvision 0.27.0+cu130。选择依据：[PyTorch 官方安装命令](https://pytorch.org/get-started/previous-versions/)和 [NVIDIA CUDA 兼容性说明](https://docs.nvidia.com/deploy/cuda-compatibility/latest/595/minor-version-compatibility.html)。

## 主要命令

以下命令均在项目根目录执行；安装时的 `PIP_CACHE_DIR`、`TORCH_HOME`、`TMP` 和 `TEMP` 均设置到本项目 `.cache` 下。

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --timeout 120 --retries 5 numpy pandas Pillow scikit-learn matplotlib PyYAML tqdm pytest
& .\.venv\Scripts\python.exe -m pip install --timeout 120 --retries 5 '.cache\wheels\torch-2.12.0+cu130-cp311-cp311-win_amd64.whl' torchvision==0.27.0+cu130 --extra-index-url https://download.pytorch.org/whl/cu130
& .\.venv\Scripts\python.exe scripts\verify_environment.py
& .\.venv\Scripts\python.exe -m pip check
```

官方 torch wheel 因网络读超时，改用项目 `.cache` 内的分段续传工具下载；合并后通过 ZIP CRC 检查，再由 pip 从本地 wheel 安装。首次安装期间移动 SSD 曾短暂断开；重新连接并确认原路径后继续。最终安装与验证均成功。

## 已验证的直接依赖

| 包 | 版本 |
| --- | --- |
| torch | 2.12.0+cu130 |
| torchvision | 0.27.0+cu130 |
| numpy | 2.4.6 |
| pandas | 3.1.1 |
| Pillow | 13.0.0 |
| scikit-learn | 1.9.1 |
| matplotlib | 3.11.2 |
| PyYAML | 6.0.3 |
| tqdm | 4.70.1 |
| pytest | 9.2.0 |

所有上述包均已成功导入，`pip check` 返回 `No broken requirements found.`。`torch.cuda.is_available()` 为 `True`，GPU 张量加法结果为 `[5.0, 7.0, 9.0]`。机器可读的验证结果见 `results/logs/environment_validation.json`。

本阶段未下载数据或训练模型。
