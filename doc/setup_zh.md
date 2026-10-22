# 安装说明

[English](setup_en.md) | [项目说明](../README_zh.md)

使用 Python 3.11，在仓库根目录执行以下命令。GPU 训练需要 NVIDIA GPU、兼容的
驱动和支持 CUDA 的 PyTorch。

## 创建环境

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

依赖版本由 [setup.py](../setup.py) 定义；安装使用其中已有的版本锁定，不另设一组
依赖版本。若已有配置好的环境，可只注册项目而不改动依赖：

```bash
.venv/bin/python -m pip install --no-deps -e .
```

发行包名为 `supermimic_mjlab`，Python 导入路径为 `src.tasks.supermimic`。
机器人定义和网格资源来自已安装的 mjlab，无需本地 Unitree SDK2 部署模拟器。

## 验证环境

```bash
.venv/bin/python - <<'PYTHON'
from importlib.metadata import version
import torch

for package in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "scipy"):
    print(package, version(package))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PYTHON
```

训练配置和 checkpoint 也会记录实际运行版本。GPU 训练前，应确认启动训练使用的
同一个 Python 环境能够访问 CUDA。

## 准备动作与训练

动作数据和训练产物不纳入版本控制。将 G1 重定向 LAFAN CSV 放在
`src/assets/motions/LAFAN1_Retargeting_Dataset/`，按[数据转换说明](lafan_conversion_zh.md)
生成 NPZ 动作文件和 manifest。

缩小规模、论文规模训练、断点恢复和评估命令见 [README](../README_zh.md#在线训练)，
回归检查见[测试说明](../README_zh.md#测试)。测试在临时目录生成小型动作与 Replay；
真实数据集检查还需要准备转换后的 LAFAN 数据集。
