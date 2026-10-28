# 安装说明

[English](setup_en.md) | [项目说明](../README_zh.md)

在仓库根目录执行以下命令，推荐使用 `uv sync` 自动创建环境并安装项目。
GPU 训练需要 NVIDIA GPU、兼容的驱动和支持 CUDA 的 PyTorch。

## 使用 uv 快速启动

```bash
uv sync
source .venv/bin/activate
```

`uv sync` 读取 [pyproject.toml](../pyproject.toml)，按 `.python-version` 选择
Python 3.11，并以 editable 模式安装项目。`uv.lock` 锁定解析后的依赖；
需要严格使用已提交的锁文件时执行 `uv sync --locked`。无需单独创建环境或执行 editable 安装。

需要 W&B 和 Rich 日志功能时执行：

```bash
uv sync --extra logging
```

每次打开新终端后，先执行 `source .venv/bin/activate`。后续命令统一使用
该环境中的 `python` 和 `wandb`。

项目版本和直接依赖锁定保留在 [setup.py](../setup.py)，由 pyproject.toml 的动态
元数据直接读取，避免重复声明。同步安装使用 uv.lock 中解析后的依赖集合。若已有配置好的环境，可只注册项目而不改动依赖：

```bash
uv pip install --no-deps -e .
```

发行包名为 `supermimic_mjlab`，Python 导入路径为 `src.tasks.supermimic`。
G1 XML 和网格资源保存在仓库的 [`src/assets/robots/g1`](../src/assets/robots/g1/README.md)，
mjlab 提供训练使用的 PD 执行器，无需本地 Unitree SDK2 部署模拟器。
模型指纹仅用于溯源，不再阻止加载；仍要求机器人关节／刚体顺序、拓扑、网络维度和仿真
配置兼容。复现历史成绩仍需使用原始配置和资产。

## 验证环境

```bash
python - <<'PYTHON'
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

[README](../README_zh.md#在线训练) 提供发布模型的训练配置和播放入口，缩小规模、论文规模、
checkpoint 和评估说明见[操作指南](training_and_playback_zh.md)，
回归检查见其中的[测试说明](training_and_playback_zh.md#测试)。测试在临时目录生成小型动作与 Replay；
真实数据集检查还需要准备转换后的 LAFAN 数据集。
