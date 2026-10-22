# SuperMimic Mjlab

[English](README.md) | 简体中文

基于 mjlab、MuJoCo 和 MuJoCo Warp 的 SuperMimic 复现，面向 Unitree G1 的多动作跟踪。
方法参考 [SuperMimic 论文](doc/supermimic.pdf)：当前策略采集真实仿真数据，世界模型
用 8 步自回归窗口学习动力学，再通过 32 步可微想象联合训练 Actor 和 Critic。

当前使用 G1 重定向的 LAFAN 数据集，共 40 段动作、约 2.45 小时，控制频率为 50 Hz。
已实现在线训练、断点恢复和固定起点真实仿真评估；策略跟踪效果需通过完整动作评估确认。

## 运行环境

以下命令均在仓库根目录使用已配置的 `.venv/bin/python`，Python 版本为 3.11。
核心依赖为支持 CUDA 的 PyTorch、mjlab、MuJoCo、MuJoCo Warp 和 Warp，依赖声明见
[setup.py](setup.py)。检查当前 Python 是否能使用 GPU：

```bash
.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

训练和推理命令使用 `taskset -c 0-7,10-31`。下文将 Matplotlib 和其他缓存写入 `/tmp`。

## 代码结构

参考其他 task 的 `config/`、`mdp/`、`rl/` 分类，并将数据和模型分别放入独立子包：

```text
src/tasks/supermimic/
├── config/
│   ├── task_cfg.py           # G1 状态维度、动作库路径和环境运行参数
│   └── rl_cfg.py             # OnlineConfig：训练规模、网络宽度与优化参数
├── data/
│   ├── motion_library.py     # NPZ 动作库、参考帧与未来窗口
│   ├── replay_buffer.py      # 带 episode 边界的回放与历史重建
│   └── world_model_data.py   # 离线数据划分、去重与 ReplayCorpus
├── mdp/
│   ├── state.py              # 状态布局、四元数运算、观测与传感器噪声
│   ├── rewards.py            # 可微跟踪奖励
│   ├── terminations.py       # 物理失败判定
│   └── reset_sampler.py      # 按回报分配重置起点
├── models/
│   ├── kinematics.py         # 可微 G1 运动学与机器人元数据
│   ├── world_model.py        # 动力学及任务预测头
│   └── policy.py             # MotionActor、MotionCritic 与参考特征
├── rl/
│   ├── imagination.py        # BPTT、软生存权重、GAE 与 Critic 损失
│   ├── online_training.py    # 采集、世界模型拟合、Actor–Critic 更新
│   ├── world_model_training.py # 独立世界模型训练与验证
│   └── evaluation.py         # 固定起点完整动作评估
├── env.py                    # SuperMimicEnv：真实仿真接口
└── env_cfg.py                # mjlab 场景、PD 动作与域随机化
```

各子包的 `__init__.py` 保持轻量，回归测试位于 `tests/`。脚本按使用频率和用途组织：

```text
scripts/
├── train_supermimic.py            # 在线联合训练
├── evaluate_supermimic.py         # checkpoint 策略评估
├── data/
│   └── convert_lafan_to_npz.py     # LAFAN CSV 转 NPZ
└── offline/
    ├── collect_supermimic.py      # 采集快照或运动学回放校验
    └── train_world_model.py       # 独立世界模型训练与验证
```

日常在线训练和评估使用根目录的两个入口。`offline/` 中的脚本用于诊断和离线实验，
在线联合训练无需先运行这些脚本。导入示例：

```python
from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.online_training import OnlineTrainer
```

## 准备动作数据

原始 CSV 位于 `src/assets/motions/LAFAN1_Retargeting_Dataset/`。
尚未转换时运行：

```bash
.venv/bin/python scripts/data/convert_lafan_to_npz.py \
  --input-dir src/assets/motions/LAFAN1_Retargeting_Dataset \
  --output-dir src/assets/motions/g1/lafan1_npz \
  --input-fps 30 --output-fps 50
```

输出为每段动作的 NPZ 和 `manifest.json`。当前已准备好这些文件，可直接训练。
转换字段与坐标约定见 [LAFAN 数据转换说明](doc/lafan_conversion_zh.md)。

## 在线训练

### 当前硬件使用的缩小规模

使用默认网络、64 个环境、每轮 10×256 条世界模型窗口和 64 条想象轨迹：

```bash
taskset -c 0-7,10-31 env MPLCONFIGDIR=/tmp/supermimic-mpl XDG_CACHE_HOME=/tmp/supermimic-cache \
  .venv/bin/python scripts/train_supermimic.py \
  --device cuda:0 --iterations 1000 \
  --output outputs/supermimic/online_training
```

默认启用域随机化、观测噪声和按回报采样的重置。
不指定 `--clips` 时使用全部动作；调试时可指定 `--clips walk1_subject1 run1_subject2`。

### 论文单 GPU 训练规模

在仓库根目录运行以下命令，使用论文 Table A.4–A.5 的网络宽度、训练批量、
环境数量与训练轮数。默认读取已转换的
`src/assets/motions/g1/lafan1_npz/manifest.json`，使用全部 40 段 LAFAN 动作，
从头联合训练世界模型、Actor 和 Critic，无需先运行独立世界模型预训练。

```bash
taskset -c 0-7,10-31 env MPLCONFIGDIR=/tmp/supermimic-mpl XDG_CACHE_HOME=/tmp/supermimic-cache \
  .venv/bin/python scripts/train_supermimic.py \
  --device cuda:0 \
  --num-envs 4096 --capacity 1000 \
  --collect-steps 5 --updates-per-collect 2 \
  --wm-batches 10 --wm-batch-size 2048 \
  --policy-batch-size 4096 --horizon 32 \
  --critic-epochs 5 --critic-minibatches 4 \
  --world-dims 2048 2048 2048 \
  --actor-dims 2048 1024 512 256 \
  --critic-dims 256 128 \
  --iterations 320000 \
  --eval-interval 320000 --checkpoint-interval 10000 \
  --output outputs/supermimic/lafan_paper_scale
```

每采集 5 个向量步执行 2 轮更新，对应 update-to-data ratio `0.4`。
世界模型窗口长度固定为 8；默认学习率 `2e-4`、折扣 `0.99`、GAE λ `0.95`、
探索动作噪声标准差 `0.1`，学习率在 250,000 轮内余弦下降至初值的 `0.03` 倍，
之后保持。评估和保存间隔是本命令的运行安排，并非论文超参数；脚本仍会在初始化
及训练结束时评估。输出目录必须不存在。

此命令对齐训练规模，不代表所有实现细节均已严格复现论文：当前仍保留 10 秒
真实采集 episode 时限，但想象不受该时限限制，保留后续价值 bootstrap。
世界模型和 Critic 损失均在有效时间步上求和、在窗口或轨迹批次上取平均。
完整规模尚未在当前 16 GiB GPU 上验证，显存需求也包含 Replay 和 32 步反向传播计算图。

### 断点恢复

恢复时从 checkpoint 读取原有网络与 batch 参数；`--iterations` 是包含已完成轮数的总轮数：

```bash
taskset -c 0-7,10-31 env MPLCONFIGDIR=/tmp/supermimic-mpl XDG_CACHE_HOME=/tmp/supermimic-cache \
  .venv/bin/python scripts/train_supermimic.py \
  --device cuda:0 --resume outputs/supermimic/online_training/last.pt \
  --iterations 5000
```

新训练的输出目录必须不存在。主要输出如下：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 训练配置、实际运行版本和数据/机器人签名 |
| `metrics.jsonl` | 采集、世界模型、Actor、Critic 与耗时指标 |
| `last.pt` | 最新网络、优化器、Replay、采样器和 RNG 状态 |
| `best.pt` | 固定起点真实仿真评估回报最高的 checkpoint |
| `iteration_*.pt` | 定期保存的训练断点 |
| `evaluation*.json` | 各动作回报、成功数与失败原因 |

## 策略评估

加载在线 checkpoint，从每段动作首帧进行真实物理仿真，关闭域随机化和观测噪声：

```bash
taskset -c 0-7,10-31 env MPLCONFIGDIR=/tmp/supermimic-mpl XDG_CACHE_HOME=/tmp/supermimic-cache \
  .venv/bin/python scripts/evaluate_supermimic.py \
  --device cuda:0 --checkpoint outputs/supermimic/online_training/best.pt \
  --output outputs/supermimic/online_policy_evaluation.json
```

默认评估全部训练动作，直到动作完成或物理失败。归一化回报按完整动作的最大可达
回报计算，提前失败会降低得分。`--max-steps-per-clip` 仅用于调试，报告会标记未完整
评估的动作。`best.pt` 可能仍是初始化策略，应以评估报告判断训练是否改善。

## 测试

```bash
taskset -c 0-7,10-31 env MPLCONFIGDIR=/tmp/supermimic-mpl XDG_CACHE_HOME=/tmp/supermimic-cache \
  SUPERMIMIC_TEST_DEVICE=cuda:0 \
  .venv/bin/python -m unittest discover -s tests -v
```

测试覆盖数据转换、动作库、Replay 边界、真实仿真、可微运动学、BPTT、损失归约、
在线更新、断点恢复与 checkpoint 推理。无 GPU 时可改为 `SUPERMIMIC_TEST_DEVICE=cpu`；
仅适用于 CUDA 的测试会跳过。

## 详细文档

- [LAFAN 数据转换](doc/lafan_conversion_zh.md)
- [动作库、采集环境与回放池](doc/supermimic_foundation_zh.md)
- [独立世界模型训练与验证](doc/supermimic_world_model_zh.md)
- [在线训练与评估协议](doc/supermimic_online_zh.md)

独立采集与世界模型训练入口用于诊断和离线实验，在线训练可直接从头运行。
