# SuperMimic Mjlab

[English](README.md) | 简体中文

<p align="center">
  <img src="doc/assets/demo_v0.gif" alt="SuperMimic demo">
</p>

面向 Unitree G1 多动作跟踪的 [SuperMimic](doc/supermimic.pdf) 复现。当前策略采集
真实仿真转移，世界模型用 8 步自回归窗口学习动力学，再通过 32 步可微想象训练
Actor 和 Critic。

当前数据集包含 40 段 G1 重定向 LAFAN 动作，约 2.45 小时，控制频率为 50 Hz。

机器人使用本地复制的 [RoboJuDo-Plus G1 资源](src/assets/robots/g1/README.md)，
加载 `g1_29dof_rev_1_0.xml`，碰撞体包含网格、肩部圆柱体及每脚 4 个半径 5 mm 的
球体。训练沿用 mjlab 的 PD 控制配置，关闭自碰撞。

## 运行环境

在仓库根目录安装并激活环境：

```bash
uv sync
source .venv/bin/activate
```

依赖和 GPU 检查见[安装说明](doc/setup_zh.md)。

## 代码结构

按职责划分配置、数据、任务函数、模型与训练代码：

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
│   ├── training_logger.py    # 终端表格和 W&B 日志
│   ├── checkpoints.py        # 完整断点与轻量策略导出
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
├── play_supermimic.py             # MuJoCo 实时策略／参考对照
├── export_supermimic_policy.py    # 从完整断点提取策略或世界模型
├── data/
│   └── convert_lafan_to_npz.py     # LAFAN CSV 转 NPZ
└── offline/
    ├── collect_supermimic.py      # 采集快照或运动学回放校验
    └── train_world_model.py       # 独立世界模型训练与验证
```

日常在线训练、评估和播放使用根目录脚本。`offline/` 用于诊断和离线实验，
在线训练不需要先运行这些脚本。

## 准备动作数据

原始 CSV 位于 `src/assets/motions/LAFAN1_Retargeting_Dataset/`。尚未转换时运行：

```bash
python scripts/data/convert_lafan_to_npz.py \
  --input-dir src/assets/motions/LAFAN1_Retargeting_Dataset \
  --output-dir src/assets/motions/g1/lafan1_npz \
  --input-fps 30 --output-fps 50
```

输出为每段动作的 NPZ 和 `manifest.json`，已有转换结果时可跳过。
字段和坐标约定见 [LAFAN 转换说明](doc/lafan_conversion_zh.md)。

## 在线训练

### 复现当前预览模型（第 10000 轮）

当前预览使用以下配套权重，均从
`outputs/supermimic/lafan_2680_fp32/last.pt` 的**第 10000 轮**导出：

| 文件 | 内容 |
| --- | --- |
| `checkpoints/preview/policy_10000.pt` | 真实物理播放使用的 Actor 权重，参考 schema 为 `full_body_268` |
| `checkpoints/preview/world_model_10000.pt` | 同轮世界模型，用于叠加预测姿态 |

**模型仍在训练中，目前提供的是第 10000 轮的 preview 版本。** 全部 40 段 LAFAN
动作的评估记录：

| 指标 | 第 10000 轮 |
| --- | --- |
| 归一化回报 | 0.498 |
| 成功动作 | 35/40（87.5%） |
| 失败动作 | 5/40 |
| 平均完成率 | 94.0% |
| 并行评估环境 | 40 |
| 评估耗时（含初始化） | 125.5 秒 |

**下述 4096 环境、FP32 配置的训练显存占用约 30 GB。** 此预览权重未启用
Actor／Critic RunningNorm。`checkpoints/preview/` 中的两个文件均为推理导出文件，
续训需要完整的 `last.pt`。

在仓库根目录运行以下训练配置：

```bash
python scripts/train_supermimic.py \
  --device cuda:0 \
  --reference-dim 2680 \
  --world-prediction velocity_residual \
  --amp none --no-compile \
  --no-actor-input-normalization --no-critic-input-normalization \
  --seed 0 \
  --num-envs 4096 --capacity 1000 \
  --warmup-steps 32 --wm-only-iters 0 \
  --collect-steps 5 --updates-per-collect 2 \
  --wm-batches 10 --wm-batch-size 2048 \
  --policy-batch-size 4096 --horizon 32 \
  --critic-epochs 5 --critic-minibatches 4 \
  --learning-rate 2e-4 \
  --world-dims 2048 2048 2048 \
  --actor-dims 2048 1024 512 256 \
  --critic-dims 256 128 \
  --iterations 32000 \
  --eval-interval 2000 --checkpoint-interval 2000 \
  --output outputs/supermimic/lafan_2680_fp32
```

若只复现预览的训练长度，将总轮数改为 `--iterations 10000`。从同一个第 10000 轮
完整 checkpoint 导出两份权重；继续训练时 `last.pt` 会更新，不能将最终 32000 轮的
权重误标为 10000 轮：

```bash
python scripts/export_supermimic_policy.py \
  --checkpoint outputs/supermimic/lafan_2680_fp32/last.pt \
  --output checkpoints/preview/policy_10000.pt

python scripts/export_supermimic_policy.py \
  --component world-model \
  --checkpoint outputs/supermimic/lafan_2680_fp32/last.pt \
  --output checkpoints/preview/world_model_10000.pt
```

在真实物理中播放指定动作：

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/preview/policy_10000.pt \
  --clips walk1_subject1 run1_subject2 --loop
```

同时预览世界模型内策略在未来 32 个控制步（0.64 秒）后的预测：

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/preview/policy_10000.pt \
  --world-model checkpoints/preview/world_model_10000.pt \
  --world-model-mode future --world-model-horizon 32 \
  --clips walk1_subject1 run1_subject2 --loop
```

不指定 `--clips` 时播放全部 40 段动作。播放默认允许鼠标外力，并在超过高度／倾斜
阈值后继续执行；加 `--no-perturbations --termination` 可关闭交互外力并恢复失败重置。
全部动作的标准无窗口评估命令：

```bash
python scripts/evaluate_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/preview/policy_10000.pt \
  --output outputs/supermimic/evaluation_policy_10000.json
```

训练默认启用域随机化、观测噪声和按回报采样的重置；不指定 `--clips` 时使用全部
动作，调试可指定 `--clips walk1_subject1 run1_subject2`。

机器人关闭自碰撞，保留全身触地。训练重置扰动参考位姿、根速度和关节位置；摩擦与
躯干质心在启动时采样，编码器偏差逐 episode 重采样。评估关闭重置噪声和域随机化。
范围见[环境配置](doc/supermimic_foundation_zh.md)。训练变体、可选 RunningNorm、
checkpoint 格式和查看器操作见[操作指南](doc/training_and_playback_zh.md)。

### 断点恢复

网络和 batch 设置从完整 checkpoint 恢复；`--iterations` 为包含已完成轮数的总轮数：

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --resume outputs/supermimic/lafan_2680_fp32/last.pt \
  --iterations 32000
```

## 训练日志与 W&B

终端按输出区间统计平均值；`Fail%` 为每条采集转移对应的物理失败事件频率，
`Steps/s` 包含采集和优化时间，`GPU` 为 PyTorch 分配器记录的显存峰值（GiB）。
仿真初始化、kernel 和 seed 信息写入 `simulator.log`，异常回溯保留在 stderr。

`--log-interval 50` 可降低输出频率，`--verbose-sim` 显示仿真详情，
`--log-format json` 切换 JSON 输出。`metrics.jsonl` 始终逐轮保存原始指标；
安装 `rich` 时使用格式化表格，否则使用普通文本。

安装可选依赖并登录一次：

```bash
uv sync --extra logging
wandb login
```

为新训练启用 W&B：

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --iterations 1000 \
  --wandb --wandb-project supermimic --wandb-name g1-lafan \
  --output outputs/supermimic/online_wandb
```

续训时再次传入 `--wandb`。离线日志和 run 延续规则见
[操作指南](doc/training_and_playback_zh.md#训练日志与-wb)。

## 详细文档

- [安装说明](doc/setup_zh.md)
- [训练变体、checkpoint 格式、评估、查看器操作与测试](doc/training_and_playback_zh.md)（[English](doc/training_and_playback_en.md)）
- [LAFAN 数据转换](doc/lafan_conversion_zh.md)
- [动作库、采集环境与 Replay Buffer](doc/supermimic_foundation_zh.md)
- [独立世界模型训练与验证](doc/supermimic_world_model_zh.md)
- [在线训练与评估协议](doc/supermimic_online_zh.md)

独立采集和模型训练入口用于诊断及离线实验，在线训练可以直接从头开始。
