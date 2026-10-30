# SuperMimic Mjlab

[English](README.md) | 简体中文

## 内容目录

- [运行环境](#运行环境)
- [代码结构](#代码结构)
- [准备动作数据](#准备动作数据)
- [在线训练](#在线训练)
  - [消融结果与发布模型](#消融结果与发布模型)
  - [断点恢复](#断点恢复)
- [Play OOD 动作](#play-ood-动作)
- [Sim2Real](#sim2real)
- [训练日志与 W&B](#训练日志与-wb)
- [详细文档](#详细文档)

<p align="center">
  <img src="doc/assets/demo_v0.gif" alt="SuperMimic demo">
</p>

面向 Unitree G1 多动作跟踪的 [SuperMimic](doc/supermimic.pdf) 复现。当前策略采集
真实仿真转移，世界模型用 8 步自回归窗口学习动力学，再通过 32 步可微想象训练
Actor 和 Critic。

当前数据集包含 40 段 G1 重定向 LAFAN 动作，约 2.45 小时，控制频率为 50 Hz。

新增 93 维观测的速度控制接口，支持速度 encoder 预热、motion/velocity 联合训练以及
不依赖参考动作的独立播放。训练方法与短跑验证结果见[速度控制说明](doc/velocity_controller_zh.md)。

完整规模训练使用单张 **NVIDIA RTX A6000（48 GB 显存）**，在 4096 环境、FP32
配置下训练 **15000 轮约需 4 小时**。显存较小的 GPU 可通过减少环境数和 batch size
缩小训练规模，具体配置见[训练操作指南](doc/training_and_playback_zh.md#缩小训练规模)，性能未经验证。

<p align="center">
  <img src="doc/assets/evaluation.png" alt="Evaluation result">
</p>

最佳模型在全部 40 段 LAFAN 动作的固定起点、完整动作标准评估中取得 **40/40
成功动作（100%）**、**100% 平均完成率**和 **0.8528 归一化回报**。
配置对比与权重使用方法见[消融结果与发布模型](#消融结果与发布模型)。

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

G1 29DoF 和 23DoF 使用相同流程，通过 `--robot` 选择机器人（默认 `g1_29dof`），
并准备与之匹配的 retarget 动作数据。下面以
`src/assets/motions/LAFAN1_Retargeting_Dataset/` 中的 CSV 为例；其他 retarget 数据集
用 `--input-dir` 指定：

```bash
python scripts/data/convert_lafan_to_npz.py --robot g1_29dof \
  --input-dir src/assets/motions/LAFAN1_Retargeting_Dataset \
  --output-dir src/assets/motions/g1/lafan1_npz \
  --input-fps 30 --output-fps 50
```

输出为每段动作的 NPZ 和 `manifest.json`，已有转换结果时可跳过。
字段和坐标约定见 [LAFAN 转换说明](doc/lafan_conversion_zh.md)。

23DoF 使用 `--robot g1_23dof`，并将 `--output-dir` 改为
`src/assets/motions/g1/lafan1_npz_23dof`，两个机器人的转换数据分开存放。
训练会选择对应的默认 manifest；评估／播放通过 `--motion` 或 `--motion-dir`
指定匹配的 NPZ 数据。

## 在线训练

两种机器人使用相同训练命令，通过 `--robot g1_29dof` 或 `--robot g1_23dof` 选择。
默认使用紧凑参考；全身参考在 29DoF 下使用 `--reference-dim 2680`，23DoF 下使用
`--reference-dim 2560`。恢复、评估和播放从 checkpoint 自动读取机器人类型，权重和
动作数据需与机器人匹配。下文发布的权重及消融结果均为 29DoF。速度训练和离线训练
也使用相同的 `--robot` 选择方式。

`motion_anchor_ori_b` 是可选消融变量，默认关闭，可在训练命令中添加
`--motion-anchor-ori-b` 开启。

### 消融结果与发布模型

消融实验统一使用 `full_body_268` 参考特征、FP32 网络、4096 个环境，并在全部 40
段 LAFAN 动作上进行完整确定性评估。下表是各配置已完成重跑在第 15000 轮的均值：

| 配置 | 成功动作 | 归一化回报 |
| --- | ---: | ---: |
| Actor／Critic 输入归一化 | 38-40/40 | 0.6016 |
| + torso 坐标系动作朝向 anchor | 38–40/40 | 0.8130 |
| + 世界模型归一化 | 40/40 | 0.8504 |
| + 修正后的 15000 轮余弦学习率 | **40/40** | **0.8526** |

朝向 anchor 带来了最大的策略性能提升，世界模型归一化则是保证其稳定训练的关键；
收益主要集中在 walk 动作，部分 get-up、jump 和 run 动作仍有取舍。

| 文件 | 内容 |
| --- | --- |
| `checkpoints/best/policy_best.pt` | 第 15000 轮最佳 Actor 权重，用于真实物理播放 |
| `checkpoints/best/world_model_best.pt` | 同轮配套世界模型，用于叠加预测姿态 |

选中策略在完整标准评估中取得 **40/40 成功动作**、**1.0 平均完成率**和
**0.8528 归一化回报**。

训练硬件为单张 **NVIDIA RTX A6000（48 GB 显存）**。使用下述 4096 环境、FP32
配置，完成 **15000 轮训练约需 4 小时**。

在仓库根目录运行以下配置可复现所选训练设置：

```bash
python scripts/train_supermimic.py \
  --robot g1_29dof \
  --device cuda:0 \
  --reference-dim 2680 \
  --world-prediction state_delta \
  --actor-input-normalization --critic-input-normalization \
  --world-model-normalization --motion-anchor-ori-b \
  --amp none --no-compile \
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
  --iterations 15000 \
  --eval-interval 1000 --checkpoint-interval 1000 \
  --output outputs/supermimic/run
```

训练器按照成功动作数优先、归一化回报次优的规则，配套输出 `policy_best.pt` 和
`world_model_best.pt`。

在真实物理中播放指定动作：

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/best/policy_best.pt \
  --clips walk1_subject1 run1_subject2 --loop \
  --motion-dir src/assets/motions/g1/lafan1_npz
```

同时预览世界模型内策略在未来 32 个控制步（0.64 秒）后的预测：

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/best/policy_best.pt \
  --world-model checkpoints/best/world_model_best.pt \
  --world-model-mode future --world-model-horizon 32 \
  --clips walk1_subject1 run1_subject2 --loop \
  --motion-dir src/assets/motions/g1/lafan1_npz
```

不指定 `--clips` 时播放全部 40 段动作。播放默认允许鼠标外力，并在超过高度／倾斜
阈值后继续执行；加 `--no-perturbations --termination` 可关闭交互外力并恢复失败重置。
全部动作的标准无窗口评估命令：

```bash
python scripts/evaluate_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/best/policy_best.pt \
  --output outputs/supermimic/evaluation_best.json \
  --motion-dir src/assets/motions/g1/lafan1_npz
```

训练默认启用域随机化、观测噪声和按回报采样的重置；不指定 `--clips` 时使用全部
动作，调试可指定 `--clips walk1_subject1 run1_subject2`。

评估／播放必须显式选择一个动作来源：`--motion /path/to/clip.npz`、
`--motion-dir /path/to/motions` 或 `--manifest /path/to/manifest.json`，三者互斥。
文件和目录模式无需 manifest；目录按相对路径排序并递归加载 NPZ。不指定 `--clips`
时使用该来源的全部动作，包括训练中未出现的动作。
模型和数据 SHA 仅作为可选溯源信息，不再阻止加载；仍检查网络维度、机器人关节／刚体
顺序及拓扑、动作格式和仿真配置。在线续训为恢复 replay 和采样器状态，仍保留原 clip 顺序。


机器人关闭自碰撞，保留全身触地。训练重置扰动参考位姿、根速度和关节位置；摩擦与
躯干质心在启动时采样，编码器偏差逐 episode 重采样。评估关闭重置噪声和域随机化。
范围见[环境配置](doc/supermimic_foundation_zh.md)。训练变体、可选 RunningNorm、
checkpoint 格式和查看器操作见[操作指南](doc/training_and_playback_zh.md)。

### 断点恢复

网络和 batch 设置从完整 checkpoint 恢复；`--iterations` 为包含已完成轮数的总轮数：

```bash
python scripts/train_supermimic.py \
  --robot g1_29dof \
  --device cuda:0 --resume outputs/supermimic/run/last.pt \
  --iterations 32000
```

## Play OOD 动作

<p align="center">
  <img src="src/assets/motions/examples/C5_-_walk_to_run_stageii.gif" alt="分布外动作播放：行走转跑步" width="48%" />
  <img src="src/assets/motions/examples/dance_video.gif" alt="分布外动作播放：视频舞蹈" width="48%" />
</p>

<p align="center">左：ACCAD 数据集中的动作示例。右：从视频转换为动作数据的示例。</p>

也可以使用 [video2motion](https://github.com/Breezewrf/video2motion)，从视频、Kimodo、
GMR 动作或 ACCAD 数据集导出 SuperMimic 可用的动作数据。转换方法参考该仓库，
导出的 NPZ 可直接搭配对应机器人的 checkpoint 播放：

```bash
python scripts/play_supermimic.py \
  --checkpoint checkpoints/best/policy_best.pt \
  --motion src/assets/motions/examples/C5_-_walk_to_run_stageii.npz \
  --loop
```

## Sim2Real

SuperMimic 已完成 sim2real 迁移，并在
[RoboJuDo-Plus 的 `supermimic` 分支](https://github.com/chifongip/RoboJuDo-Plus/tree/supermimic)
中完成部署。动作跟踪、速度控制、loco/mimic 切换、权重与动作配置及真机操作步骤见
[SuperMimic 部署指南](https://github.com/chifongip/RoboJuDo-Plus/blob/supermimic/docs/supermimic.md)。
该指南包含 23DoF 真机配置，支持 SuperMimic 速度控制与动作跟踪切换，以及
Maniloco 与 SuperMimic 切换。

**安全注意事项：**

1. 不同真机的实际状态和表现存在差异，我们无法保证该策略一定能在你的设备上稳定运行。
2. 我们的真机部署使用 **G1 23DoF，而非 29DoF**。
3. 初次 sim2real 建议使用幅度较小的 mimic 动作，配合稳定的 locomotion 策略。
   **当前 SuperMimic velocity tracking 仍不够稳定。**
4. RoboJuDo-Plus 支持通过 Unitree 手柄方便地切换模式。遇到紧急情况时，可按 **Y**
   让机器人恢复默认姿态，按 **B** 进入阻尼模式，或按 **A** 进入零力矩（被动）模式。
   请熟悉这些按键，以帮助保护机器人和测试人员的安全。

## 训练日志与 W&B

终端按 Reward、Error、Loss、Objective、Failure 和 Performance 分区，误差显示物理单位。
JSONL/W&B 提供 `reward/`、`error/`、`loss/`、`objective/` 分类，并保留旧字段兼容。
日志按输出区间统计平均值；失败百分比为每条采集转移对应的物理失败事件频率，
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
  --robot g1_29dof \
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
