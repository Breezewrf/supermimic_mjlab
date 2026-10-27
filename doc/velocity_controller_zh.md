# 93 维速度控制接口

速度控制器接收 `(vx, vy, wz)`，使用最近 10 帧本体观测输出 29 维归一化关节位置动作。
`vx/vy` 使用机器人 yaw-only heading 坐标系，`wz` 使用世界坐标系角速度的 z 分量。
推理不需要 motion library、motion anchor、参考朝向或步态 phase。

## 网络和任务定义

每帧观测顺序与原 SuperMimic 一致：相对关节位置 29、关节速度 29、root 局部角速度 3、
投影重力 3、上一实际执行动作 29，共 93 维。动作噪声在真实采集时加入，history 和 replay
记录执行后的动作；推理不添加动作噪声。

Velocity encoder 将归一化历史的 930 维展平结果与三维 command 拼接，经过
`MLP(256,256) → 32 → tanh`。Command 固定除以 `[2.0, 1.32, 3.35]`，不裁剪参考指令。
共享 decoder 的输入为 `32 + 10×93 = 962`，沿用 motion decoder 的结构。
导出仅包含 velocity encoder、decoder、观测归一化及控制元数据。

Velocity critic 使用高度、去除 yaw 的 6D root 朝向、关节状态、heading-frame 线/角速度、
command、上一动作和仿真 context，共 136 维；MLP 宽度为 `(256,128)`。

| 默认奖励 | 权重/形式 |
| --- | --- |
| 平面速度 | `2 * exp(-sum(error_xy²) / 0.5²)` |
| Yaw rate | `2 * exp(-error_wz² / 0.5)` |
| 动作平滑 | `-0.3 * sum((action - previous_action)²)` |
| 关节限位 | `-2 * sum(limit_excess)` |
| Torso 侧倾 | `-4 * sin(sideways_lean)²`，允许前倾 |
| Torso 横向角速度 | `-0.05 * sum(omega_heading_xy²)` |
| 零指令姿态约束 | 可选，`--stand-still-weight` 默认 `0`，启用时设负值 |

每项只乘一次 `dt=0.02`。站立判断分别使用 `0.1 m/s` 和 `0.1 rad/s` 的阈值。
仿真与 imagination 共用同一套可微奖励函数。

跌倒条件为 root 高度 `<0.35 m` 或 root 倾斜 `>70°`。Imagination 以相同物理量计算
sigmoid 概率，高度宽度为 `0.02 m`，重力投影宽度为 `0.04`。时间上限属于 truncation，
不作为跌倒训练标签。默认 episode 为 10 秒。

设计参考 [SuperMimic 附录 B/G、表 A.6](supermimic.pdf) 与
[Unitree velocity 实现（1425b15）](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity)。
70° 阈值是参考 Unitree 的工程选择，论文阈值约为 73°。
Unitree 的固定 gait、contact、air-time、soft-landing 奖励需要额外接触信息，当前不加入训练目标；
没有将仿真接触值或 reference anchor 作为速度 Actor 的隐藏输入。

## 初始化及分阶段训练

默认初始化文件：

- `outputs/policy_iteration_010000.pt`：已有 93 维 motion 策略及输入归一化。
- `checkpoints/best/world_model_best.pt`：共享动力学和对应归一化。

两个文件可分别指定。脚本验证机器人签名、关节/物理配置、motion manifest 以及 93 维输入；
99 维最佳 Actor 不会被自动裁剪。如果本机没有旧版 93 维权重，先使用原 motion 训练入口生成
关闭 `relative_orientation` 和 `motion_anchor_ori_b` 的策略，或使用 `--from-scratch --stage motion`。
随机 decoder 需要先进行 motion 训练，不能将一次从零开始的短跑视为已有 motion prior。

```bash
python scripts/train_velocity.py \
  --actor-checkpoint outputs/policy_iteration_010000.pt \
  --world-checkpoint checkpoints/best/world_model_best.pt \
  --stage joint --warmup-updates 500 \
  --device cuda:0 --num-envs 4096 \
  --iterations 100000 --output outputs/velocity/run
```

`--stage joint` 先进行默认 500 次 latent 对齐预热，之后自动进入联合更新。
`--warmup-updates 0` 跳过预热。`--stage warmup` 整个 run 只训练速度 encoder 的 latent 对齐，
并持续更新世界模型；`--stage motion` 只优化 motion Actor/Critic 和世界模型，
仍使用混合环境进行采集。需要适配 motion prior 时，可将该阶段输出的
`motion_policy_last.pt` 和 `world_model_last.pt` 作为下一次 joint run 的初始化。

Motion/velocity 各占一半环境和 imagination batch；`--num-envs` 必须为偶数，
`--policy-batch-size` 和 `--wm-batch-size` 表示两类任务的总 batch。
真实物理仿真只创建一个实例，两个任务使用独立 replay。

| 阶段/更新 | 可训练参数 |
| --- | --- |
| Velocity 预热 | Velocity encoder；motion encoder、decoder 和共享观测统计固定 |
| Joint motion return | Motion encoder、decoder |
| Joint velocity return + `0.1 * latent MSE` | Velocity encoder |
| Critic | 两个任务各自的 critic |
| World model | 混合两类转移更新动力学；motion termination head 只使用 motion 标签 |

冻结参数不会关闭计算图。速度回报仍通过 decoder、世界模型和末端 critic 的输入 Jacobian
反传到 encoder。归一化只从新采集的真实数据更新；imagination 不更新统计量。

默认只选 `walk*`、`run*`、`sprint*` clip，也可显式传 `--clips`。
训练开始前默认对所选 clip 各评估 250 步，写入 `initial_motion_evaluation.json`。
这是有步数上限的检查，不代表完整动作成功率；`--initial-evaluation-steps 0` 可关闭。

## 指令、数据边界与恢复

每次 reset 或每 250 控制步重新选择指令来源：50% 从参考动作读取，50% 随机采样；
随机分支以 10% 概率置零。预热阶段使用参考来源。
参考线速度先按每帧 heading 转换，再进行中心 25 帧平均，clip 边缘复制端点，禁止跨 clip 平滑。

初始随机范围为 `vx[-0.5,1.0]`、`vy[-0.5,0.5]`、`wz[-1,1]`；累计 120000 个采集控制步后
扩展至 `vx[-1.5,2]`、`vy[-1.32,1.32]`、`wz[-3.35,3.35]`。
此计数为每环境共享的控制步计数，包含预热，不乘并行环境数。
速度训练 reset 以各 50% 概率选择站姿和 locomotion 参考姿态。

`a_t` 与转移奖励使用 `c_t`；下一步才采用 `c_(t+1)`。
Command 切换不会切断历史或改变物理速度。参考指令片段结束只切换片段，不重置机器人。
Replay 保存 reset 前的最终状态和当前/下一 command，并通过 episode ID 隔离历史。
Imagination 使用同一个指令调度器重建未来命令，随机来源不计算 latent 对齐损失。

```bash
python scripts/train_velocity.py \
  --resume outputs/velocity/run/last.pt \
  --device cuda:0 --iterations 120000 --output outputs/velocity/run
```

`--iterations` 是包含已完成轮数的总量，控制余弦学习率衰减。
新训练要求输出目录不存在，避免覆盖已有配置、日志和权重。续训必须使用 checkpoint
所在目录；省略 `--output` 时自动选用该目录，新训练的默认目录仍为 `outputs/velocity/run`。
Resume 从 checkpoint 恢复模型、优化器、两个 replay、归一化、阶段、指令来源/游标/计时、
curriculum 计数与 RNG；网络/训练参数以保存配置为准。
物理仿真重新开始 episode，使用新的 episode ID，避免把恢复后的新历史拼接到旧 replay。
保存文件不是逐 substep 的 MuJoCo 物理快照，不承诺续训物理轨迹逐位一致。

终端按 Reward、Error、Loss、Objective、Failure 和 Performance 分区显示，`--log-interval 10` 控制区间统计输出频率；
`--log-format json` 可切回逐条 JSON 输出。完整逐轮指标始终保存至 `metrics.jsonl`。

两个训练入口共用 `rl/training_runner.py`，统一处理训练循环、日志间隔、保存调度、
JSONL 恢复、W&B run 生命周期、异常报告和仿真资源清理。
任务各自提供评估、权重导出与指标映射。

在训练命令后加上以下参数即可启用 W&B（与 motion 入口相同）：

```bash
--wandb --wandb-project supermimic --wandb-name velocity_joint
```

使用已有 W&B 登录凭据，`--wandb-entity` 可指定团队；没有网络时加 `--wandb-mode offline`，
记录会保存在输出目录的 `wandb/` 下。可选依赖通过 `uv sync --extra logging` 安装。
默认不启用 W&B，也不上传模型权重。

记录包括 motion/velocity 奖励和误差、跌倒率、两类策略与 critic 指标、latent 对齐、
世界模型两类损失、梯度、学习率、阶段和采集控制步。
W&B 新增按含义命名的分组（原字段作为兼容别名保留）：

| 分组 | 示例 | 含义 |
| --- | --- | --- |
| `reward/` | `reward/velocity/linear_tracking`、`reward/velocity/total_per_step` | 已乘权重与 dt 的每控制步奖励，越大越好；惩罚项为负数 |
| `error/` | `error/velocity/planar_velocity_m_per_s`、`error/velocity/yaw_rate_rad_per_s` | 真实物理误差，保留 m/s、rad/s 等单位，越小越好 |
| `loss/` | `loss/world_model/total`、`loss/velocity/latent_alignment_mse`、`loss/critic/velocity` | 模型、latent 对齐及 critic 拟合损失 |
| `objective/` | `objective/velocity/return` | 想象 rollout 的折扣回报与末端价值，不等于实际每步奖励 |
| `failure/` | `failure/velocity/events_per_transition` | 失败事件占转移数的比例，终端转换为百分比 |
| `gradient/`、`diagnostic/` | 梯度范数、动作变化、想象存活率 | 训练诊断 |

JSONL 保留原始字段，并增加 `metric_schema_version=1` 与分类后的 `metric_groups`。
旧 JSONL 也可直接由新 W&B reporter 分类，不需要重写日志。
区间内的实际采集指标按新采集转移数加权，未采集轮次不会重复计入。
真实采集指标只在产生新数据时上报；最终的 1/8/32 步回放诊断单独归入
`velocity_replay_prediction/`，不表示 held-out 评估。

在线模式在同一输出目录续训时沿用 `wandb_run.json` 中的 run；回退旧 checkpoint 或离线续训
会创建带 `parent_run_id` 的新 run 段。续训命令也需要传 `--wandb`，并保留原输出目录和
`wandb_run.json`。程序正常结束或异常退出都会关闭 W&B run。


输出包含 `last.pt`、`velocity_policy_last.pt`、`motion_policy_last.pt`、`world_model_last.pt`、
`metrics.jsonl` 和 `velocity_diagnostics.json`。
最后一个文件报告 velocity replay 上 logged-action 的 1/8/32 步预测误差，属于回放内诊断，
不是 held-out 泛化分数。原 motion-only 脚本和旧 checkpoint 格式继续可用。

## 独立播放和阶跃评估

```bash
python scripts/play_velocity.py \
  --checkpoint outputs/velocity/run/velocity_policy_last.pt --device cuda:0
```

窗口按键：`W/S` 增减前向速度，`A/D` 增减侧向速度，`Q/E` 增减 yaw rate，空格清零。
默认启用鼠标外力：双击机器人身体选中，再按住 `Ctrl` 拖拽，对真实物理仿真施加扰动。
`--no-perturbations` 可关闭；松开拖拽、跌倒重置及退出时会清除外力，headless 模式不启用。
`--command 0.5 0 0.2` 设置初始指令。有窗口播放默认持续运行直到关闭窗口；
`--steps N` 可显式限制播放步数。`--headless` 使用相同策略与物理环境执行无窗口检查，
默认运行 1000 步，也可用 `--steps` 调整。
播放默认关闭跌倒 termination，拖倒后继续执行；`--termination` 可启用跌倒后自动回到站姿。

```bash
python scripts/evaluate_velocity.py \
  --checkpoint outputs/velocity/run/velocity_policy_last.pt \
  --device cuda:0 --num-envs 64 --trials 4 \
  --output outputs/velocity/run/evaluation.json
```

默认包含站立、前后移动、侧移、原地转向和组合指令共 9 种条件。
每次从站立开始，运行 5 秒，以最后 2 秒的平均速度误差计算稳态误差。
报告跌倒率、成功试验的相对稳态误差中位数、平移/转动误差、位移和动作变化。
零指令不计算相对误差；跌倒试验保留记录，其稳态分数为 `null`。
四次试验采用相同确定性初始条件，因此是重复性检查，不是四个随机扰动鲁棒性试验。

`--full-grid` 运行三个命令平面的 `3×(15×15−1)=672` 个非零条件，另外测试站立。
每个轴将负区间和正区间分别划分为 7 段以精确包含零；前向范围不对称时不是全轴等距采样。
共享轴上的点保留为不同平面的测试。默认不对显示或 JSON 的误差做 0.9 截断。
`--duration` 和 `--steady-seconds` 只用于调试缩短评估。

## 短程验证

以下配置使用完整预训练网络，但缩小采集和更新规模：

```bash
python scripts/train_velocity.py \
  --device cuda:0 --num-envs 64 --capacity 100 \
  --warmup-steps 32 --collect-steps 5 --updates-per-collect 1 \
  --wm-batches 1 --wm-batch-size 64 --policy-batch-size 64 --horizon 32 \
  --critic-epochs 1 --critic-minibatches 1 \
  --warmup-updates 5 --iterations 25 \
  --clips walk1_subject1 run1_subject2 --initial-evaluation-steps 50 \
  --checkpoint-interval 25 --log-interval 1 --output outputs/velocity/smoke
```

新增测试：

```bash
python -m unittest tests.test_velocity tests.test_velocity_training -v
```

测试覆盖坐标不变性、奖励尺度、硬/软终止、指令切换和 clip 边界、梯度隔离、
末端 bootstrap、eager/编译接口一致性、真实物理奖励一致性、timeout/reset、两任务更新、
checkpoint 恢复以及不读取 manifest 的独立推理。

本次短跑已完成 5 次预热和 20 次联合更新，并完成续训检查。
9 类指令、每类 4 次的 5 秒测试未跌倒，但非零指令的相对误差约为 1，策略主要保持站立；
这验证了训练和推理链路，尚未得到收敛的速度跟踪策略。完整训练与 672 条件评估尚未运行。
