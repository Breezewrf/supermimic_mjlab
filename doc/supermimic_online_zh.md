# SuperMimic 在线训练：采集、世界模型与 Actor–Critic

在线入口实现论文 §3.1、§3.3 和 Algorithm A.1 的三阶段循环：当前策略采集真实
仿真转移，8 步自回归拟合世界模型，再通过固定参数的世界模型进行 32 步想象，更新
Actor 和 Critic。默认从头初始化三个网络，不加载独立预训练世界模型，不以离线
32 步验证覆盖率作为前置门槛。

## 运行

先按[安装说明](setup_zh.md)使用 uv 安装并激活环境。以下命令在仓库根目录运行，
使用缩小 batch 配置：

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --iterations 1000 \
  --output outputs/supermimic/online_training
```

不指定 `--clips` 时使用 manifest 中全部 40 段 LAFAN 动作。输出目录必须不存在。

默认配置：

| 参数 | 默认值 |
| --- | --- |
| 物理环境 / 回放容量 | 64 个环境，每环境 1000 步 |
| 初始采集 | 当前随机初始化 Actor 采集 32 个向量步，共 2048 条转移 |
| 世界模型预热 | `--wm-only-iters 0`，默认不跳过策略更新 |
| 数据与更新比例 | 每采集 5 个向量步，执行 2 轮训练 |
| 每轮世界模型拟合 | 10 个 batch，每个 batch 256 条 8 步窗口 |
| 每轮策略更新 | 64 条 32 步想象轨迹，更新 Actor 一次 |
| 每轮 Critic 更新 | 同一批固定 GAE targets，5 个 epoch，每个 epoch 4 个 minibatch |
| 学习率 / 折扣 | 三个网络均为 `2e-4`；γ=0.99、λ=0.95 |
| 学习率日程 | 250,000 轮内余弦下降至 0.03 倍，之后保持 |
| 保存 / 评估 | 每 100 轮，结束时也保存、评估 |

这里的 iteration 指一轮世界模型拟合；联合训练阶段还包括一次 Actor 更新及相应
Critic 拟合。初始采集不更新网络。世界模型归一化由初始真实转移拟合并冻结，
后续每轮继续更新网络参数。
这些 batch 和环境数量是面向当前硬件的缩放配置，不是论文完整训练规模。
仿真使用每环境接触分配 128、约束容量 1024；每个物理子步检查碰撞、接触和约束
计数，超限立即报错，避免继续保存不完整物理计算产生的转移。
机器人关闭自碰撞，保留全身与地面的接触。训练重置加入参考实现的位姿、速度和
关节位置扰动；摩擦和 torso CoM 在启动时采样，跨 episode 固定，编码器偏差每次
重置采样。具体范围见[仿真与初态配置](supermimic_foundation_zh.md)。评估关闭初态
扰动、传感器噪声和域随机化。

需要先预热世界模型时，在训练命令中增加 `--wm-only-iters 500`，可同时将
`--warmup-steps` 设为 `64`。初始采集后，前 500 轮继续按原频率采集真实转移，
每轮拟合世界模型，但跳过 Actor 和 Critic 更新；第 501 轮开始联合训练。
`--wm-batches 10` 时预热共执行 5000 次世界模型优化。预热使用与后续在线训练
相同的模型、损失、Replay 和优化器，不需要独立的离线模型 checkpoint。

`--wm-only-iters` 默认为 `0`，必须为非负整数；`--warmup-steps` 仅控制最初的
采集步数。预热轮数计入 `--iterations` 和学习率调度，设定的总轮数不必大于预热
轮数。配置和已完成的总迭代数随 checkpoint 保存，续训时自动完成剩余预热或直接
继续联合训练，不重复预热；缺少该字段的旧断点按 `0` 恢复，续训须保持原配置。
终端提示预热及联合训练的阶段切换，预热期间 Actor J 和 V loss 显示 `--`，JSON
和 W&B 不记录未发生的策略更新指标。跨阶段日志窗口的策略均值只统计实际更新。

快速检查可用较小的网络和单段动作：

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --iterations 5 --num-envs 4 --warmup-steps 16 \
  --world-dims 64 64 --actor-dims 64 64 --critic-dims 64 32 \
  --wm-batches 1 --wm-batch-size 16 --policy-batch-size 4 \
  --clips walk1_subject1 --evaluation-limit 100 \
  --output outputs/supermimic/online_debug
```

`--evaluation-limit` 只用于调试；报告会标记未完整评估的动作，不能据此声称完整动作
跟踪成功。`--horizon` 可用于缩短调试轨迹，正式默认值保持 32。

## 网络、参考与观测

Actor 的 motion encoder 和 decoder 隐藏层均为 `(2048, 1024, 512, 256)`、ELU。
Encoder 读取当前参考帧及其后 9 帧，输出 32 维 tanh latent。Decoder 接收 latent
与最近 10 帧、每帧 93 维本体观测，输出 29 维线性动作；仅 latent 使用 tanh。
输出层以小权重初始化，初始动作接近默认关节设定值。

每个参考帧展开为 73 维：相对窗口首帧的根位置 3 维、相对首帧 yaw 的根 6D
姿态、各帧根自身坐标系下的线速度和角速度各 3 维、原始关节角 29 维、关节速度
乘 0.1 后的 29 维。10 帧共 730 维，不再重复输入 14 个刚体的 FK 目标。
字段及量纲对齐 `supermimic_vibed` 的紧凑表示；窗口仍沿用当前帧及其后 9 帧，
本次没有改变动作库的时间索引约定。尾部沿用动作库末帧填充。

Critic 为 `(256, 128)` ELU 网络，接收 71 维特权物理状态、730 维参考特征与
33 维上下文，共 834 维，输出标量价值。物理状态同样变换到当前参考窗口锚点；
四元数统一符号。Critic 与 Actor 不共享参数，真实推理仅需要 Actor。

Actor 本体观测沿用第一阶段定义：关节角、关节速度、机体根角速度、投影重力、
上一动作，采用固定量纲缩放。参考展开、6D 姿态表示及这些缩放属于本项目明确的
实现选择，论文没有给出逐字段展开约定。

`--relative-orientation` 可在投影重力和上一动作之间增加 6D 相对根朝向
`rot6d(R_robot^T R_reference)`，观测变为 99 维；上一动作始终占最后 29 维。
采集与想象都使用对应状态时刻的参考朝向，额外六维训练噪声半宽为 0.05。
默认关闭此选项，以便与修复采样器后的 93 维基线独立对照。整体旋转机器人与参考
不改变相对朝向；仅旋转机器人则改变相对朝向，不恢复参考的全局 yaw。

CUDA 训练默认启用 `--compile --amp bf16`。编译边界包括世界模型 8 步动力学
rollout 及损失、想象单步、参考特征、批量奖励和 Critic 特征/价值、GAE 与价值损失。
每个编译边界使用 `fullgraph=True`，编译失败会报错；不会静默切回普通执行。
MLP 矩阵运算使用 BF16 autocast，权重及物理状态仍为 FP32；积分、FK、奖励、
观测与损失归约保持 FP32。CPU 自动使用普通执行及 FP32；不支持 BF16 的 GPU
可显式选择 `--amp none`，可用 `--no-compile` 单独排查编译问题。
首次训练需编译内核，耗时与稳定阶段不同；应在编译完成后比较每轮时间。

动作与状态按时间顺序生成，循环中仅计算动力学与观测。完整想象轨迹生成后，
合并 batch/time 一次计算 FK、奖励和终止；所有时刻的 Critic 特征及目标价值
批量计算并停止梯度。终点 bootstrap 单独保留对终点状态的梯度。Critic 的
时间求和损失和轨迹 minibatch 划分保持原样。

训练可通过 `--reference-dim` 选择两种参考输入，默认保持 730：

| 参数 | 每帧参考 | 10 帧编码器输入 | Critic 总输入 | checkpoint schema |
| --- | --- | --- | --- | --- |
| `--reference-dim 730` | pelvis 和关节，73 维 | 730 | 834 | `root_joint_73_v1` |
| `--reference-dim 2680` | 14 个 body 和关节，268 维 | 2680 | 2784 | `full_body_268` |

2680 模式复用原版的特征顺序、参考水平/yaw 锚点及缩放，保留绝对参考高度。
Actor、Critic、真实采集、普通及编译想象训练均使用所选特征。要对照旧版 FP32
训练，可在原命令中增加 `--reference-dim 2680 --amp none --no-compile`；
该选项只选择参考架构，不自动调整学习率、horizon 或预热配置，也不保证逐位复现旧训练。

完整训练 checkpoint 保存所选维度和 schema，`--resume` 自动恢复，无需重复指定。
缺少这些字段的历史 checkpoint 从 Actor 编码器输入宽度推断；缺少 `compile`、
`amp` 的旧断点按非编译 FP32 恢复。旧 2680 维完整训练断点可在当前代码中续训，
仍要求原有 motion manifest、机器人、物理配置、训练配置和 RNG 后端一致。
仅有 Actor 的轻量策略不支持续训。不能在续训中把 2680 改成 730 或反向切换，
不同架构需新建训练。推理、评估、可视化和导出继续自动识别两种架构。

## 三阶段更新

真实采集使用 `Actor(history, future_reference) + N(0, 0.1²)`，开启原有域随机化、
观测噪声和回报重置采样。Replay 保存实际执行动作以及 `policy_version`，后者
记录采集时已完成的 Actor 更新次数，便于核对采集确实使用了后续策略。

世界模型继续使用已有可微动力学和解析奖励、观测。在线拟合损失为归一化状态误差
加物理失败 BCE；解析奖励和观测误差只作为诊断。独立 `train_world_model.py` 的
原有四项拟合方式保留，以免改变已有离线实验的含义。
`--termination-mode analytic` 改用真实高度/倾斜阈值的解析软终止，sigmoid 温度
分别为 0.02 m 和 0.04；该模式没有终止 BCE。默认仍为 `learned`，两种模式都在
真实仿真中使用相同的硬终止条件。固定状态损失尺度和初始归一化冻结保持原样；
可通过 `--wm-only-iters` 在联合训练前增加在线世界模型预热。
每轮只在最后一个 WM batch 计算完整 FK、奖励和观测诊断，其他 batch 只计算
动力学及终止项；优化目标与梯度不变。`wm_reward`/`wm_observation` 因此代表最后
一个 batch，`wm_state`/`wm_termination` 仍取本轮全部 batch 的平均。
各项误差先按窗口对有效时间步求和，再对 batch 中的窗口取平均，对齐论文式（1）
的时间维归约；padding 不参与损失，也不作为平均分母。

每条想象轨迹从 Replay 的真实起点、噪声历史和上下文开始，之后由当前 Actor
自主生成动作。新状态经解析观测头重建观测，再添加一次传感器噪声，以可微拼接
更新历史；过去的噪声观测不会重新采样。探索动作噪声仅用于真实采集，不加到想象
中的确定性 Actor 动作。

策略目标为带软生存权重的折扣奖励和末端价值 bootstrap。更新 Actor 时冻结世界
模型和 Critic 的权重，保留它们对状态和动作的梯度；冻结不等于 no_grad 或 detach。
预测物理失败概率进入连续生存权重。真实采集仍在 10 秒时限处 reset，Replay
窗口和观测历史不会跨 reset；该采集时限不终止想象。即使起点处于真实 episode
第 499 步，也可继续完成 32 步想象并保留末端价值 bootstrap。
对齐参考实现，策略起点只从满足 `frame_id + horizon <= clip_length - 1` 的
Replay 状态中采样，保证有完整 horizon 的参考后继帧；无需 Replay 中存有这些
未来转移。每条轨迹都想象完整 horizon，末端保留 Critic 价值 bootstrap，包括
恰好到达动作最后一帧的情况。后续目标特征不足 10 帧时重复末帧，只有预测物理
失败概率影响生存权重，不因动作结束将 continuation 清零。真实采集仍在动作
末尾 reset，并隔断 Replay 窗口和历史。若 Replay 没有符合条件的起点，明确报错，
需补充有效采集数据或调整 horizon。

Actor 使用 Adam `β=(0.9, 0.95)`，梯度先按全局范数 1 裁剪，再按元素 0.1 裁剪。
世界模型与 Critic 使用 Adam，仅按全局范数 1 裁剪。Critic 的 GAE targets 基于
该轮想象轨迹计算一次并 detach，拟合 Critic 时不会更新 Actor 或世界模型。
Critic 按整条轨迹打乱、划分 minibatch；平方价值误差在每条轨迹的有效时间步上
求和，再对 minibatch 内的轨迹取平均，对齐论文式（3）。时间维不再取平均，因此
完整 8 步世界模型损失和 32 步 Critic 损失相较旧实现分别放大 8 倍、32 倍；
梯度裁剪保持论文配置。

重置评分按完成 episode 的起始 bin 批量更新，同批同 bin 先求平均再做 EMA。
物理失败的尾部价值为零；正常时限截断用本 episode 的平均奖励外推至动作末尾。
评分分母是从起始帧到动作末尾的最大折扣回报，因此满分短尾段和正常超时都不会被
误判为困难区间，提前跌倒仍得到低分。此尾部外推仅用于采样评分，不改 Critic
targets。采集用累计奖励和累计折扣回报，避免逐环境、逐时间步的 Python 循环。
Replay 常规采样也改为对随机候选做窗口/历史检查；仅候选极稀疏时回退到完整扫描。

## 断点、推理与真实评估

`last.pt` 是完整续训断点，包含三个网络、优化器、完整 Replay、重置采样器、训练计数
和各随机数发生器状态。`policy_best.pt` 按固定起点真实仿真的平均归一化回报选择，
包括初始化策略候选；它和 `policy_last.pt`、`policy_iteration_*.pt` 仅保存 Actor
与推理所需的配置和签名，不包含 Replay、World Model、Critic 或优化器。
两种格式都可评估并校验 manifest、动作名称、机器人签名和物理配置；只有完整断点
能用于 `--resume`。默认不再保存历史完整断点；显式添加
`--full-checkpoint-interval 10000` 才会生成 `iteration_010000.pt` 等文件。

已有完整断点可用 `scripts/export_supermimic_policy.py --checkpoint <完整断点>
--output <轻量策略.pt>` 提取推理权重，内存映射读取，不复制 Replay，也不改写源文件。
续训旧目录时，如旧 `best.pt` 存在而 `policy_best.pt` 不存在，会自动导出最佳策略。
现存历史文件不会自动删除；正在运行的进程须从 `last.pt` 重启后才会使用新的保存方式。
缺少 `reference_schema` 时，导出和推理会按 Actor 编码器输入维度识别 730／2680
格式，并标记 `reference_schema_inferred: true`；未知维度不会擅自补成新版格式。

采样器评分语义已升级为版本 2，旧训练断点不能直接续训；请使用新的输出目录开始
对照实验。关闭自碰撞后机器人模型签名也已改变，因此更早物理配置下的断点不能
直接在新配置下续训或做同配置评估；需用对应版本代码复现旧结果。
新断点保存观测/终止模式以及固定模型诊断窗口，续训必须使用原有配置。
`simulation` 字段记录物理协议版本、自碰撞设置、训练初态扰动范围和随机化频率，
供续训及评估校验。相同 seed 和环境数量会重建相同的启动随机化参数。

继续训练时可省略原有网络和 batch 参数，它们会从断点恢复；显式改变训练配置将被
拒绝。`--iterations` 是包含已经完成轮数的总轮数：

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --resume outputs/supermimic/online_training/last.pt \
  --iterations 5000 --output outputs/supermimic/online_training
```

恢复后物理仿真开始新的 episode，编号高于保存 Replay 中各环境的最大 episode
编号。历史数据继续可用，窗口不会跨越恢复边界；不会声称逐位恢复仿真轨迹。训练
恢复要求同类设备后端，单独的 Actor 推理可切换 CPU/CUDA。评估长度协议在续训中
保持一致，以便比较 best checkpoint。

直接加载在线 checkpoint 进行真实仿真评估：

```bash
python scripts/evaluate_supermimic.py \
  --device cuda:0 --checkpoint outputs/supermimic/online_training/policy_best.pt \
  --output outputs/supermimic/online_policy_evaluation.json
```

默认评估训练动作库中的所有动作，从各动作首帧开始、关闭域随机化和观测噪声，
运行至物理失败或动作结束。评估使用独立环境，并恢复训练 RNG，不消耗训练环境的
episode。原有 `--policy module:function` 入口继续支持，与 `--checkpoint` 互斥。

终端以 `Evaluation return <分数>` 汇总，单动作分数为累计奖励除以
`5 × 0.02 × (完整动作帧数 − 1)`，再对动作取平均。`normalized_return_framew`
给出按完整动作长度加权的分数；`n_failed` / `failed` 只统计物理失败。
调试步数限制和意外超时会将 `evaluation_complete` 标为 `false`，终端显示
`INCOMPLETE`。加 `--csv outputs/supermimic/eval.csv` 可追加 checkpoint 比较记录。

`python scripts/play_supermimic.py --checkpoint <路径> --clips walk1_subject1 --loop`
提供真实仿真策略与蓝色半透明参考动作的 MuJoCo 实时对照。空格暂停，右方向键单步，
`G` 切换参考动作；`--speed 0.5` 慢放，`--global-ghost` 查看绝对位移与朝向偏差。
无桌面环境可用 `MUJOCO_GL=egl python scripts/evaluate_supermimic.py --checkpoint <路径>
--clips walk1_subject1 --video outputs/supermimic/walk.mp4` 导出视频，需要 `ffmpeg`。
渲染使用独立 MuJoCo 缓冲区，显示 `transition.next_state`，不会改写评估仿真或将
自动重置后的新姿态误当作失败终帧。详见 [README 实时验证说明](../README_zh.md#mujoco-实时动作验证)。
播放入口默认支持鼠标外力：双击实体机器人身体，按住 `Ctrl`＋右键拖拽施力，
`Ctrl`＋左键拖拽施加力矩。外力通过原生查看器相同的 `qfrc_applied` 通道进入物理
仿真，未将拖拽转换为姿态写入。松开、换动作或关窗口会清除外力；可用
`--no-perturbations` 禁用。标准评估默认禁用鼠标力，受干预的播放轮次在终端及
JSON／CSV 中明确标记为 `interactive`，不会混作标准评估成绩。
播放默认关闭高度／倾斜失败终止，机器人跌倒后继续执行到当前动作结束；加
`--termination` 可恢复失败重置。标准评估仍默认开启终止检测。关闭终止的无外力播放
标为 `playback`，并记录 `termination_enabled: false`；失败阈值仍用于诊断首次失败
步数和原因，完成播放不等于跟踪成功。动作结束及显式时限仍正常截断和重置。

输出还包括 `config.json`、每轮 `metrics.jsonl`、最近评估 `evaluation.json` 和
编号评估报告。训练日志记录真实奖励、失败次数、想象奖励、生存权重、三类更新计数、
梯度范数、回放规模、各阶段时间和吞吐。PyTorch 显存峰值只统计其分配器，不包括
MuJoCo Warp 和驱动的其他分配。
新增 `actor_reward_objective`、`actor_bootstrap_objective` 分解 Actor J；
`real_*` 包含奖励各分项及物理误差。`episode_steps_sum`、`episode_returns_sum`、
`normal_timeouts`、`normal_motion_ends` 记录已结束 episode 的长度、回报和结束原因，
失败优先于动作结束，动作结束优先于时限截断，避免重复统计。W&B 同步记录这些指标。
采集批次的奖励、失败计数等统计保留最近一次值；汇总采集批次时只计入
`collected_this_iteration=true` 的行，实际累计转移数使用 `environment_steps`。

每次真实策略评估同时输出 `diagnostics_NNNNNN.json` 和 `diagnostics.json`：

- 1/8/32 步记录动作回放的状态各组端点 RMSE、奖励 MAE、终止 Brier，以及恒速
  预测基线；每个 horizon 只使用完整窗口，不把 padding 当成有效预测。
- 首次取得完整窗口后复制并固定，随 checkpoint 保存，之后不受 ring buffer 覆盖
  影响。初始数据不足的 horizon 标为 unavailable，下次评估重试。
- `--diagnostics-batch-size` 控制每个 horizon 的窗口数，默认 64。报告记录快照轮数、
  覆盖及缺失动作；它们是训练 Replay 的回归快照，不是独立留出验证集。诊断采样不
  消耗训练的 Replay RNG。
- 采样器熵和概率最高的 10 个 bin，包含动作名称、起始秒数和归一化回报估计。

## 修复后的短程对照

先使用下列完整规模、4000 轮基线检查采样器修复后的固定起点评估；每 1000 轮保存
和评估。需要更低成本时可同时减小环境和 batch 数量，但比较各组时保持相同规模。

```bash
.venv/bin/python scripts/train_supermimic.py \
  --device cuda:0 --num-envs 4096 --capacity 1000 \
  --collect-steps 5 --updates-per-collect 2 \
  --wm-batches 10 --wm-batch-size 2048 \
  --policy-batch-size 4096 --horizon 32 \
  --critic-epochs 5 --critic-minibatches 4 \
  --world-dims 2048 2048 2048 \
  --actor-dims 2048 1024 512 256 --critic-dims 256 128 \
  --iterations 4000 --eval-interval 1000 --checkpoint-interval 1000 \
  --diagnostics-batch-size 256 \
  --output outputs/supermimic/lafan_sampler_fixed
```

第二组仅增加 `--termination-mode analytic`，并换输出目录；第三组再增加
`--relative-orientation`，再次换目录。每组从头开始，使用同一 seed 和完整 40 段
动作评估。重点比较真实 normalized return、完成率、逐动作失败，以及 32 步奖励
误差；单看 Actor J 或 WM loss 下降不能证明策略改善。

失败时保留 traceback；输出目录已建立后，另写 `failure.json`。若再次出现此前偶发
的 CUDA/CPU stream 错误，可加 `--detect-anomaly`，并在命令环境设置
`TORCH_SHOW_CPP_STACKTRACES=1` 定位。重跑未复现不等于确认已修复。

## 终端与 W&B 日志

终端默认以表格显示区间平均指标，仿真器详细输出写入 `simulator.log`。
每轮原始 JSON 仍写入 `metrics.jsonl`。用 `--log-format json` 或 `--verbose-sim`
可切换输出方式。

追加 `--wandb --wandb-project supermimic --wandb-name RUN_NAME` 启用 W&B；
服务器离线时再加 `--wandb-mode offline`。安装、登录、分组曲线、逐动作评估图表和
续训行为见 [README](../README_zh.md#训练日志与-wb)。

## 验证

```bash
SUPERMIMIC_TEST_DEVICE=cuda:0 \
python -m unittest discover -s tests -v
```

无 GPU 时可将 `SUPERMIMIC_TEST_DEVICE` 改为 `cpu`。新增测试覆盖参考坐标和
四元数符号不变性、32 步状态梯度、历史动作反馈、末端 bootstrap 梯度、传感器
噪声时序、完整参考起点筛选、末帧 bootstrap、跨采集超时的想象与 GAE bootstrap、
世界模型及 Critic 损失的时间求和与 padding 排除、优化器隔离、新策略数据采集、
Replay 恢复以及
训练 checkpoint 的独立推理。

100 轮短训练用于验收在线闭环和数值稳定性。完整动作跟踪能力仍需以更充分训练后
的真实仿真结果判断，不能用想象回报上升代替真实策略性能。

历史短训结果单独保存在[实现验证记录](experiments/implementation_validation_zh.md)，不作为当前策略性能的验证结果。

## 梯度范数溢出与训练发散

`clip_grad_norm_` 报 `total norm ... is non-finite` 有两种情况：梯度元素已包含
NaN/Inf，或者梯度元素仍有限、但 FP32 L2 范数计算溢出（例如梯度量级为 `1e20`）。
不能仅凭这条异常区分两者。`checked_step` 保留常规裁剪路径；范数溢出时先检查
每个参数的梯度，全部有限才使用 FP64 重算范数并裁剪，随后执行原来的 Actor
逐元素裁剪。发现真实 NaN/Inf 时清空梯度并停止，错误中列出模型和参数名；
不会执行该次 optimizer step。不要通过 `error_if_nonfinite=False` 绕过检查，
也不要把 NaN/Inf 替换成零后继续更新。

如果真实奖励下降而 Critic loss 持续上升，应先检查 `metrics.jsonl` 中的
`actor_gradient_norm`、`critic_gradient_norm`、`actor_reward_objective`、
`actor_bootstrap_objective` 以及世界模型的状态误差。范数计算修复并不保证解决
世界模型、Actor 和 Critic 联合训练的发散。

可先在新输出目录进行较保守的短训实验：在原命令中增加
`--warmup-steps 64 --wm-only-iters 500 --learning-rate 5e-5`，将
`--horizon 32` 改成 `--horizon 8`，并缩短评估及存档间隔，例如
`--eval-interval 2000 --checkpoint-interval 200`。这些是排查配置，尚未验证
论文规模下的收敛；稳定后再单独验证 32 步设置。更改这些训练参数需要新建运行，
不能对原配置的 checkpoint 使用 `--resume`。

若再次报告真实非有限梯度，可用 `--detect-anomaly --no-compile --amp none`
定位最早出问题的反向算子；该诊断配置速度较慢，并可能增加显存占用。
