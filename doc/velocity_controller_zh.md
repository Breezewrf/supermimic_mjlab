# Velocity Controller

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
| 自碰撞 | 开启自碰撞时，`self_collision_weight * 超过 10 N 的子步数`，默认权重 `-1` |

每项只乘一次 `dt=0.02`。站立判断分别使用 `0.1 m/s` 和 `0.1 rad/s` 的阈值。
仿真与 imagination 共用速度、姿态和动作等解析奖励；自碰撞在仿真中读取真实接触力，
在 imagination 中使用由真实标签监督的可微代价预测，具体见下节。

跌倒条件为 root 高度 `<0.35 m` 或 root 倾斜 `>70°`。Imagination 以相同物理量计算
sigmoid 概率，高度宽度为 `0.02 m`，重力投影宽度为 `0.04`。时间上限属于 truncation，
不作为跌倒训练标签。默认 episode 为 10 秒。

设计参考 [SuperMimic 附录 B/G、表 A.6](supermimic.pdf) 与
[Unitree velocity 实现（1425b15）](https://github.com/unitreerobotics/unitree_rl_mjlab/tree/1425b15f73bd4095f0df53709d7c389c3eb9e790/src/tasks/velocity)。
70° 阈值是参考 Unitree 的工程选择，论文阈值约为 73°。
Unitree 的固定 gait、contact、air-time、soft-landing 奖励需要额外接触信息，当前不加入训练目标；
没有将仿真接触值或 reference anchor 作为速度 Actor 的隐藏输入。

## 训练闭环与优化目标

当前做法是在已有 motion policy 上增加 velocity encoder，并在世界模型中通过可微 rollout
优化速度回报。Motion encoder 根据未来参考动作生成 latent，velocity encoder 根据本体历史和
速度指令生成 latent，两者使用同一个 decoder：

$$
z_t^{motion}=E_{motion}(r_t),\qquad
z_t^{vel}=E_{vel}(h_t,c_t),\qquad
a_t=D(h_t,z_t).
$$

这里 `h_t` 表示归一化后的最近 10 帧本体观测，`r_t` 表示未来参考特征，
`c_t` 为三维速度指令，进入 encoder 前按 `command_scale` 缩放。
29DoF Actor 每帧不直接读取 root 线速度；真实速度用于训练奖励和 privileged critic。
控制周期为 `0.02 s`（50 Hz），物理仿真周期为 `0.005 s`，每个动作保持 4 个物理步。
Decoder 输出经各关节 action scale 和默认姿态偏置转换为关节位置 PD 目标。

训练数据和参数更新形成以下闭环：

1. **真实采集。** 一个仿真实例中的环境各分一半执行 motion tracking 和 velocity tracking，
   分别写入两个 replay。默认启动先采集 32 个控制步，之后每 2 轮训练采集 5 步。
   训练采集包含域随机化、观测／reset 噪声，以及标准差为 `0.1` 的动作噪声。
2. **动力学更新。** 在线模式每轮从两类 replay 各取一半 batch，用记录动作执行 8 步
   world-model rollout，监督预测状态；motion termination head 只使用 motion 标签。
   默认每轮更新 10 个 world-model batch。冻结模式跳过这一步，并固定模型归一化统计。
3. **Motion 更新。** Joint 阶段从 motion replay 抽取起点，在世界模型中优化 motion tracking
   回报，更新 motion encoder 和共享 decoder，再拟合 motion critic。
4. **Velocity 更新。** 从 velocity replay 抽取真实状态、历史及指令调度状态作为起点，
   用当前 velocity encoder 和共享 decoder 生成动作，在世界模型中预测下一状态，重新计算
   本体观测、奖励和跌倒概率，再将观测加入历史。默认展开 32 步，即 `0.64 s`。
   未来指令由与真实采集相同的调度器生成，包含参考速度变化和随机指令重采样。
5. **Velocity critic 更新。** 使用 imagination 中的奖励、存活概率和价值构造 GAE/λ-return
   目标，拟合 velocity critic；默认 `gamma=0.99`、`gae_lambda=0.95`。
   两个 critic 各自默认训练 5 个 epoch，每个 epoch 分 4 个 minibatch。

令 `p_t` 为第 `t` 次预测转移后的跌倒概率，`S_0=1`、
`S_t=∏_{k=0}^{t-1}(1-p_k)` 为到达第 `t` 步前的累计存活概率。
Velocity encoder 最大化折扣奖励与末端价值之和：

$$
J_{vel}=\mathbb{E}\left[
\sum_{t=0}^{H-1}\gamma^t S_t r_t
+\gamma^H S_H V_{vel}(\hat s_H,c_H,a_{H-1},\mathrm{context})
\right],\qquad H=32.
$$

预热阶段只优化参考来源样本上的 latent MSE。Joint 阶段的 velocity 损失为：

$$
L_{vel}=-J_{vel}+\lambda_{align}L_{align},\qquad
L_{align}=\underset{\text{参考来源样本}}{\mathrm{mean}}
\left[\mathrm{mean}_{32}\left((z_t^{vel}-z_t^{motion})^2\right)\right].
$$

`alignment_weight` 默认 `0.1`。随机指令没有对应的 motion latent，不计算对齐损失。
Velocity 更新时固定 motion Actor、world model 和 velocity critic 的参数，但保留其输入
Jacobian，使梯度能通过 decoder、预测动力学、奖励和末端价值传回 velocity encoder。
**速度回报只更新 velocity encoder；共享 decoder 由 motion tracking 回报更新。**
预热阶段两个 critic 都不进行价值拟合。

具体实现见 [训练器](../src/tasks/supermimic/rl/velocity_training.py)、
[velocity imagination](../src/tasks/supermimic/rl/velocity_imagination.py) 和
[共享 decoder 接口](../src/tasks/supermimic/models/velocity_policy.py)。

### Motion critic 与 velocity critic

两个 critic 分别估计各自任务下的未来折扣回报：motion critic 判断当前状态能否继续跟好
参考动作，velocity critic 判断能否继续完成速度指令。

| 项目 | Motion critic | Velocity critic |
| --- | --- | --- |
| 任务条件 | 未来参考动作特征 | 三维速度指令 `(vx,vy,wz)` |
| 身体状态坐标 | 相对参考 anchor 的位置、朝向和速度 | 高度、去除 yaw 的 6D 朝向、heading-frame 速度 |
| 其他输入 | 关节状态、仿真 context | 关节状态、上一动作、仿真 context |
| 回报目标 | Motion tracking 奖励 | Velocity tracking、stand-still、自碰撞等 velocity 奖励 |
| 默认隐藏层／输出 | `(256,128)`／一个价值标量 | `(256,128)`／一个价值标量 |

29DoF、`reference_dim=2680` 时，motion critic 输入为 `71+2680+33=2784` 维，
velocity critic 输入为 136 维。两者均使用训练时的特权物理状态，不受 Actor 的本体观测限制。

相同身体状态的任务价值可能不同：稳定站立对于零速度指令可能具有较高价值，但对于跑步
参考动作可能具有较低价值。两个任务的奖励、终止规则和后续策略也不同，因此分别拟合
`V_motion` 和 `V_velocity`，并分别维护归一化统计和优化器。

两者都使用各自 imagination 轨迹的 GAE/λ-return 目标拟合价值，并用末端价值补充默认
32 步 rollout 之外的未来回报。更新 Actor 时固定 critic 参数，但保留末端价值对预测状态的
输入梯度。Motion critic 服务于 motion encoder／共享 decoder 的更新；velocity critic
服务于 velocity encoder 的更新。两个 critic 都不参与部署推理。

分别使用两个独立网络是当前的实现选择，并非理论上的唯一方案；也可以设计共享主干、
两个价值输出头或带任务条件的统一 critic，但仍需区分两种任务的价值目标。
具体见 [Motion critic](../src/tasks/supermimic/models/policy.py) 和
[Velocity critic](../src/tasks/supermimic/models/velocity_policy.py)。

## 与 SuperMimic 论文的对照

以下以[本地论文](supermimic.pdf) §4.5、附录 B、表 A.4/A.5/A.6 和附录 G.1 为依据，
对照当前 velocity 训练入口与对照 launcher。论文没有明确说明的设置标为“未说明”，
不据此推断作者实现一定采用另一种方式。该对照聚焦 velocity 接口，不代表对底层仿真、
观测和 world model 所有细节做了完整一致性验证。

### 实现一致的部分

| 项目 | 论文方法与当前实现 |
| --- | --- |
| 多接口策略 | Motion encoder 和 velocity encoder 使用同一个 decoder，并共用动力学模型 |
| Velocity 输入与 latent | 三维速度指令、最近 10 帧本体观测信息；MLP 隐藏层 `(256,256)`，输出经过 `tanh` 的 32 维 latent；论文未进一步指定观测历史特征的全部实现细节 |
| 共享 decoder | 输入 latent 和最近 10 帧观测，默认隐藏层 `(2048,1024,512,256)`；当前预训练初始化继承 checkpoint 的结构 |
| 参数更新边界 | Velocity 回报只更新 velocity encoder；梯度穿过 decoder，但不更新其参数；motion tracking 回报更新 motion encoder 和共享 decoder |
| Latent 对齐 | 参考来源样本上的 velocity latent 对齐对应 motion latent，MSE 权重默认 `0.1`；对 motion latent 停止梯度 |
| Joint 阶段保留对齐 | 对齐并非仅用于预热，当前 joint 损失仍为 `-J_velocity + 0.1 * L_alignment`；随机来源样本不计算对齐，与论文一致 |
| Critic | 每个接口有独立 critic，默认隐藏层 `(256,128)`；分别拟合各自任务的价值 |
| 环境和 imagination 配比 | Motion／velocity 各占一半环境与 imagined batch；论文和当前 launcher 均使用 4096 个环境 |
| Joint 指令来源 | 每次重采样以 50% 概率选择参考来源，否则使用随机来源；随机分支以 10% 概率置零 |
| 指令坐标与参考平滑 | 平移速度在 yaw-only heading 系，yaw rate 为世界系角速度 z 分量；参考指令使用中心 25 帧平均 |
| 最终随机指令范围 | `vx[-1.5,2] m/s`、`vy[-1.32,1.32] m/s`、`wz[-3.35,3.35] rad/s` |
| 想象轨迹起点 | 每种接口从其自身仿真采集的数据中抽取状态 |
| 基础速度奖励 | 平面速度、yaw rate、动作平滑、关节限位、torso 侧倾和横滚／俯仰角速度；默认基础公式及权重对应表 A.6，各项乘控制周期 `0.02 s` |
| 主要优化设置 | 默认 horizon 32、discount 0.99、GAE λ 0.95、world-model 窗口 8 步、critic 5 epoch × 4 minibatch；当前 launcher 使用 4096 条 imagined rollout 和 10 × 2048 个 world-model 窗口 |
| 在线动力学训练 | 默认持续用真实转移拟合共享 world model；策略反传期间固定模型参数，保留输入梯度 |

### 差异与额外选项

| 项目 | 论文描述 | 当前实现 |
| --- | --- | --- |
| 独立 latent 预热 | 未明确描述先做纯 MSE 预热的阶段 | 默认先做 500 次 latent MSE 更新，再进入 joint；预热不拟合 critic |
| 随机指令课程 | G.1 直接给出上述完整随机范围，没有描述小范围扩围课程 | 初始使用较小范围，默认累计 5000 个采集控制步后扩围到论文范围；计数不乘环境数 |
| 指令重采样周期 | Reset 时和每 250 个控制步，即固定 5 秒 | Reset 时和各环境独立采样的 3～8 秒保持时间到期时 |
| Velocity reset 姿态 | G.1 未明确说明 velocity 环境独立的站姿／参考姿态配比 | 50% 默认站姿，50% 参考动作身体状态 |
| 首次超时打散 | 未描述这一专门安排 | 训练启动和恢复时默认随机安排首个 episode 的超时，随后使用完整时长 |
| 跌倒倾角 | `1 + u_z > 0.7`，约 73°；高度阈值 0.35 m | 倾角 70°；高度阈值同为 0.35 m |
| 整个 run 冻结 world model | 算法 A.1 持续拟合模型；只在策略反传时固定模型参数 | 额外支持 `--freeze-world-model`，固定整个 run 的模型权重和归一化统计 |
| Stand-still 姿态奖励 | 表 A.6 未列出 | 可选 Unitree 默认姿态或 MotionBrick 固定姿态；训练入口权重默认 0，对照 launcher 默认 -1 |
| 自碰撞和代价预测头 | 表 A.6 未列出该奖励，也未描述碰撞代价预测头 | 可选物理自碰撞、指定排除对、真实接触力惩罚及 imagination 代价预测头；新 velocity 入口默认开启，对照 launcher 默认关闭 |
| 额外指令／奖励设置 | G.1／表 A.6 未描述当前这些扩展 | `axis_command_probability`、`command_deadband`、`tracking_vertical_weight`、`tracking_angular_xy_weight` 默认均为 0，未启用时不改变对应基础行为 |
| 动作库 | G.1 使用从 LAFAN1 和 BONES-SEED 筛选的走跑库 | Launcher 分别选择 BONES-SEED、LAFAN1 或 MotionBrick |
| 训练长度 | Velocity 实验为 100000 轮 | 训练入口默认 100000 轮，对照 launcher 为 5000 轮 |
| 学习率衰减 | 通用表 A.5 为 250000 轮余弦衰减到 0.03 倍，之后保持；G.1 未另说明修改 | 按当前 run 总轮数完成余弦衰减到 0.03 倍；launcher 的 5000 轮也会完成整段衰减 |

论文的 distillation baseline 也不等同于当前的 latent MSE 预热：该 baseline 以参考速度作为
command，通过 motion-tracking 回报训练 velocity encoder；RL encoder 则使用速度任务回报，
并在参考来源 rollout 上保留 latent 对齐。原文 §6 讨论了并发 motion 训练可能并非必要，
但 G.1 实际报告的实验仍采用联合训练，不能据此声称论文验证了固定整个 motion Actor 的方案。

### Command 来源与 reset 姿态是两次不同的采样

“50% 参考速度／50% 随机速度”决定机器人接收什么指令；“50% 站姿／50% 参考姿态”
决定机器人从什么身体状态开始运动。论文和当前 joint 阶段都明确采用前一种 command 配比；
后一种独立 reset 配比是当前代码的设置，不能将论文“未说明”理解为全部使用随机速度或随机姿态。

当前两次选择分开进行，因此既可能从默认站姿开始并接收参考速度，也可能从参考动作姿态
开始并接收随机速度。参考 command 的片段／帧游标不要求与 reset 所选参考姿态相同。
预热时 command 全部使用参考来源，但机器人历史仍来自当前策略实际控制的真实仿真。

## 自碰撞、排除对与惩罚

新 velocity run 默认开启机器人自碰撞，并加入自碰撞惩罚。`--self-collisions` 可显式开启，
`--no-self-collisions` 可关闭；独立 motion 训练默认仍保留原来的无自碰撞配置。
碰撞只作用于 `.*_collision` 几何，视觉 mesh 不参与。脚部保留 `condim=3`，其余碰撞几何
使用 `condim=1`；开启时 `contype=conaffinity=1`，地面碰撞继续生效。

默认排除以下身体对，参考 Unitree 的 G1 XML，并对应本仓库的身体命名：

| 变体 | 排除对 |
| --- | --- |
| 29DoF | 左／右 `elbow_link` 与同侧 `wrist_pitch_link` |
| 23DoF | 左／右 `elbow_link` 与同侧 `wrist_roll_rubber_hand` |
| 两种变体 | `pelvis` 与左／右 `hip_roll_link` |

MuJoCo 原有的同刚体和父子刚体碰撞过滤仍生效。排除对集中定义于
[`self_collision_excludes()`](../src/tasks/supermimic/config/robot_cfg.py)，需要增加指定身体对时
修改该函数；实际列表写入 checkpoint 的 `simulation.self_collision_excludes`，恢复和播放时
严格核对，避免使用不同的碰撞规则。训练中 motion/velocity 两半环境采用同一套碰撞物理设置。

真实代价沿用 Unitree 的力阈值子步计数口径：每个 `0.005 s` 物理子步检测机器人自身接触，
若任意自碰撞力的范数超过 `10 N`，该子步记为 1。每个控制步共 4 个子步，代价 `C_t` 为
`0～4`，奖励为 `self_collision_weight * C_t * 0.02`。默认权重 `-1`，可通过
`--self-collision-weight` 或 velocity JSON 的 `self_collision_weight` 设置为非正有限值。
这里使用 `maxforce` 选择最强接触，避免 Unitree 的 `reduce="none"`、单 slot 随机保留接触时
漏掉超过阈值的接触；接触历史在 reset 时清空，标签在 reset 前写入两类 replay。

Imagination 没有真实接触力，因此开启自碰撞的 world model 增加一个
`MLP(128,128) → 1` 代价预测头，输入为与动力学相同的归一化状态、动作和 context 特征。
以真实 `C_t/4` 为 soft label，使用 masked BCE 与动力学损失一起训练，
在想象轨迹中用 `4 * sigmoid(logit)` 估计代价。策略更新时固定预测头参数，但保留其对
状态和动作的输入梯度，使碰撞惩罚能够传回 velocity encoder。该预测不读取未来参考，
也不作为 Actor 的额外输入。共享 decoder 仍只通过 motion tracking 更新。

真实惩罚和预测代价的口径一致，但预测值不保证等于真实计数；它是新增的学习模型，
应同时检查 `loss/world_model/self_collision` 和真实 `reward/velocity/self_collisions_per_second`。
`velocity_diagnostics.json` 另记录各窗口真实输入上的 `self_collision_cost_mae`，单位为每控制步
的碰撞子步数；这是 replay 内预测诊断，不代表 held-out 泛化。
开启自碰撞会改变动力学。新 run 允许从无自碰撞的 motion policy/world model 初始化，
仅新增碰撞预测头，其余权重严格匹配，并记录 `initialization_collision_transfer`。
这种迁移要求在线更新 world model；从旧无自碰撞 world checkpoint 初始化时不能同时指定
`--freeze-world-model`。已经适配且含预测头的新 world checkpoint 才可冻结使用。

```bash
python scripts/train_velocity.py \
  --actor-checkpoint outputs/policy_iteration_010000.pt \
  --world-checkpoint checkpoints/best/world_model_best.pt \
  --self-collisions --self-collision-weight -1.0 --no-freeze-world-model \
  --stage joint --warmup-updates 500 --device cuda:0 --num-envs 4096 \
  --iterations 100000 --output outputs/velocity/self_collision_run
```

旧 checkpoint 恢复时缺失的 `self_collisions` 按 `False` 处理，保留原物理设置。
`--resume` 不允许改变碰撞开关或惩罚权重；更改物理设置应使用新 run 初始化。
导出的 velocity/motion policy 和 world model 均保存实际碰撞配置，播放和评估自动恢复该设置。

## 初始化及分阶段训练

默认初始化文件：

- `outputs/policy_iteration_010000.pt`：已有 93 维 motion 策略及输入归一化。
- `checkpoints/best/world_model_best.pt`：共享动力学和对应归一化。

两个文件可分别指定。脚本验证机器人关节／刚体顺序、拓扑、物理配置以及 93 维输入；
模型和数据 SHA 仅作可选溯源，允许使用新的 motion manifest 和训练中未见的动作。
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
在线模式同时更新世界模型；`--stage motion` 优化 motion Actor/Critic，以及在线模式的世界模型，
仍使用混合环境进行采集。需要适配 motion prior 时，可将该阶段输出的
`motion_policy_last.pt` 和 `world_model_last.pt` 作为下一次 joint run 的初始化。

### 预热阶段的 command 与 latent 对齐

预热时，velocity 环境全部使用 motion dataset 的参考来源指令：调度器选择参考片段和帧，
读取该帧经过中心 25 帧平均的速度作为 `c_t=(vx,vy,wz)`。`vx/vy` 使用 heading 坐标系，
`wz` 使用世界坐标系角速度的 z 分量。参考帧逐步推进，指令按参考速度变化；目标 latent
使用同一参考游标对应的未来 10 帧动作特征。

$$
z_t^{motion}=E_{motion}(r_{t:t+9}),\qquad
z_t^{velocity}=E_{velocity}(h_t,c_t),\qquad
L_{warmup}=\mathbb{E}\left[\mathrm{mean}_{32}
\left((z_t^{velocity}-z_t^{motion})^2\right)\right].
$$

这里 `h_t` 是归一化后的最近 10 帧真实仿真观测。Velocity 环境由当前 velocity encoder
和共享 decoder 控制，历史来自它们实际控制的机器人，而不是直接读取 dataset 的观测历史。
Motion encoder 只提供不带梯度的 latent 监督目标，不替代 velocity 环境中的动作输出。
参考指令游标也不要求机器人实际姿态与 dataset 中该帧姿态完全一致。

预热只更新 velocity encoder 的 latent MSE，不通过速度回报优化 encoder。Motion encoder、
共享 decoder 和共享 Actor 观测归一化统计固定；两个 critic 不进行价值拟合，但采集仍会
更新 critic 输入归一化统计。在线模式下 world model 继续用真实采集数据监督更新，
冻结模式则保持 world model 权重和统计不变。

预热结束后进入 joint，开始速度回报优化和两个 critic 的价值拟合，参考／随机指令按正常
调度混合；latent 对齐继续保留，但只对参考来源的样本计算。
具体见 [预热与联合更新](../src/tasks/supermimic/rl/velocity_training.py) 和
[参考指令调度](../src/tasks/supermimic/mdp/velocity_commands.py)。

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

默认在线更新 world model。使用 `--freeze-world-model` 可固定预训练动力学权重及其
输入／输出归一化统计，并跳过动力学训练；冻结模式要求预训练 world checkpoint。
该选项不会冻结 joint 阶段的共享 decoder，也不会切断策略优化所需的模型输入梯度。

冻结参数不会关闭计算图。速度回报仍通过 decoder、世界模型和末端 critic 的输入 Jacobian
反传到 encoder。归一化只从新采集的真实数据更新；imagination 不更新统计量。

默认只选 `walk*`、`run*`、`sprint*` clip，也可显式传 `--clips`。
训练开始前默认对所选 clip 各评估 250 步，写入 `initial_motion_evaluation.json`。
这是有步数上限的检查，不代表完整动作成功率；`--initial-evaluation-steps 0` 可关闭。

## 动作风格的继承与边界

当前结构具有从 motion dataset 继承动作风格的机制，但没有保证风格一致的硬约束。
这里的风格包括步频、步幅、摆臂、躯干姿态和接触时序等；速度跟踪误差本身不能衡量这些特征。

共享 decoder 在 motion tracking 中学习 latent 与身体状态到动作的映射，可能保留数据集中的
关节协调方式。Velocity encoder 借助已有 decoder 完成速度指令，参考来源样本上的 latent
对齐进一步鼓励它使用 motion encoder 的表示。预热先学习这一表示，joint 阶段继续保留
对齐损失，同时通过 motion 分支持续训练 decoder。

对相同历史 `h_t`，两个 latent 的局部动作差异可近似为：

$$
\Delta a_t\approx\frac{\partial D(h_t,z_t)}{\partial z_t}\Delta z_t.
$$

Latent 接近通常有利于单步动作接近，但放大程度取决于 decoder 的 Jacobian；
单步接近也不能保证长期轨迹和接触时序接近。具体边界包括：

- **Decoder 没有限定动作分布。** 它是普通神经网络，`tanh` 只限制 latent 的数值范围，
  不能保证每个 latent 都对应数据集中的自然动作。相同 decoder 在不同身体历史下也会输出不同动作。
- **速度不能唯一指定风格。** 同一个 `(vx,vy,wz)` 可以通过不同步频、步幅和摆臂实现。
  数据集包含多种风格时，当前接口没有额外的风格选择条件。
- **随机指令缺少直接对齐监督。** 默认 joint 重采样有 50% 概率选择参考来源，其余使用
  随机来源；参考库覆盖不足的组合指令需要策略组合或外推已有能力。
- **速度回报允许风格偏移。** Joint 目标在速度回报与 latent 对齐之间权衡。
  `alignment_weight=0.1` 的实际约束强度取决于两项的数值和梯度尺度。
  当前速度奖励没有直接约束参考动作的步频、足端轨迹或接触节奏。
- **Motion prior 在 joint 中仍会变化。** Motion encoder、共享 decoder 和共享观测统计
  并非始终固定，因此对齐目标也不是初始化 checkpoint 中永久不变的风格模板。

参考库同时影响参考指令、reset 状态、latent 对齐目标和共享 decoder 的 motion 训练；
在线模式下还通过 motion replay 影响 world model。更接近参考库覆盖范围的指令和身体状态，
理论上更容易沿用参考动作的协调方式，但实际保留程度需要单独评估。

风格评估应在实测速度相近时比较步频、步幅、摆臂幅度、躯干姿态和足端接触时序，
避免把速度差异当成风格差异。最近的[对比实验](experiments/velocity_tracking_2026-11-26_zh.md)
测量了速度误差、停车漂移和跌倒，已观察到参考库选择对跟踪效果的影响；尚未直接测量风格保持程度。

## 指令、数据边界与恢复

每次 reset，以及各环境独立采样的 3～8 秒保持时间到期时，重新选择指令来源：
50% 从参考动作读取，50% 随机采样；随机分支以 10% 概率置零。预热阶段使用参考来源。
`--command-hold-seconds MIN MAX` 可调整区间；仿真与 imagination 使用同一个调度器，
保持时间向上取整到 0.02 秒控制步，当前指令剩余时间进入 replay 和 checkpoint。
这是参考 Unitree 的去同步设置，与论文 G.1 的固定 250 步指令周期不同。
参考线速度先按每帧 heading 转换，再进行中心 25 帧平均，clip 边缘复制端点，禁止跨 clip 平滑。

初始随机范围为 `vx[-0.5,1.0]`、`vy[-0.5,0.5]`、`wz[-1,1]`；累计 5000 个采集控制步后
扩展至 `vx[-1.5,2]`、`vy[-1.32,1.32]`、`wz[-3.35,3.35]`。
此计数为每环境共享的控制步计数，包含预热，不乘并行环境数。
速度训练 reset 以各 50% 概率选择站姿和 locomotion 参考姿态。

课程配置集中在 [`VelocityConfig`](../src/tasks/supermimic/config/velocity_cfg.py)，
默认 `curriculum_steps=5000`。对照脚本直接使用代码默认值，不再依赖附带的 JSON 配置。
历史初始课程的门槛为 120000；提前扩围实验使用 5000。

[对照脚本](../scripts/velocity/train_velocity_comparison.sh)使用 4096 个环境、
`policy_batch_size=4096`、`wm_batch_size=2048`，从 BONES-SEED 10k motion policy 和
world model 初始化，总共训练 5000 轮（包含 500 次 latent 预热）。
该轮实验比较 BONES-SEED 参考子集和 LAFAN1 的 18 条 walk/run/sprint，以及 world model
在线／冻结两种模式。训练轮数与上述采集控制步计数分别累计。
这些历史实验没有开启自碰撞；当前对照脚本默认关闭自碰撞，也可显式开启。

新 run 可通过 `--velocity-config <settings.json>` 覆盖速度配置，未指定的字段沿用默认值。
`curriculum_steps` 和 `final_ranges` 控制随机指令何时扩围及扩围后的范围。
`axis_command_probability` 是随机分支中只保留一个均匀选取轴的概率。
`command_deadband` 将范数低于或等于阈值的随机指令置零。参考指令仍逐帧从 clip 读取。
`tracking_vertical_weight` 和 `tracking_angular_xy_weight` 可将 root 局部竖直速度和
横滚／俯仰角速度的平方分别加到线速度和 yaw 奖励的指数误差中。默认两项权重均为 0。

训练启动和恢复后的首个 episode，为 motion/velocity 两半环境独立随机安排首次超时，
剩余时长均匀分布于 1～500 控制步（默认最大 episode 为 10 秒）。随后每个 episode
恢复完整的 10 秒上限。使用独立超时计时，不伪造 `episode_step`、历史或折扣回报；
`--no-randomize-initial-episode-progress` 可关闭。该设置不影响独立播放或固定起点评估。

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
允许在 resume 时显式覆盖 `--command-hold-seconds` 和 episode 初始相位选项，
其他 velocity 配置仍需匹配。CLI 默认启用随机初始相位（包括 resume），
显式 `--no-randomize-initial-episode-progress` 可关闭。不兼容使用旧固定步数计时配置或缺少 episode 相位 RNG 的 checkpoint。
要在续训时调整指令保持时间，使用：

```bash
python scripts/train_velocity.py \
  --resume outputs/velocity/run/last.pt --device cuda:0 --iterations 100000 \
  --command-hold-seconds 3 8
```

当前已保存指令的剩余时间会继续使用，下次重采样采用新范围。新的 episode 相位 RNG
随 checkpoint 保存。
物理仿真重新开始 episode，使用新的 episode ID，避免把恢复后的新历史拼接到旧 replay。
保存文件不是逐 substep 的 MuJoCo 物理快照，不承诺续训物理轨迹逐位一致。

终端按 Reward/s、Error、Loss、Objective、Failure、Episodes、Timeouts 和 Performance 分区显示，`--log-interval 10` 控制区间统计输出频率；
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
| `reward/` | `reward/velocity/linear_tracking_per_second`、`reward/velocity/total_per_second` | 完成 episode 的加权奖励累计值除以配置的最大 episode 秒数；兼容的每步均值字段仍保留 |
| `error/` | `error/velocity/planar_velocity_m_per_s`、`error/velocity/yaw_rate_rad_per_s` | 真实物理误差，保留 m/s、rad/s 等单位，越小越好 |
| `loss/` | `loss/world_model/total`、`loss/velocity/latent_alignment_mse`、`loss/critic/velocity` | 模型、latent 对齐及 critic 拟合损失 |
| `objective/` | `objective/velocity/return` | 想象 rollout 的折扣回报与末端价值，不等于实际每步奖励 |
| `failure/` | `failure/velocity/episode_fraction` | 已完成 episode 中因物理失败终止的比例，终端转换为百分比 |
| `episode/`、`reset/` | `episode/velocity/completed`、`reset/velocity/time_out_count` | 完成数量、实际长度/回报，以及失败、timeout、clip-end 的独立计数 |
| `gradient/`、`diagnostic/` | 梯度范数、动作变化、想象存活率 | 训练诊断 |

velocity JSONL 使用 `metric_schema_version=2` 和 `statistics_scope=completed_episodes`，
`episode_statistics` 保存各接口的完成计数、累计量和每 episode 均值之和；`metric_groups`
给出分类结果。旧 JSONL（v1）仍可直接由新 W&B reporter 分类，无需重写日志。

奖励和物理误差只在真实 reset 前汇总，不把尚未完成的 episode 当成样本。
Reward/s 与 mjlab 一样，按每 episode 的奖励累计值除以配置的最大时长，再对完成 episode
等权平均；早期跌倒或启动时的短 episode 不会被外推成完整 episode 的回报。
物理误差先按各 episode 的实际步数求平均，再按 episode 数量平均。
旧的 `motion/*`、`velocity/*` reward/error 字段及 `total_per_step` 保留相同单位，
统计来源改为完成 episode，避免改变单位却沿用旧名称。
区间统计按完成 episode 数合并，未采集轮次不会重复计入；没有 episode 完成时显示 `--`。
恢复物理仿真时，未完成 episode 的统计随新 episode 清空，已完成报告不会重复上报。
模型、critic、latent 对齐等优化指标仍逐更新上报。最终的 1/8/32 步回放诊断单独归入
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

早期短跑验证完成了 5 次预热和 20 次联合更新，并完成续训检查。
9 类指令、每类 4 次的 5 秒测试未跌倒，但非零指令的相对误差约为 1，策略主要保持站立；
这验证了训练和推理链路，该短跑本身尚未得到收敛的速度跟踪策略。
后续 5000 轮训练和 672 个非零条件加站立的评估，见
[2026-11-26 对比实验](experiments/velocity_tracking_2026-11-26_zh.md)。

## Stand-still 姿态参考

选择 MotionBrick 时，stand-still 奖励直接使用 `mdp/velocity.py` 中的 `STAND_STILL_JOINT_POS` 固定数组，
取自 MotionBrick `locomotion_vx_0_vy_0` 第 1617 帧（32.34 秒）。23DoF 去掉对应的
waist roll/pitch 和 wrist pitch/yaw 角度。仿真和 imagination 共用这一参考，
观测和 PD 的默认姿态基准保持原值。参考角度无需从外部文件加载。
`--stand-still-weight` 默认仍为 `0`，设置负值（例如 `-1.0`）启用姿态惩罚。

使用 `--stand-still-reference motionbrick` 选择上述固定数组，使用
`--stand-still-reference unitree` 选择原来的机器人 `default_q`。
新训练默认 `motionbrick`；选择项随 velocity 配置保存，续训使用保存值，
旧 checkpoint 没有该字段时使用 `unitree`。续训时显式指定的值必须与保存值一致。
这只切换 stand-still 奖励目标，权重仍通过 `--stand-still-weight` 设置。

```bash
--stand-still-reference motionbrick --stand-still-weight -1.0
# 或使用原来的 Unitree 参考
--stand-still-reference unitree --stand-still-weight -1.0
```

## 间隔 checkpoint 与训练课程

Velocity 训练在 `--checkpoint-interval`（默认 1000）和训练结束时保留编号文件：
`velocity_policy_iteration_005000.pt`、`motion_policy_iteration_005000.pt`、
`world_model_iteration_005000.pt`，并更新对应 `*_last.pt`。
完整可续训 checkpoint 只更新 `last.pt`，不保留带 iteration 编号的历史副本。
编号对应实际 iteration，早期 run 已被覆盖的权重无法补回。

对照 launcher 支持 `--iterations N`（默认 5000）、`--curriculum-steps N`（默认读取
`VelocityConfig.curriculum_steps`）和 `--checkpoint-interval N`（默认 1000）。
课程步数计数为采集控制步，不乘并行环境数。
默认输出目录附加总迭代数和课程门槛，显式传 OUTPUT 时使用给定目录。
续训可增加总迭代数，但课程门槛必须与 checkpoint 保存值相同。

```bash
bash scripts/velocity/train_velocity_comparison.sh motionbrick180 online cuda:0 \
  --iterations 100000 --curriculum-steps 5000 --checkpoint-interval 1000 \
  --stand-still-reference motionbrick --self-collisions
```
