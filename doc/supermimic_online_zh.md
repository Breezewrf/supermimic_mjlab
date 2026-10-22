# SuperMimic 在线训练：采集、世界模型与 Actor–Critic

在线入口实现论文 §3.1、§3.3 和 Algorithm A.1 的三阶段循环：当前策略采集真实
仿真转移，8 步自回归拟合世界模型，再通过固定参数的世界模型进行 32 步想象，更新
Actor 和 Critic。默认从头初始化三个网络，不加载独立预训练世界模型，不以离线
32 步验证覆盖率作为前置门槛。

## 运行

在仓库根目录运行，使用当前 16 GiB GPU 的缩小 batch 配置：

```bash
taskset -c 0-7,10-31 .venv/bin/python scripts/train_supermimic.py \
  --device cuda:0 --iterations 1000 \
  --output outputs/supermimic/online_training
```

不指定 `--clips` 时使用 manifest 中全部 40 段 LAFAN 动作。输出目录必须不存在。
训练和推理命令按约定使用 `taskset -c 0-7,10-31`；普通检查无需指定 affinity。

默认配置：

| 参数 | 默认值 |
| --- | --- |
| 物理环境 / 回放容量 | 64 个环境，每环境 1000 步 |
| 初始采集 | 当前随机初始化 Actor 采集 32 个向量步，共 2048 条转移 |
| 数据与更新比例 | 每采集 5 个向量步，执行 2 轮训练 |
| 每轮世界模型拟合 | 10 个 batch，每个 batch 256 条 8 步窗口 |
| 每轮策略更新 | 64 条 32 步想象轨迹，更新 Actor 一次 |
| 每轮 Critic 更新 | 同一批固定 GAE targets，5 个 epoch，每个 epoch 4 个 minibatch |
| 学习率 / 折扣 | 三个网络均为 `2e-4`；γ=0.99、λ=0.95 |
| 学习率日程 | 250,000 轮内余弦下降至 0.03 倍，之后保持 |
| 保存 / 评估 | 每 100 轮，结束时也保存、评估 |

这里的 iteration 指一轮世界模型拟合加一次 Actor 更新及相应 Critic 拟合；初始采集
不更新网络。世界模型归一化由初始真实转移拟合并冻结，后续每轮继续更新网络参数。
这些 batch 和环境数量是面向当前硬件的缩放配置，不是论文完整训练规模。
仿真使用每环境接触分配 128、约束容量 1024；每个物理子步检查碰撞、接触和约束
计数，超限立即报错，避免继续保存不完整物理计算产生的转移。

快速检查可用较小的网络和单段动作：

```bash
taskset -c 0-7,10-31 .venv/bin/python scripts/train_supermimic.py \
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

每个参考帧展开为 268 维：29 维默认姿态相对关节角、29 维关节速度，以及 14 个
刚体的位置、6D 姿态、线速度和角速度。整个窗口使用首帧参考 pelvis 的水平位置
与 yaw 作为共同坐标锚点，保留高度和未来位移。位置除以 0.3，关节角除以 0.3，
关节速度除以 5，角速度除以 3.14。参考尾部沿用动作库的末帧填充。

Critic 为 `(256, 128)` ELU 网络，接收 71 维特权物理状态、2680 维参考特征与
33 维上下文，共 2784 维，输出标量价值。物理状态同样变换到当前参考窗口锚点；
四元数统一符号。Critic 与 Actor 不共享参数，真实推理仅需要 Actor。

Actor 本体观测沿用第一阶段定义：关节角、关节速度、机体根角速度、投影重力、
上一动作，采用固定量纲缩放。参考展开、6D 姿态表示及这些缩放属于本项目明确的
实现选择，论文没有给出逐字段展开约定。

## 三阶段更新

真实采集使用 `Actor(history, future_reference) + N(0, 0.1²)`，开启原有域随机化、
观测噪声和回报重置采样。Replay 保存实际执行动作以及 `policy_version`，后者
记录采集时已完成的 Actor 更新次数，便于核对采集确实使用了后续策略。

世界模型继续使用已有可微动力学和解析奖励、观测。在线拟合损失为归一化状态误差
加物理失败 BCE；解析奖励和观测误差只作为诊断。独立 `train_world_model.py` 的
原有四项拟合方式保留，以免改变已有离线实验的含义。
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
第 499 步，也可继续完成 32 步想象并保留末端价值 bootstrap。参考动作本身结束时
仍截断后续奖励和 bootstrap；正常到达 32 步想象长度时保留价值 bootstrap。

Actor 使用 Adam `β=(0.9, 0.95)`，梯度先按全局范数 1 裁剪，再按元素 0.1 裁剪。
世界模型与 Critic 使用 Adam，仅按全局范数 1 裁剪。Critic 的 GAE targets 基于
该轮想象轨迹计算一次并 detach，拟合 Critic 时不会更新 Actor 或世界模型。
Critic 按整条轨迹打乱、划分 minibatch；平方价值误差在每条轨迹的有效时间步上
求和，再对 minibatch 内的轨迹取平均，对齐论文式（3）。时间维不再取平均，因此
完整 8 步世界模型损失和 32 步 Critic 损失相较旧实现分别放大 8 倍、32 倍；
梯度裁剪保持论文配置。旧断点可加载，但续训将使用修正后的训练目标。

## 断点、推理与真实评估

`last.pt` 保存最近训练断点，`best.pt` 按固定起点真实仿真的平均归一化回报选择，
包括初始化策略候选。编号断点为 `iteration_000100.pt` 等。每个断点包含三个网络、
优化器、归一化、完整 Replay、重置采样器、训练计数和各随机数发生器状态，并校验
manifest、动作名称和机器人模型签名。

继续训练时可省略原有网络和 batch 参数，它们会从断点恢复；显式改变训练配置将被
拒绝。`--iterations` 是包含已经完成轮数的总轮数：

```bash
taskset -c 0-7,10-31 .venv/bin/python scripts/train_supermimic.py \
  --device cuda:0 --resume outputs/supermimic/online_training/last.pt \
  --iterations 5000 --output outputs/supermimic/online_training
```

恢复后物理仿真开始新的 episode，编号高于保存 Replay 中各环境的最大 episode
编号。历史数据继续可用，窗口不会跨越恢复边界；不会声称逐位恢复仿真轨迹。训练
恢复要求同类设备后端，单独的 Actor 推理可切换 CPU/CUDA。评估长度协议在续训中
保持一致，以便比较 best checkpoint。

直接加载在线 checkpoint 进行真实仿真评估：

```bash
taskset -c 0-7,10-31 .venv/bin/python scripts/evaluate_supermimic.py \
  --device cuda:0 --checkpoint outputs/supermimic/online_training/best.pt \
  --output outputs/supermimic/online_policy_evaluation.json
```

默认评估训练动作库中的所有动作，从各动作首帧开始、关闭域随机化和观测噪声，
运行至物理失败或动作结束。评估使用独立环境，并恢复训练 RNG，不消耗训练环境的
episode。原有 `--policy module:function` 入口继续支持，与 `--checkpoint` 互斥。

输出还包括 `config.json`、每轮 `metrics.jsonl`、最近评估 `evaluation.json` 和
编号评估报告。训练日志记录真实奖励、失败次数、想象奖励、生存权重、三类更新计数、
梯度范数、回放规模、各阶段时间和吞吐。PyTorch 显存峰值只统计其分配器，不包括
MuJoCo Warp 和驱动的其他分配。
采集批次的奖励、失败计数等统计保留最近一次值；汇总采集批次时只计入
`collected_this_iteration=true` 的行，实际累计转移数使用 `environment_steps`。

失败时保留 traceback；输出目录已建立后，另写 `failure.json`。若再次出现此前偶发
的 CUDA/CPU stream 错误，可加 `--detect-anomaly`，并在命令环境设置
`TORCH_SHOW_CPP_STACKTRACES=1` 定位。重跑未复现不等于确认已修复。

## 验证

```bash
taskset -c 0-7,10-31 env MPLCONFIGDIR=/tmp/supermimic-mpl XDG_CACHE_HOME=/tmp/supermimic-cache \
SUPERMIMIC_TEST_DEVICE=cuda:0 \
  .venv/bin/python -m unittest discover -s tests -v
```

无 GPU 时可将 `SUPERMIMIC_TEST_DEVICE` 改为 `cpu`。新增测试覆盖参考坐标和
四元数符号不变性、32 步状态梯度、历史动作反馈、末端 bootstrap 梯度、传感器
噪声时序、动作结束边界、跨采集超时的想象与 GAE bootstrap、世界模型及 Critic
损失的时间求和与 padding 排除、优化器隔离、新策略数据采集、Replay 恢复以及
训练 checkpoint 的独立推理。

100 轮短训练用于验收在线闭环和数值稳定性。完整动作跟踪能力仍需以更充分训练后
的真实仿真结果判断，不能用想象回报上升代替真实策略性能。

## 历史 GPU 验收结果

以下 100 轮结果来自修正采集超时和损失时间归约之前的实现，保留用于记录当时的
训练闭环与恢复验证；不代表修正后目标的重新训练结果。

有效结果位于 `outputs/supermimic/online_gpu_validated/`。使用默认网络、64 个环境、
8 步世界模型拟合和 32 步 BPTT，从头启动并完成 100 轮；执行会话中断后，实际从
第 20 轮断点恢复，验证了 GPU 优化器、Replay、采样器与 RNG 的恢复路径。

- 真实采集共 18,048 条转移，世界模型更新 1000 次、Actor 更新 100 次、Critic
  更新 2000 次，逐轮指标均为有限值。
- 完整 GPU 回归测试 46 项全部通过；包含碰撞容量保护、新策略采集、32 步梯度、
  末端价值梯度、恢复和独立推理。
- 平均每轮约 2.58 秒，PyTorch 分配器显存峰值约 1.43 GiB。
- 有效训练和续训日志未出现碰撞缓冲区溢出或 CUDA/CPU stream 错误。
- 第 100 轮 checkpoint 的独立 GPU 推理已对全部 40 段动作运行未限长评估，平均
  归一化回报与训练内评估之差小于 `1e-7`。

`verification.json` 汇总验收结果；目录内保留 `training_initial.log`、
`training_resume.log`、`tests.log`、`inference.log` 和独立推理报告
`inference_evaluation.json`，供复查。

未限制长度的全部 40 段动作真实评估：

| 更新轮数 | 平均归一化回报 | 完整动作成功数 |
| --- | --- | --- |
| 0 | 0.00179334 | 0/40 |
| 50 | 0.00163616 | 0/40 |
| 100 | 0.00168088 | 0/40 |

这次短训尚未改善真实跟踪回报，也尚未学会完整动作，当前 `best.pt` 因此仍是
初始化策略候选。验收证明训练闭环与恢复链路可以运行，不代表已复现论文的策略性能。
后续延长训练时应继续检查真实回报、失败率和模型误差，不能仅根据想象回报判断。

早期 `outputs/supermimic/online_gpu_check/` 曾出现碰撞容量溢出，已用
`verification_status.json` 标记无效，不作为本次验收结果。
