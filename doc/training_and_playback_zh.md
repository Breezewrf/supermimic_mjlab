# 训练、断点、评估与播放操作指南

[English](training_and_playback_en.md) | [返回 README](../README_zh.md)

先完成[安装](setup_zh.md)和动作转换，以下命令均在仓库根目录运行。[README 预览模型介绍](../README_zh.md#复现当前预览模型第-10000-轮)记录当前快照、实测评估结果及其训练命令。

## 训练变体

### 缩小训练规模

使用默认网络、64 个环境、每轮 10×256 条世界模型窗口和 64 条想象轨迹：

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --iterations 1000 \
  --output outputs/supermimic/online_training
```

默认启用域随机化、观测噪声和按回报采样的重置。
训练初态加入参考实现范围的位姿、速度和关节位置扰动；关闭自碰撞，保留全身与
地面接触。摩擦与 torso CoM 在环境启动时采样并跨 episode 固定，编码器偏差逐
episode 重采样；评估关闭这些扰动。范围与生命周期见[环境说明](supermimic_foundation_zh.md)。
不指定 `--clips` 时使用全部动作；调试时可指定 `--clips walk1_subject1 run1_subject2`。

### 论文单 GPU 训练规模

在仓库根目录运行以下命令，使用论文 Table A.4–A.5 的网络宽度、训练批量、
环境数量与训练轮数。默认读取已转换的
`src/assets/motions/g1/lafan1_npz/manifest.json`，使用全部 40 段 LAFAN 动作，
从头联合训练世界模型、Actor 和 Critic，无需先运行独立世界模型预训练。
若需先预热世界模型，在下方命令中增加 `--warmup-steps 64 --wm-only-iters 500`：
初始采集后前 500 轮只训练世界模型，第 501 轮开始联合训练。预热期间仍继续采集；
预热轮数包含在 `--iterations` 中，断点续训会保留进度。该选项默认 `0`。

CUDA 训练默认启用 `--compile --amp bf16`：核心张量计算使用 `torch.compile`，
仅 MLP 使用 BF16 autocast，物理计算保留 FP32；想象奖励与 Critic 目标值跨时间
批量计算。参考输入默认每帧 73 维、每窗口 730 维。初始轮需编译内核，应在编译
结束后比较稳定速度。CPU 使用普通执行及 FP32。`--no-compile` 和 `--amp none`
可分别关闭编译和混合精度。使用 `--reference-dim 2680` 可训练原版全身参考输入，
保留原有特征顺序、缩放和绝对参考高度；`--reference-dim 730` 为默认的紧凑输入。
Actor、Critic、真实采集和想象训练统一使用所选特征。对照旧版 FP32 基线时，可加
`--amp none --no-compile`，其他训练参数沿用原实验配置。
完整 checkpoint 续训自动识别参考维度，兼容缺少 schema 元数据的旧 2680 维断点；
旧断点缺少执行配置时沿用非编译 FP32。续训不能切换参考维度，更换架构需新建输出目录。

新训练默认 `--world-prediction state_delta`：世界模型预测完整单步状态增量，
不额外加入恒速外推。Actor 仍输出归一化关节位置指令。缺少此配置的旧 checkpoint
自动使用 `velocity_residual`，包括 preview 世界模型，保持原预测行为。
续训自动恢复原模式，不能中途切换。训练新参数化需新建输出目录；如需旧版对照，
指定 `--world-prediction velocity_residual`。输入统计、隐藏层激活和损失尺度沿用现有规则。

为这组世界模型改动做消融，新训练只需增加**一个开关**：

```bash
--world-model-normalization
```

同时启用动力学隐藏层 `Linear → LayerNorm → SiLU` 和独立的输入／输出 RunningNorm。
统计量仅由每条新采集的真实转移更新一次，包含初始预热；Replay 拟合批次和想象状态
不更新统计，世界模型拟合及策略 BPTT 期间保持冻结。输入标准化后进入网络，输出通过
`y * (std + eps) + mean` 恢复物理量，再更新状态。统计量使用 FP64，物理运算保留 FP32；
输入／输出 epsilon 分别为 `0.01`／`0.0001`，累计达到 2 亿真实样本后停止更新。
状态损失仍使用现有固定物理标度，便于比较这一组改动。

默认关闭，即 ELU 加固定的初始采集统计。两组实验应使用不同输出目录，其余设置保持
一致。完整断点、推理导出及播放保留所选结构与统计，续训自动恢复开关且不能中途切换；
旧 checkpoint 缺少此字段时视为关闭。离线世界模型入口也支持同一开关，但因数据固定，
只用训练 episode 拟合一次统计，不计入验证数据。

Actor／Critic 输入可分别启用 `RunningNorm`，默认均关闭：

```bash
--actor-input-normalization --critic-input-normalization
```

Actor 对每帧本体观测和参考输入分别归一化，10 帧观测历史共享同一套逐维统计；
Critic 对完整输入特征独立归一化。统计只从真实采集数据更新，想象反传、价值拟合、
评估和播放期间固定。统计随完整断点或轻量 Actor 权重保存，兼容 730／2680 维。
可只启用其中一个开关做对照实验；新实验请使用新输出目录。恢复会自动读取开关，
不能通过 `--resume` 给已有未归一化模型直接启用它们。未记录这些配置的旧断点按关闭恢复。

```bash
python scripts/train_supermimic.py \
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
完整规模的显存需求高于缩小配置，还需容纳 Replay 和 32 步反向传播计算图。

## 断点与导出

新训练的输出目录必须不存在。主要输出如下：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 训练配置、实际运行版本和数据/机器人签名 |
| `metrics.jsonl` | 采集、世界模型、Actor、Critic 与耗时指标 |
| `simulator.log` | 仿真器初始化和 kernel 输出 |
| `wandb_run.json`、`wandb/` | 可选 W&B run 标识与本地日志 |
| `last.pt` | 完整续训断点：三个网络、优化器、Replay、采样器和 RNG 状态 |
| `policy_best.pt`、`world_model_best.pt` | 配套的最佳 Actor 与世界模型：成功动作数优先，同分比较归一化回报 |
| `policy_last.pt`、`world_model_last.pt` | 最近保存的配套推理权重及元数据 |
| `policy_iteration_*.pt`、`world_model_iteration_*.pt` | 在 checkpoint 间隔、最终轮次及每次出现新最佳时保留的配套历史权重 |
| `best_checkpoint.json` | 选择规则、评估成绩、轮次及最佳模型的历史文件路径 |
| `iteration_*.pt` | 显式开启 `--full-checkpoint-interval N` 后保存的历史完整断点 |
| `evaluation*.json` | 各动作回报、成功数与失败原因 |

默认只更新一份完整的 `last.pt`，历史完整断点默认关闭，避免每次保存都复制数 GB
Replay，同时保留配套的轻量策略和世界模型历史。轻量策略不含 Replay、World Model、Critic 或优化器，可直接用于评估和
MuJoCo 可视化，不能用于 `--resume`。若需保留每 10000 轮的完整历史，添加
`--full-checkpoint-interval 10000`；该选项不会自动删除旧文件。
长期训练保留世界模型历史会占用较多磁盘，可增大 `--checkpoint-interval` 降低保存频率。

最佳模型按固定起点评估选择，初始化策略也参与比较。40/40 优先于之后回报更高的
39/40；成功数相同时比较归一化回报，完全同分保留较早模型。`best_checkpoint.json`
在两个文件均写完后更新，指向带轮次的历史文件；即使别名更新中断，也可据此确认配对。
例如使用 `--checkpoint-interval 1000` 会保留 `policy_iteration_010000.pt` 和
`world_model_iteration_010000.pt`，后续成绩回退不会覆盖这两个文件。轻量最佳／历史
文件不包含 Critic、优化器和 Replay，不能用于恢复训练。

已有完整断点无需重新训练即可导出。以下命令在项目根目录执行，源文件保持不变，
导出不需要动作数据或 GPU，也不会把 Replay 整体读入内存：

```bash
python scripts/export_supermimic_policy.py \
  --checkpoint outputs/supermimic/lafan_horizon_fixed/best.pt \
  --output outputs/supermimic/lafan_horizon_fixed/policy_best.pt
```

同一脚本也支持单独导出世界模型：

```bash
python scripts/export_supermimic_policy.py \
  --component world-model \
  --checkpoint outputs/supermimic/lafan_horizon_fixed/last.pt \
  --output outputs/supermimic/lafan_horizon_fixed/world_model_last.pt
```

不指定 `--component` 时仍导出 policy。世界模型文件使用 `supermimic_world_model`
格式，保留完整 `world` 权重（包括归一化参数、运动学数据和终止预测头）、网络配置、
机器人／物理签名、动作来源和轮次，不含 Actor、Critic、Replay、优化器或 RNG 状态，
不能用于续训。`policy_best.pt` 等轻量策略不含世界模型，不能从中恢复对应权重。
新训练会自动配套保存两者；旧训练需要同轮次权重时，应从同一个完整断点分别导出。不要把不同轮次的
`policy_best.pt` 和 `last.pt` 当作同一轮模型。导出文件可通过 play 的 `--world-model` 加载。

`last.pt` 和已有 `iteration_*.pt` 也可同样导出。评估和播放兼容完整断点及轻量策略，
两种格式保留相同 Actor 权重和数据／机器人／物理配置校验。续训时，会用已保留策略及
不晚于恢复轮次的评估报告，按新的成功数优先规则重选最佳；旧完整断点可用于补全组件。
若最佳历史策略缺少同轮世界模型，`best_checkpoint.json` 会记录 `world_model: null`，
终端也会提示，无法补回从未保存的权重。若存在其他轮次的世界模型最佳别名，会另存为
`world_model_previous_best*.pt`，避免错误配对；已有完整历史文件不会删除。
这只改变存储格式，不转换旧版参考特征或物理配置。
缺少 `reference_schema` 的旧断点会从 `actor["encoder.0.weight"]` 的输入维度推断：
730 对应 `root_joint_73_v1`，2680 对应 `full_body_268`；导出文件记录
`reference_schema_inferred: true`，命令行也会显示推断结果。评估和播放会为 2680 维
策略自动选择旧版完整身体特征，保持原权重与特征计算不变；未知输入维度会明确报错。
正在运行的训练进程需要从 `last.pt` 重启后才会采用新的保存方式。

## 训练日志与 W&B

终端默认使用紧凑表格，不再打印完整字典。表格按两次输出之间的区间统计：`Fail%`
为每条真实采集转移对应的物理失败事件频率，`Steps/s` 包含采集和优化时间，`GPU`
为 PyTorch 分配器记录的显存峰值（GiB）。初始化表格、kernel 信息和重复 seed 输出
写入 `simulator.log`，异常回溯仍在 stderr 显示。

可用 `--log-interval 50` 降低输出频率，`--verbose-sim` 显示仿真器详细输出，
`--log-format json` 恢复原终端 JSON 格式。`metrics.jsonl` 始终逐轮保存原始指标。
安装了 `rich` 时使用格式化表格，否则使用普通文本。

启用可选的 W&B 和 Rich 依赖，再登录一次：

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

论文规模命令也可直接追加相同的 W&B 参数。团队项目可指定
`--wandb-entity YOUR_TEAM`。终端会显示 run 链接，W&B 按 `world_model/`、`policy/`、
`critic/`、`collection/`、`gradients/`、`performance/`、`evaluation/` 分组绘图，
横轴为训练轮数。评估包括归一化回报、成功率、完成比例、跟踪误差，以及逐动作柱状图
和表格。采集指标只在实际采集的轮次记录，不重复计算沿用值。模型断点和 Replay
保存在本地输出目录。SDK 接口见 [W&B Run 文档](https://docs.wandb.ai/ref/python/experiments/run/)。

无网络服务器可先离线记录，之后同步：

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --iterations 1000 \
  --wandb --wandb-mode offline \
  --output outputs/supermimic/online_offline

wandb sync outputs/supermimic/online_offline/wandb/offline-run-*
```

续训时再次传入 `--wandb`。`wandb_run.json` 保存 run 标识和最后记录的轮数；在线
续训且未回退到更早断点时沿用同一 run。离线续训或断点回退会创建新的 run 分段，
通过 `parent_run_id` 关联，保留原有曲线。日志参数可以在续训时调整，模型和训练规模
参数仍须与断点一致。已经启动的训练进程需要从已保存断点重启后才会使用新的日志代码。

## 策略评估

加载在线 checkpoint，从每段动作首帧进行真实物理仿真，关闭域随机化和观测噪声：

```bash
python scripts/evaluate_supermimic.py \
  --device cuda:0 --checkpoint outputs/supermimic/online_training/policy_best.pt \
  --output outputs/supermimic/online_policy_evaluation.json \
  --csv outputs/supermimic/eval.csv
```

默认评估全部训练动作，直到动作完成或物理失败。归一化回报按完整动作的最大可达
回报计算，提前失败会降低得分。`--max-steps-per-clip` 仅用于调试，报告会标记未完整
评估的动作。`policy_best.pt` 可能仍是初始化策略，应以评估报告判断训练是否改善。

训练内评估和无窗口评估默认使用最多 64 个并行环境，40 个 LAFAN1 动作可一次并行运行。
训练时用 `--eval-num-envs N`，独立评估用 `--num-envs N` 限制并行数量；设为 `1`
可回到串行评估，动作数量超过环境数时分批执行。每个动作独立运行至完成或失败，
结束后即使环境自动重置，也不会继续累计该动作的奖励、误差或失败记录。
play、实时窗口及视频导出保持单环境。JSON 中的 `evaluation_num_envs` 记录实际环境数；
训练评估另记录 `evaluation_wall_seconds`，包含环境初始化耗时。

参照 [supermimic_vibed 的评估协议](https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/evaluate.py)，
终端输出 `Evaluation return <分数>`，例如 `Evaluation return 0.824`（仅为输出格式示例，
不是当前模型的实测成绩）。每段动作的分数为 `累计奖励 / (5 × 0.02 × (帧数 − 1))`，
再对全部动作取算术平均。报告同时包含逐动作分数、物理失败原因、完成率、
`normalized_return_framew`（按完整动作帧数加权）、`n_failed` 和 `failed`。
截断调试或意外超时会显示 `INCOMPLETE`，并将 `evaluation_complete` 设为 `false`；
完整评估中提前跌倒仍计为物理失败，分母始终使用完整动作长度。
CSV 记录 checkpoint、训练轮数、动作选择及分数，便于比较相同动作集上的模型。

## 播放与世界模型可视化

加载同一个 checkpoint，实时对照策略执行与参考动作：

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint outputs/supermimic/online_training/policy_best.pt \
  --clips walk1_subject1 run1_subject2 --loop
```

加入世界模型预测对照：

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/preview/policy_10000.pt \
  --world-model checkpoints/preview/world_model_10000.pt \
  --world-model-horizon 32 --clips walk1_subject1 run1_subject2 --loop
```

默认 `--world-model-mode future` 会在世界模型内根据预测观测运行策略，橙色机器人
显示**未来 32 步（0.64 秒）后的预测姿态**，并绘制橙色根部路径。
`--world-model-refresh 4` 表示每 4 个控制步从当前真实状态和观测历史重新预测，
其间沿缓存轨迹前进，保持固定的未来时间差；参考特征批量编码以减少开销。
若预览影响流畅度，可增大 refresh。接近动作末尾时提前量缩短，不预测到下一个动作。

`--world-model-mode compare` 保留原来的同时刻动力学误差对照，使用真实动作连续
预测并每隔 horizon 步同步；在此模式下设 `--world-model-horizon 1` 可检查单步误差。
`W` 显隐橙色预测，`G` 显隐蓝色参考。预测始终保留绝对位置和朝向。
未来模式会显示 `FUTURE +0.64s`，等真实仿真到达目标帧后才统计该预测的误差，
不会将未来姿态与当前姿态直接比较。JSON 的 `world_model_comparison` 记录模式和
逐动作误差；预测出现 NaN／Inf 时隐藏至下次刷新，不影响真实仿真。

世界模型会校验动作库、所选动作、机器人签名和物理配置，建议与策略使用同一轮权重。
该选项也支持 `--video` 及 `--viewer none` 的诊断运行，固定使用单环境，不改变真实
仿真回报。鼠标外力只进入真实物理；世界模型没有外力输入，因此拖拽时预测可能偏离。

实体机器人是策略在 MuJoCo Warp 物理仿真中的执行结果，蓝色半透明机器人是对应帧
的参考姿态。默认参考根部 XY 和朝向按 body tracking 奖励对齐；加 `--global-ghost`
查看绝对位移与朝向偏差。空格暂停，右方向键单步，`G` 切换参考机器人，关闭窗口退出。
`--speed 0.5` 可慢放，`--no-ghost` 可只看策略。`--loop` 反复播放所选动作，
每轮完整评估后写报告；中途关闭窗口不会保存该轮的不完整成绩。开启终止检测时，
窗口显示重置前的最后一帧。播放默认关闭高度／倾斜失败终止，跌倒后
继续运行直到当前动作结束，方便观察恢复过程；动作结束后仍正常切换下一段。
加 `--termination` 可恢复失败时重置。`evaluate_supermimic.py --viewer native` 也可
边评估边观看；不加 `--viewer` 的评估默认无窗口。

播放入口默认开启鼠标外力，交互方式与 `AMP_mjlab` 的原生查看器一致：先左键双击
实体机器人上的一个身体部位，再按住 `Ctrl` 拖拽。`Ctrl`＋右键拖拽施加平移外力，
加 `Shift` 可在水平面拖拽；`Ctrl`＋左键拖拽施加旋转力矩。不按 `Ctrl` 时仍是相机操作。
松开鼠标会在下一控制步清除拖拽外力；切换动作、关闭窗口也会清除。
暂停时不会推进物理，按右方向键可单步观察受力响应。操作依据
[MuJoCo 原生鼠标控制](https://mujoco.readthedocs.io/en/stable/programming/samples.html#mouse)。

`--no-perturbations` 可关闭鼠标外力。标准 `evaluate_supermimic.py` 默认关闭外力；
若要在评估窗口交互测试，使用 `--viewer native --perturbations`。实际施加过鼠标力的
轮次会显示 `INTERACTIVE`，JSON／CSV 中记录 `evaluation_mode: interactive` 和
`mouse_perturbation_steps`，方便与标准评估区分。关闭终止检测但未施力的轮次标为
`playback`；只有保留终止检测且未施力时才标为 `standard`。JSON／CSV 另存
`termination_enabled`；关闭终止后仍记录首次越过失败阈值的步数和原因，不会把跌倒
后播放到结尾算作跟踪成功。标准评估默认保留失败终止，显式 `--no-termination`
可用于诊断播放。

无桌面服务器可导出同一评估过程的视频（需要 `ffmpeg`）：

```bash
MUJOCO_GL=egl python scripts/evaluate_supermimic.py \
  --device cuda:0 --checkpoint outputs/supermimic/online_training/policy_best.pt \
  --clips walk1_subject1 --video outputs/supermimic/walk.mp4 \
  --output outputs/supermimic/walk_evaluation.json
```

视频顺序拼接所选动作；默认 960×540、50 FPS。实时窗口需要桌面显示环境，视频导出
不需要窗口。评估与播放共用 checkpoint、数据签名和物理参数校验，自动识别已知的
730／2680 维策略，包括缺少 `reference_schema` 的旧断点；不匹配的物理配置仍会拒绝加载。
播放必须指定 `--checkpoint` 或 `--policy`；评估未指定策略时运行零动作基线，并明确标注。

## 测试

```bash
SUPERMIMIC_TEST_DEVICE=cuda:0 \
python -m unittest discover -s tests -v
```

测试覆盖数据转换、动作库、Replay 边界、真实仿真、可微运动学、BPTT、损失归约、
在线更新、断点恢复与 checkpoint 推理。无 GPU 时可改为 `SUPERMIMIC_TEST_DEVICE=cpu`；
仅适用于 CUDA 的测试会跳过。

## Python API

```python
from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.online_training import OnlineTrainer
```
