# SuperMimic Mjlab

[English](README.md) | 简体中文

基于 mjlab、MuJoCo 和 MuJoCo Warp 的 SuperMimic 复现，面向 Unitree G1 的多动作跟踪。
方法参考 [SuperMimic 论文](doc/supermimic.pdf)：当前策略采集真实仿真数据，世界模型
用 8 步自回归窗口学习动力学，再通过 32 步可微想象联合训练 Actor 和 Critic。

当前使用 G1 重定向的 LAFAN 数据集，共 40 段动作、约 2.45 小时，控制频率为 50 Hz。
已实现在线训练、断点恢复和固定起点真实仿真评估；策略跟踪效果需通过完整动作评估确认。

机器人使用本地复制的 [RoboJuDo-Plus G1 资源](src/assets/robots/g1/README.md)，
加载 `g1_29dof_rev_1_0.xml`，碰撞体包含网格、肩部圆柱体及每脚 4 个半径 5 mm 的球体。
训练沿用 mjlab 的 PD 控制配置，关闭自碰撞，保留全身触地；动作转换也使用同一模型。
原 mjlab 模型的 checkpoint 与新模型指纹不匹配，需要重新开始训练。

## 运行环境

在仓库根目录，使用 uv 自动创建环境并安装项目：

```bash
uv sync
source .venv/bin/activate
```

`uv sync` 按 `.python-version` 使用 Python 3.11，并以 editable 模式安装项目；
`uv.lock` 记录解析后的依赖版本。每次打开新终端后先激活环境，
下文统一使用该环境中的 `python` 和 `wandb`。
核心依赖包括支持 CUDA 的 PyTorch、mjlab、MuJoCo、MuJoCo Warp 和 Warp，
版本声明保留在 [setup.py](setup.py)，由 [pyproject.toml](pyproject.toml) 引用；
完整步骤见[安装说明](doc/setup_zh.md)。
检查当前 Python 是否能使用 GPU：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

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

日常在线训练、评估和可视化使用根目录的脚本。`offline/` 中的脚本用于诊断和离线实验，
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
python scripts/data/convert_lafan_to_npz.py \
  --input-dir src/assets/motions/LAFAN1_Retargeting_Dataset \
  --output-dir src/assets/motions/g1/lafan1_npz \
  --input-fps 30 --output-fps 50
```

输出为每段动作的 NPZ 和 `manifest.json`。转换完成后即可训练；已有文件时可跳过转换。
转换字段与坐标约定见 [LAFAN 数据转换说明](doc/lafan_conversion_zh.md)。

## 在线训练

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
episode 重采样；评估关闭这些扰动。范围与生命周期见[环境说明](doc/supermimic_foundation_zh.md)。
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

### 断点恢复

恢复时从 checkpoint 读取原有网络与 batch 参数；`--iterations` 是包含已完成轮数的总轮数：

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --resume outputs/supermimic/online_training/last.pt \
  --iterations 5000
```

新训练的输出目录必须不存在。主要输出如下：

| 文件 | 内容 |
| --- | --- |
| `config.json` | 训练配置、实际运行版本和数据/机器人签名 |
| `metrics.jsonl` | 采集、世界模型、Actor、Critic 与耗时指标 |
| `simulator.log` | 仿真器初始化和 kernel 输出 |
| `wandb_run.json`、`wandb/` | 可选 W&B run 标识与本地日志 |
| `last.pt` | 完整续训断点：三个网络、优化器、Replay、采样器和 RNG 状态 |
| `policy_best.pt` | 真实仿真评估回报最高的轻量 Actor 权重及推理元数据 |
| `policy_last.pt` | 最近保存的轻量 Actor 权重及推理元数据 |
| `policy_iteration_*.pt` | 按 `--checkpoint-interval` 保存的轻量策略文件 |
| `iteration_*.pt` | 显式开启 `--full-checkpoint-interval N` 后保存的历史完整断点 |
| `evaluation*.json` | 各动作回报、成功数与失败原因 |

默认只更新一份完整的 `last.pt`，历史完整断点默认关闭，避免每次保存都复制数 GB
Replay。轻量策略不含 Replay、World Model、Critic 或优化器，可直接用于评估和
MuJoCo 可视化，不能用于 `--resume`。若需保留每 10000 轮的完整历史，添加
`--full-checkpoint-interval 10000`；该选项不会自动删除旧文件。

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
需要比较同一轮策略与世界模型时，应从同一个完整断点分别导出；不要把不同轮次的
`policy_best.pt` 和 `last.pt` 当作同一轮模型。导出文件可通过 play 的 `--world-model` 加载。

`last.pt` 和已有 `iteration_*.pt` 也可同样导出。评估和播放兼容完整断点及轻量策略，
两种格式保留相同 Actor 权重和数据／机器人／物理配置校验。续训旧目录时，若存在
旧 `best.pt` 且尚无 `policy_best.pt`，会自动导出最佳策略；已有完整历史文件不会删除。
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

### MuJoCo 实时动作验证

加载同一个 checkpoint，实时对照策略执行与参考动作：

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint outputs/supermimic/online_training/policy_best.pt \
  --clips walk1_subject1 run1_subject2 --loop
```

加入世界模型预测对照：

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint outputs/policy_best.pt \
  --world-model outputs/world_model_7000.pt \
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

## 详细文档

- [LAFAN 数据转换](doc/lafan_conversion_zh.md)
- [动作库、采集环境与回放池](doc/supermimic_foundation_zh.md)
- [独立世界模型训练与验证](doc/supermimic_world_model_zh.md)
- [在线训练与评估协议](doc/supermimic_online_zh.md)

独立采集与世界模型训练入口用于诊断和离线实验，在线训练可直接从头运行。
