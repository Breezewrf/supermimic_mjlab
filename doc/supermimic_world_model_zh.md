# SuperMimic 第二阶段：可微世界模型

本阶段实现 G1 的可微运动学、8 步自回归动力学训练、物理失败预测，以及固定
1/8/32 步验证。这是工程实现和独立调试的一个阶段。Actor、Critic 和策略 BPTT 的
在线入口已实现，见[在线训练说明](supermimic_online_zh.md)。论文 §3.1 与 Algorithm A.1 的完整算法在同一个在线循环内交替执行：当前策略
采集真实转移、拟合世界模型、通过固定参数的世界模型更新策略并拟合 Critic。
世界模型和策略在训练过程中持续共同改进，论文没有要求先完成独立的世界模型预训练。

## 直接运行

已有全量采集快照 `outputs/supermimic/world_model_data/replay.pt`，包含全部 40 段动作、
32,000 条转移和 825 个 episode，启用了域随机化、观测噪声和回报采样。
可直接在 GPU 上训练：

```bash
taskset -c 0-7,10-31 .venv/bin/python scripts/offline/train_world_model.py \
  --replay outputs/supermimic/world_model_data/replay.pt \
  --device cuda:0 --steps 5000 --batch-size 256 \
  --output outputs/supermimic/world_model_cuda
```

需要重新采集时，在仓库根目录运行，输出目录必须尚不存在：

```bash
.venv/bin/python scripts/offline/collect_supermimic.py \
  --device cuda:0 --num-envs 32 --steps 1000 --capacity 1000 \
  --training --output outputs/supermimic/world_model_data_new
```

采集器使用默认关节设定值加标准差 0.1 的动作噪声。该数据适合检验动力学学习，
动作分布仍局限于这个控制器；后续策略更新需要增加新策略产生的真实转移。

原来的两段动作 CUDA 快照也可用于快速检查：

```bash
taskset -c 0-7,10-31 .venv/bin/python scripts/offline/train_world_model.py \
  --replay outputs/supermimic/cuda_smok/replay.pt \
  --smoke --device cuda:0 --hidden-dims 64 64 64 \
  --steps 20 --batch-size 16 --validation-windows 16 \
  --eval-interval 10 --output outputs/supermimic/world_model_cuda_smoke
```

`--smoke` 允许动作子集，但报告始终设置 `candidate_for_bptt=false`。
调试 CPU 可使用 `--device cpu --cpu-threads 4`。按当前运行约定，训练和推理命令使用
`taskset -c 0-7,10-31`，普通检查命令不加。

## 模型与损失

物理状态保持第一阶段的 71 维，四元数采用 wxyz。动力学输入为 68 维状态特征、
29 维动作及 33 维上下文，共 130 维：

| 状态特征 | 维数 |
| --- | --- |
| 根高度 | 1 |
| 机体坐标中的重力方向 | 3 |
| 关节角 | 29 |
| 机体坐标中的根线速度、角速度 | 6 |
| 关节速度 | 29 |

不输入根的绝对水平位置或 yaw，使预测对全局水平平移和 yaw 旋转保持等变。
上下文为脚底摩擦系数、29 维编码器偏差和三维躯干质心偏移，episode 内保持不变。

动力学 MLP 默认三层 2048 维隐藏层、ELU，输出 70 维残差：位置、旋转向量、关节角、
线速度、角速度、关节速度，分组为 `(3, 3, 29, 3, 3, 29)`。
位置、旋转和速度残差使用当前机体坐标；关节量直接相加。在常速度积分基础上施加
残差，旋转通过指数映射更新，再归一化四元数。输出层零初始化时恰好等于常速度基线。

输入均值和标准差只从训练 episode 拟合。输出残差按训练集标准差缩放，不减均值；
前 35 维缩放下限为 `1e-3`，后 35 维为 `1e-2`。归一化统计随模型保存。

`G1Kinematics` 从编译后的 G1 提取父子树、关节轴、锚点和参考关节角，使用纯 Torch
计算 14 个跟踪刚体的位置、姿态及速度。计算保留对根状态和关节量的梯度，支持非零
关节锚点。奖励和无随机噪声观测复用第一阶段的纯张量函数。失败预测头使用归一化
根高度误差和倾斜误差，网络为 `2→64→64→1`，输出 logit。

每条训练序列只在起始时读取一次真实状态，随后连续预测 8 步并使用日志中的动作，
中途不注入真实状态，也不截断梯度。损失包含四个等权项：

1. 状态误差：六组分别归一化后取组内 MSE，再对组取平均；尺度依次为
   `(0.3, 0.4, 0.3, 1, 3.14, 5)`，旋转使用相对四元数的旋转向量误差。
2. 观测误差：关节角、关节速度、根角速度和重力四组，尺度为 `(0.3, 5, 3.14, 1)`。
   上一动作可精确复制，不计入这项损失。
3. 奖励误差：奖励除以 `0.1` 后的 MSE。
4. 失败误差：logit 与物理 `terminated` 标签的 BCE；`time_out`、`motion_end` 不作为失败。

结束步本身参与训练，之后 padding 的步骤通过 mask 排除。每项损失在时间维求和，
再对 batch 中的窗口取平均，遵循论文式（1）；padding 不计入平均分母。
默认优化器 Adam、学习率
`2e-4`、batch 256、梯度范数上限 1、FP32，并允许 TF32。以上残差参数化、归一化与
损失分组是本项目的明确实现选择。

## 数据划分与验证

正式模式使用 manifest 中全部动作，每段至少需要两个独立 episode。按动作分层，
以 `(数据源, 环境编号, episode编号)` 为组进行约 80/20 划分。同一 episode 的重叠窗口
不会同时进入训练和验证。多个快照可通过 `--replay a.pt b.pt` 合并，动作 ID 按名字
映射到统一动作库；相同文件哈希的重复输入只加载一次。

当前数据划分为 663 个训练 episode、162 个验证 episode，对应 25,866 和 6,134 条
转移。断点续训固定数据文件内容、manifest、动作顺序、episode 划分和验证窗口。
源文件可以移动路径，但内容哈希必须保持一致。

每个时域固定抽取最多 256 个验证起点，并优先包含所有有合格窗口的动作。1、8、32
步均只使用完整窗口，不用 padding 填满短 episode。验证从真实起点连续预测，报告：

- 终点六组物理 RMSE、归一化状态 MSE、14 个刚体的平均位置误差（MPJPE，m）。
- 预测过程中观测 RMSE、奖励 MAE。
- 物理失败 BCE、Brier、precision、recall 和正例比例；分母为零的指标记录为 null。
- 每段动作的同类指标，以及缺少合格窗口的动作名单。

同时运行常线速度、角速度和关节速度积分基线；其失败概率固定为训练转移的物理失败
比例。两者使用完全相同的起点、动作和标签。

当前独立评估脚本的 `qualification` 检查正式模式、1/8 步状态 MSE 优于基线、32 步所有预测有限且四元数
范数误差小于 `1e-4`，并且三个时域均覆盖全部动作。这是本项目额外设置的离线诊断
指标，不是论文的算法要求，也不应作为接入
在线策略训练的硬性门槛。仍需结合逐动作位置误差和失败指标判断模型质量。

当前 32 步验证缺少 `dance1_subject2`、`fallAndGetUp1_subject1`、`jumps1_subject2`、
`run2_subject4`，因此质量标记为 false。可用不同 seed 追加采集并合并快照，再重新
训练和划分，以补充离线评估。论文中的 32 步是模型想象 rollout，策略更新不依赖
每段动作具有 32 步真实验证窗口。仅把 `--steps` 增大而仍保留 `--capacity 1000`，会覆盖前面的采集；需要
保留更多数据时同步增大 capacity。

## 输出、评估和续训

默认每 100 次更新验证一次，每 500 次保存编号 checkpoint：

| 文件 | 内容 |
| --- | --- |
| `config.json`、`split.json` | 运行参数、实际依赖版本、数据哈希和 episode 划分 |
| `metrics.jsonl` | 每次更新的分项损失、梯度范数，及定期验证指标 |
| `validation.json` | 最近一次验证报告 |
| `best.pt` | 8 步验证状态 MSE 最低的模型，包含初始常速度模型候选 |
| `last.pt` | 最近保存的训练断点；正常结束时保存最终更新 |
| `step_000500.pt` 等 | 每隔指定更新次数保存的断点 |

checkpoint 包含模型、优化器、归一化、机器人运动学元数据、编译模型签名、实际依赖
版本、数据与 manifest 哈希、划分、固定验证起点和训练 RNG。模型签名也纳入质量、
惯量、阻尼、执行器及碰撞参数。已有采集快照未携带模型签名，因此训练时仍须使用
与采集时一致的 G1 和任务配置。

独立评估会恢复原来的验证窗口，并写入一个新目录：

```bash
taskset -c 0-7,10-31 .venv/bin/python scripts/offline/train_world_model.py --mode evaluate \
  --replay outputs/supermimic/world_model_data/replay.pt \
  --checkpoint outputs/supermimic/world_model_cuda/best.pt \
  --device cuda:0 --output outputs/supermimic/world_model_cuda_eval
```

续训必须保持原 batch size、数据和 `--smoke` 设置。`--steps` 表示包括已完成更新的
总更新次数；默认 batch size 为 256：

```bash
taskset -c 0-7,10-31 .venv/bin/python scripts/offline/train_world_model.py \
  --replay outputs/supermimic/world_model_data/replay.pt \
  --resume outputs/supermimic/world_model_cuda/last.pt \
  --device cuda:0 --steps 10000 --batch-size 256 \
  --output outputs/supermimic/world_model_cuda
```

已在 CPU 上验证续训与连续训练逐参数一致；跨设备加载可以运行，但不承诺不同硬件
和数值后端之间逐位一致。

纯 Torch 推理可直接使用 checkpoint 中的运动学元数据，无需重新编译 MuJoCo：

```python
import torch
from src.tasks.supermimic.models.world_model import WorldModel

saved = torch.load("outputs/supermimic/world_model_cuda/best.pt",
                   map_location="cpu", weights_only=True)
model = WorldModel(saved["robot_metadata"], tuple(saved["hidden_dims"]))
model.load_state_dict(saved["model"])
model = model.to("cuda:0").eval()
# next_state = model(state, action, context)
# 不使用 no_grad 时，可对动作反向传播。
```

## 本次实际验证

最初在 CPU 上执行全量采集、默认 2048×3 网络的 100 次更新、独立评估和真实
MJWarp 测试，共 34 项中 33 项通过，1 项 CUDA 对照当时因设备不可见而跳过。

随后在可访问 GPU 的执行环境中确认 RTX 4090 Laptop（16 GiB）可用。完整测试
34 项全部通过，没有跳过，包括 CUDA MJWarp 集成测试和 CPU/CUDA 预测、梯度
对照。默认 2048×3、batch 256 的 GPU 训练已从断点续训到第 100 次更新；使用
`taskset -c 0-7,10-31` 的普通模式全新启动也成功完成 30 次更新。

GPU 训练结果位于 `outputs/supermimic/world_model_gpu_check/`，全新启动复测结果
位于 `outputs/supermimic/world_model_gpu_fresh_check/`。第一次 GPU 短训练曾出现
CUDA/CPU stream 不匹配错误；异常定位重跑、普通模式续训和全新启动复测均通过，
目前未复现，原因尚未确定，不能据此认定 CPU affinity 是修复原因。

独立 GPU 评估位于 `outputs/supermimic/world_model_gpu_eval/validation.json`，加载
第 40 次更新的最佳 checkpoint；1/8/32 步预测均有限，8 步归一化状态 MSE 为
`0.03036`，对应常速度基线为 `0.12961`。

短训练及评估保存在 `outputs/supermimic/world_model_formal_check/` 和
`outputs/supermimic/world_model_formal_eval/`，前者的最佳 checkpoint 来自第 50 次更新。
这些是实现验证产物，训练次数尚不足。尽管归一化状态 MSE 优于常速度基线，短训练的
刚体位置误差和失败 BCE 仍有落后项，且 32 步离线验证覆盖不全，因此脚本的诊断标记
为 false。现已接入论文的[在线训练闭环](supermimic_online_zh.md)，让数据分布随当前策略更新；这些离线
结果不能据此推断必须先把独立世界模型训练达标，才能开始策略训练。

复查命令：

```bash
MPLCONFIGDIR=/tmp/supermimic-mpl XDG_CACHE_HOME=/tmp/supermimic-cache \
SUPERMIMIC_TEST_DEVICE=cpu \
  .venv/bin/python -m unittest discover -s tests -v
```

GPU 终端中可去掉 `SUPERMIMIC_TEST_DEVICE=cpu`，同时运行 CUDA 环境集成测试及
CPU/CUDA 世界模型对照。测试覆盖非零关节锚点、MuJoCo 位姿和速度、有限差分梯度、
坐标等变性、动作梯度、无教师强制的多步反传、训练集归一化、跨设备回放加载、动作
名映射、划分隔离、padding、独立评估、错误数据拒绝和断点续训一致性。
