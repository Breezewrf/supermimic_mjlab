# SuperMimic 第一阶段：动作库、采集环境与回放池

这一阶段实现模型训练的数据基础：40 条 LAFAN 动作统一加载、参考状态重置、历史观测、
论文奖励与终止、物理转移采集、连续序列回放、固定起点评估。世界模型训练见
[第二阶段说明](supermimic_world_model_zh.md)；采集入口的 `--training` 表示启用
训练时的随机化和采样配置。

## 运行

先按[安装说明](setup_zh.md)使用 uv 安装并激活环境，以下命令均在仓库根目录运行。
世界模型训练入口会把实际运行版本写入配置和 checkpoint。

先用两条动作完成短采集，默认关闭随机化：

```bash
python scripts/offline/collect_supermimic.py \
  --device cuda:0 --num-envs 4 --steps 128 \
  --clips walk1_subject1 run1_subject2 \
  --output outputs/supermimic/debug
```

不指定 `--clips` 时使用 manifest 中的全部 40 条动作。输出目录必须尚不存在；生成：

- `replay.pt`：回放池及采样器状态，附带有序动作名称。
- `summary.json`：配置、转移数量、失败计数、窗口及历史形状。

开启论文随机化、传感器噪声和回报采样：

```bash
python scripts/offline/collect_supermimic.py \
  --device cuda:0 --num-envs 4 --steps 1024 --training \
  --output outputs/supermimic/randomized
```

采集动作为零动作加标准差 0.1 的独立高斯噪声，用于验证数据链路。零动作表示默认
关节设定值，不能代表已学会跟踪。`--action-noise 0` 可关闭动作噪声。

运动学回放校验会逐帧写入参考状态，并检查真实模型的 link 位置与 NPZ 是否一致：

```bash
python scripts/offline/collect_supermimic.py \
  --mode replay --device cuda:0 --steps 50 \
  --output outputs/supermimic/replay_check
```

这是无积分的运动学检查，不是已训练策略的物理跟踪，也不打开可视化窗口。

固定起点评估，默认使用零动作控制器：

```bash
python scripts/evaluate_supermimic.py \
  --device cuda:0 --clips walk1_subject1 run1_subject2 \
  --output outputs/supermimic/evaluation.json
```

调试可加 `--max-steps-per-clip 20`，结果会标记 `evaluation_limited`。完整评估关闭
随机化和噪声，从每条动作首帧播放到末帧或物理失败；奖励除以完整动作最大可达回报，
因此提前结束会降低归一化回报。报告同时包含完成比例、成功与否、失败原因和平均误差。

后续策略通过 `--policy your_module:your_function` 接入，函数约定为：

```python
def your_function(history, future_reference):
    # history: (N, 10, 93)，从旧到新。
    # future_reference: 字典，含 joint_pos、joint_vel、body_pos_w、
    # body_quat_w、body_lin_vel_w、body_ang_vel_w；时间维长度为 10。
    # 返回设备一致的 (N, 29) 归一化动作。
    return history.new_zeros((history.shape[0], 29))
```

回调接收到的参考窗口包含当前参考帧与其后 9 帧，末尾重复末帧。动作执行后的状态
与下一参考帧计算奖励。已实现的 Encoder 展开与在线训练见
[在线 Actor–Critic 说明](supermimic_online_zh.md)。

## 状态与数据约定

`SuperMimicEnv` 继承 mjlab 的环境，复用机器人、动作执行和随机化管理器，自行管理
参考时间、历史观测及任务信号。内置 MuJoCo 传感器不参与计算；本体观测直接从刷新后的
物理状态重建，并显式加入噪声，便于后续可微模型复用。

`state.py` 定义动力学状态，形状 `(N, 71)`：

| 范围 | 内容 |
| --- | --- |
| `0:3` | 根节点世界位置减环境原点，m |
| `3:7` | 根节点 wxyz 四元数 |
| `7:36` | 模型关节顺序的关节角，rad |
| `36:39` | 根节点世界线速度，m/s |
| `39:42` | 根节点世界角速度，rad/s |
| `42:71` | 关节速度，rad/s |

上下文形状 `(N, 33)`，依次为实际脚底摩擦系数、29 维编码器偏差、躯干质心的三维
偏移。脚底摩擦 `U[0.3,1.2]` 和躯干 CoM 偏移 `U[±(0.025,0.05,0.05)] m`
在环境启动时逐环境采样，之后跨 episode 固定；仅编码器偏差 `U[±0.01] rad`
每个 episode reset 时重采样。上下文始终读取实际应用的模型参数。关闭随机化时
摩擦为 1.0，编码器偏差和 CoM 偏移均为零。
Actor 的单帧观测为 `[q-q0+bias, qdot, root_omega_body, gravity_body, previous_action]`，
共 93 维；reset 后上一动作清零，10 帧历史均填为该 episode 的初始观测。

仿真关闭机器人自碰撞，但保留所有身体碰撞体与地面的接触。训练配置默认开启
`reset_noise`，参考状态上的均匀扰动半宽为：

| 状态 | x / y / z（关节为逐关节） |
| --- | --- |
| 根位置 | 0.05 / 0.05 / 0.01 m |
| 世界系旋转向量 | 0.1 / 0.1 / 0.2 rad，左乘参考旋转 |
| 根世界线速度 | 0.5 / 0.5 / 0.2 m/s |
| 根世界角速度 | 0.52 / 0.52 / 0.78 rad/s |
| 关节位置 | 0.1 rad，随后裁剪至硬限位 |
| 关节速度 | 保持参考值，不额外扰动 |

这些范围与参考仓库 `supermimic_vibed` 对齐，属于实现选择，不宣称是论文公开的
逐字段设置。`FoundationConfig.training(reset_noise=False)` 可单独关闭初态扰动。
普通 `FoundationConfig()` 用于确定性采集/评估，不扰动初态，也不启用域随机化或
传感器噪声；完整评估入口拒绝开启初态扰动的配置。

动作和状态中的关节顺序一致。动作噪声由采集端加入，Replay 中保存实际送往 PD 映射
的动作；如为环境配置 `action_clip`，保存限幅后的值。Body 字段使用当前 tracking 的
14 个关键刚体，第一项为 pelvis。坐标均减去环境原点，但保留参考轨迹的全局位移。

奖励使用 BeyondMimic 风格的 heading 对齐 body 目标，并单独跟踪绝对 pelvis 位姿；
不使用自碰撞惩罚，关节限位权重为 −2，所有项乘 0.02 秒。物理失败条件为根高度差
大于 0.25 m，或根与参考重力方向 z 分量差大于 0.8。

物理子步结束后先执行 `forward()`，再读取状态、计算奖励和结束原因、复制转移，最后
执行 reset。`terminated`、`time_out`、`motion_end` 分开保存，可同时成立；其中只有
`terminated` 是世界模型的物理失败标签。重置后的状态不充当上一条转移的下一状态。

## 核心 API

模块位于 `src/tasks/supermimic/`，按职责拆分：

- `config/`：环境运行参数与在线训练配置。
- `data/`：动作库、ReplayBuffer 和离线世界模型数据。
- `mdp/`：状态与观测、跟踪奖励、物理失败、重置采样。
- `models/`：可微运动学、世界模型、Actor 和 Critic。
- `rl/`：想象、世界模型拟合、在线更新和评估。

完整文件结构见 [README](../README_zh.md#代码结构)。主要 API：

- `MotionLibrary.get_frame(motion_id, frame_id)`：当前帧；按名称映射关节与刚体。
- `MotionLibrary.get_future(..., horizon=10)`：当前帧起的窗口，在片段边界填充末帧。
- `SuperMimicEnv.reset(options={"motion_id": ..., "frame_id": ...})`：指定重置点。
- `SuperMimicEnv.step(action)`：返回策略输入、奖励、失败、截断和 extras。
- `extras["transition"]`：完整的 reset 前转移，张量具有独立存储。
- `ReplayBuffer.append(transition)`：每次插入所有环境的一步转移，默认保留每环境 1000 步。
- `sample_windows(batch_size, length=8)`：返回 `(batch, length, ...)` 张量和布尔 `mask`。
- `sample_starts(batch_size, history_length=10)`：返回起始状态、上下文、动作和观测历史。

窗口必须有完整后续数据或明确的结束边界；结束步自身有效，之后的值置零且 mask 为
false。窗口不会跨越 episode、动作边界或环形覆盖边界。起点历史通过逐帧观测重建，
episode 初始部分重复初始观测；已被覆盖、无法恢复的历史不会被采样。

读取导出的回放池：

```python
import torch
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer

saved = torch.load("outputs/supermimic/debug/replay.pt", weights_only=True)
buffer = ReplayBuffer.from_state_dict(saved["replay"], device="cuda:0")
windows = buffer.sample_windows(32, length=8)
starts = buffer.sample_starts(32, history_length=10)
```

回放导出是训练数据快照，包含采样器 RNG，但不是仿真环境的完整断点；当前没有恢复
物理仿真至同一步的入口。

回放数据可在 CPU 与 CUDA 之间加载；跨后端时，回放采样器以保存的 seed 重新开始，
旧快照未保存 seed 时使用 0。CPU 与 CUDA 的 RNG 状态格式不同，因此不承诺跨后端
延续同一随机序列。世界模型使用独立的 CPU 采样器，其状态随训练 checkpoint 保存。

训练重置按 1 秒分桶：50% 均匀、40% 当前低回报 softmax、10% 历史困难分布。
每个已结束的 episode 反向计算折扣回报，按桶取平均，以 EMA 0.1 更新；历史分布使用
历次 softmax 的运行平均。这些更新细节属于本项目的明确复现选择。

直接导入 SuperMimic 的数据模块不会触发仿真器初始化。独立采集入口位于
`scripts/offline/collect_supermimic.py`；常用的在线训练与评估入口保留在 `scripts/` 根目录。

## 验证

数据和数值测试：

```bash
python -m unittest discover -s tests -p 'test_supermimic_foundation.py' -v
```

CUDA 集成测试（没有 CUDA 时明确跳过）：

```bash
python -m unittest discover -s tests -p 'test_supermimic_integration.py' -v
```

也可以用 CPU 执行同一套真实 MJWarp 仿真测试：

```bash
SUPERMIMIC_TEST_DEVICE=cpu python -m unittest discover -s tests -v
```

测试涵盖全部 40 条动作、名字映射、未来窗口、观测噪声和梯度、回放覆盖与 padding、
三种结束边界、reset 前状态捕获、上下文与模型一致、部分环境独立重置及 NPZ 运动学。

第二阶段已在此接口之上加入可微运动学和 8 步世界模型拟合，入口与评估约定见
[世界模型说明](supermimic_world_model_zh.md)。32 步 BPTT Actor–Critic 在线闭环已实现，
见[在线训练说明](supermimic_online_zh.md)。
