# 历史实现验证记录

本文保留早期本地实验的配置、产物路径和结果，用于追溯实现过程。产物不随仓库分发，这些结果不构成安装要求，也不代表当前代码重新训练后的性能。

旧离线数据集包含 40 段动作、32,000 条转移、825 个 episode；旧划分为 663 个训练 episode 和 162 个验证 episode（25,866 / 6,134 条转移）。

旧 32 步验证未覆盖 dance1_subject2、fallAndGetUp1_subject1、jumps1_subject2、run2_subject4。

## 世界模型验证

最初在 CPU 上执行全量采集、默认 2048×3 网络的 100 次更新、独立评估和真实
MJWarp 测试，共 34 项中 33 项通过，1 项 CUDA 对照当时因设备不可见而跳过。

随后在可访问 GPU 的执行环境中确认 RTX 4090 Laptop（16 GiB）可用。完整测试
34 项全部通过，没有跳过，包括 CUDA MJWarp 集成测试和 CPU/CUDA 预测、梯度
对照。默认 2048×3、batch 256 的 GPU 训练已从断点续训到第 100 次更新；

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
为 false。现已接入论文的[在线训练闭环](../supermimic_online_zh.md)，让数据分布随当前策略更新；这些离线
结果不能据此推断必须先把独立世界模型训练达标，才能开始策略训练。

## 在线训练验证


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
