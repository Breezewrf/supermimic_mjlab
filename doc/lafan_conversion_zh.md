# LAFAN G1 CSV 转 NPZ

先按[安装说明](setup_zh.md)使用 uv 安装并激活环境，再从仓库根目录运行：

```bash
python scripts/data/convert_lafan_to_npz.py
```

默认读取 `src/assets/motions/LAFAN1_Retargeting_Dataset/*.csv`，输出到
`src/assets/motions/g1/lafan1_npz/`，每条动作一个同名 NPZ，并生成本次转换的
`manifest.json`。全程使用 CPU 运动学，不需要 CUDA 或图形界面。

输入为无表头的 36 列 CSV，依次为世界坐标
根位置 xyz（米）、根四元数 xyzw、29 个 G1 关节角（弧度）。具体关节顺序见脚本的
`CSV_JOINT_NAMES`，转换时按名称映射至模型顺序。CSV 不含帧率信息，默认按 30 Hz
解释并重采样为 50 Hz；默认四元数约定是显式假设，不做自动猜测。

指定参数或只转换一条动作：

```bash
python scripts/data/convert_lafan_to_npz.py \
  --input-dir src/assets/motions/LAFAN1_Retargeting_Dataset \
  --output-dir src/assets/motions/g1/lafan1_npz \
  --input-fps 30 --output-fps 50 \
  --quaternion-order xyzw \
  --pattern 'walk1_subject1.csv'
```

重新生成已有文件需要 `--overwrite`。使用 `--pattern '**/*.csv'` 可递归匹配，输出
保留相对子目录。manifest 描述本次选择的文件；完整动作库转换时使用默认 pattern。

| 字段 | 形状 | 含义 |
| --- | --- | --- |
| `fps` | `(1,)` | 输出帧率 |
| `joint_pos` | `(T, 29)` | 模型顺序的关节角，rad |
| `joint_vel` | `(T, 29)` | 关节角速度，rad/s |
| `body_pos_w` | `(T, B, 3)` | 刚体 link 原点世界位置，m |
| `body_quat_w` | `(T, B, 4)` | 刚体世界姿态，**wxyz** |
| `body_lin_vel_w` | `(T, B, 3)` | link 原点世界线速度，m/s |
| `body_ang_vel_w` | `(T, B, 3)` | 世界角速度，rad/s |

运动数组均为 float32。`B` 是机器人全部刚体数量，不包含 MuJoCo world body，首项
为 pelvis；不是仅导出 tracking 使用的关键刚体。还保存 `joint_names`、`body_names`、
四元数顺序、源文件和 SHA256、输入帧率、模型来源和编译模型 SHA256、MuJoCo 版本。
所有字段均可使用 `np.load(path, allow_pickle=False)` 读取。

默认模型从 `mjlab.asset_zoo.robots.get_g1_robot_cfg()` 构建，与当前 G1 tracking
环境使用的工厂一致。也支持 `--model-xml /path/to/g1.xml`，此时只需要 NumPy 和
MuJoCo，但调用者需保证该 XML 的刚体顺序和运动学与训练机器人一致。

位置和关节角线性插值，四元数归一化、符号连续化后进行最短路径 SLERP。输出时间点
等间隔，不超出原动作时长；末帧恰好落在输出时间网格上时予以保留。速度用姿态流形上的
中心差分计算，边界使用单边差分，再通过 MuJoCo 计算刚体速度。线速度对应 link 原点，
与 mjlab 的 `body_link_lin_vel_w` 一致，不是刚体质心速度。

脚本校验输入列数、有限值、四元数、模型关节及输出数组，写入后重新加载校验，再原子
替换目标 NPZ。它不会进行动力学仿真、修改动作高度、裁剪关节角或筛除不可行片段。

SuperMimic 在线训练入口 `scripts/train_supermimic.py` 默认读取生成的 manifest，
加载全部动作；可通过 `--clips` 选择部分动作。

数值验证：

```bash
python -m unittest discover -s tests -p 'test_convert_lafan_to_npz.py' -v
```
