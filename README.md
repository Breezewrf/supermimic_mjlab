# SuperMimic Mjlab

English | [简体中文](README_zh.md)

<p align="center">
  <img src="doc/assets/demo_v0.gif" alt="SuperMimic demo">
</p>

A reproduction of [SuperMimic](doc/supermimic.pdf) for multi-motion tracking on the
Unitree G1. The current policy collects simulator transitions, a differentiable world model learns from 8-step autoregressive
windows, and the Actor and Critic train through 32-step imagined rollouts.

The current dataset contains 40 G1-retargeted LAFAN clips, approximately 2.45 hours
of motion, at a control rate of 50 Hz.

The robot uses the vendored [RoboJuDo-Plus G1 assets](src/assets/robots/g1/README.md),
loading `g1_29dof_rev_1_0.xml` with collision meshes, shoulder cylinders, and four
5 mm radius spheres per foot. Training retains the mjlab PD control setup and
disables self-collisions.

## Environment

From the repository root, create the environment and install the project with uv:

```bash
uv sync
source .venv/bin/activate
```

See the [installation guide](doc/setup_en.md) for dependencies and GPU checks.

## Code structure

Task components are organized by responsibility, with separate packages for
configuration, data, task functions, models, and training:

```text
src/tasks/supermimic/
├── config/
│   ├── task_cfg.py           # G1 dimensions, motion paths, environment settings
│   └── rl_cfg.py             # OnlineConfig: workload, networks, optimizer settings
├── data/
│   ├── motion_library.py     # NPZ motions, reference frames, future windows
│   ├── replay_buffer.py      # Episode-aware replay and history reconstruction
│   └── world_model_data.py   # Offline splits, deduplication, ReplayCorpus
├── mdp/
│   ├── state.py              # State layout, quaternion math, observations, noise
│   ├── rewards.py            # Differentiable tracking rewards
│   ├── terminations.py       # Physical failure criteria
│   └── reset_sampler.py      # Return-based reset sampling
├── models/
│   ├── kinematics.py         # Differentiable G1 kinematics and robot metadata
│   ├── world_model.py        # Dynamics and task prediction heads
│   └── policy.py             # MotionActor, MotionCritic, reference features
├── rl/
│   ├── imagination.py        # BPTT, soft survival, GAE, Critic loss
│   ├── online_training.py    # Collection, model fitting, Actor-Critic updates
│   ├── training_logger.py    # Terminal tables and W&B tracking
│   ├── checkpoints.py        # Full training snapshots and actor-only exports
│   ├── world_model_training.py # Standalone model training and validation
│   └── evaluation.py         # Fixed-start, full-clip simulator evaluation
├── env.py                    # SuperMimicEnv simulator interface
└── env_cfg.py                # mjlab scene, PD actions, domain randomization
```

Package initializers remain lightweight. Regression tests live in `tests/`.
Scripts are organized by purpose and frequency of use:

```text
scripts/
├── train_supermimic.py            # Online joint training
├── evaluate_supermimic.py         # Checkpoint policy evaluation
├── play_supermimic.py             # Live MuJoCo policy/reference comparison
├── export_supermimic_policy.py    # Extract policy or world-model weights
├── data/
│   └── convert_lafan_to_npz.py     # LAFAN CSV-to-NPZ conversion
└── offline/
    ├── collect_supermimic.py      # Replay collection or kinematic replay checks
    └── train_world_model.py       # Standalone model training and validation
```

Use the root scripts for routine online training, evaluation, and playback.
The `offline/` scripts support diagnostics and offline experiments; online training
can start directly without running them.

## Prepare motion data

Raw CSV files are stored in `src/assets/motions/LAFAN1_Retargeting_Dataset/`.
If they have not been converted, run:

```bash
python scripts/data/convert_lafan_to_npz.py \
  --input-dir src/assets/motions/LAFAN1_Retargeting_Dataset \
  --output-dir src/assets/motions/g1/lafan1_npz \
  --input-fps 30 --output-fps 50
```

This writes one NPZ per clip and a `manifest.json`. Proceed to training after
conversion; if these files already exist, this step can be skipped. See the
[LAFAN conversion guide (Chinese)](doc/lafan_conversion_zh.md) for fields and coordinate conventions.

## Online training

`motion_anchor_ori_b` is an optional ablation, disabled by default; enable it with
`--motion-anchor-ori-b` in the training command.

### Reproduce the preview checkpoint (iteration 10000)

The current preview uses the following pair, both exported at **iteration 10000**
from `outputs/supermimic/lafan_2680_fp32/last.pt`:

| File | Contents |
| --- | --- |
| `checkpoints/preview/policy_10000.pt` | Actor weights for real-physics playback; `full_body_268` reference schema |
| `checkpoints/preview/world_model_10000.pt` | Matching learned world model for prediction overlays |

**Training is ongoing. The current preview is the iteration-10000 snapshot.**
Recorded evaluation on all 40 LAFAN motions:

| Metric | Iteration 10000 |
| --- | --- |
| Normalized return | 0.498 |
| Successful motions | 35/40 (87.5%) |
| Failed motions | 5/40 |
| Mean completion | 94.0% |
| Parallel evaluation environments | 40 |
| Evaluation time, including initialization | 125.5 s |

**Observed training GPU memory usage: approximately 30 GB** for the 4096-environment
FP32 configuration below. Actor and Critic RunningNorm were **disabled** for this
preview. The two files under `checkpoints/preview/` are inference exports; training
resume requires the full `last.pt`.

Original training configuration, run from the repository root:

```bash
python scripts/train_supermimic.py \
  --device cuda:0 \
  --reference-dim 2680 \
  --world-prediction velocity_residual \
  --amp none --no-compile \
  --no-actor-input-normalization --no-critic-input-normalization \
  --seed 0 \
  --num-envs 4096 --capacity 1000 \
  --warmup-steps 32 --wm-only-iters 0 \
  --collect-steps 5 --updates-per-collect 2 \
  --wm-batches 10 --wm-batch-size 2048 \
  --policy-batch-size 4096 --horizon 32 \
  --critic-epochs 5 --critic-minibatches 4 \
  --learning-rate 2e-4 \
  --world-dims 2048 2048 2048 \
  --actor-dims 2048 1024 512 256 \
  --critic-dims 256 128 \
  --iterations 32000 \
  --eval-interval 2000 --checkpoint-interval 2000 \
  --output outputs/supermimic/lafan_2680_fp32
```

To reproduce the preview training length, set `--iterations 10000`. Export both
components from the same iteration-10000 full checkpoint; `last.pt` advances if
training continues, so exporting the final 32000-iteration run would produce newer weights:

```bash
python scripts/export_supermimic_policy.py \
  --checkpoint outputs/supermimic/lafan_2680_fp32/last.pt \
  --output checkpoints/preview/policy_10000.pt

python scripts/export_supermimic_policy.py \
  --component world-model \
  --checkpoint outputs/supermimic/lafan_2680_fp32/last.pt \
  --output checkpoints/preview/world_model_10000.pt
```

Play selected motions in real physics:

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/preview/policy_10000.pt \
  --clips walk1_subject1 run1_subject2 --loop
```

Add the matching world model to preview the policy 32 control steps (0.64 s) ahead:

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/preview/policy_10000.pt \
  --world-model checkpoints/preview/world_model_10000.pt \
  --world-model-mode future --world-model-horizon 32 \
  --clips walk1_subject1 run1_subject2 --loop
```

Omit `--clips` to play all 40 motions. Playback defaults to allowing mouse forces
and continuing after height/tilt failures; add `--no-perturbations --termination`
for playback with failure resets. For standard headless evaluation of all motions:

```bash
python scripts/evaluate_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/preview/policy_10000.pt \
  --output outputs/supermimic/evaluation_policy_10000.json
```

Domain randomization, observation noise, and return-based reset sampling are enabled
by default. All motions are used unless `--clips` is specified; for debugging, use
`--clips walk1_subject1 run1_subject2`.

Robot self-collisions are disabled while full-body ground contacts remain enabled.
Training resets perturb the reference pose, root velocities and joint positions.
Foot friction and torso CoM are sampled once at startup; encoder bias is resampled
on each episode reset. Evaluation disables reset noise and domain randomization.
See the [environment configuration](doc/supermimic_foundation_zh.md) for ranges.
Training variants, optional RunningNorm, checkpoint formats and viewer controls are
in the [operational guide](doc/training_and_playback_en.md).

### Resume training

Network and batch settings are restored from the checkpoint. `--iterations` specifies
the total number of iterations, including those already completed:

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --resume outputs/supermimic/lafan_2680_fp32/last.pt \
  --iterations 32000
```

## Training logs and W&B

The terminal shows a compact table instead of raw dictionaries. Values are averaged
between reports; `Fail%` counts physical failure events per collected transition,
`Steps/s` includes collection and optimization time, and `GPU` is the PyTorch peak
allocation in GiB. Simulator initialization tables, kernel messages, and repeated
seed messages are saved to `simulator.log`. Exceptions remain visible on stderr.

Use `--log-interval 50` to print less often, `--verbose-sim` to show simulator
messages, or `--log-format json` for the previous terminal JSON format.
`metrics.jsonl` still stores every raw iteration, independently of terminal output.
Rich formatting is used when `rich` is installed; plain text is available otherwise.

Enable the optional W&B and Rich dependencies, then sign in once:

```bash
uv sync --extra logging
wandb login
```

Enable it for a new run:

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --iterations 1000 \
  --wandb --wandb-project supermimic --wandb-name g1-lafan \
  --output outputs/supermimic/online_wandb
```

When resuming training, pass `--wandb` again. Offline logging and run continuation
are covered in the [operational guide](doc/training_and_playback_en.md#training-logs-and-wb).

## Detailed guides

- [Installation](doc/setup_en.md)
- [Training variants, checkpoint formats, evaluation, viewer controls and tests](doc/training_and_playback_en.md) ([中文](doc/training_and_playback_zh.md))

Implementation guides (Chinese):

- [LAFAN data conversion](doc/lafan_conversion_zh.md)
- [Motion library, collection environment, and replay buffer](doc/supermimic_foundation_zh.md)
- [Standalone world-model training and validation](doc/supermimic_world_model_zh.md)
- [Online training and evaluation protocol](doc/supermimic_online_zh.md)

Standalone collection and model training support diagnostics and offline experiments.
Online training can start directly from scratch.
