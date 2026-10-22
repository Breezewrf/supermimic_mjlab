# SuperMimic Mjlab

English | [简体中文](README_zh.md)

A reproduction of [SuperMimic](doc/supermimic.pdf) for multi-motion tracking on the
Unitree G1, built with mjlab, MuJoCo, and MuJoCo Warp. The current policy collects
simulator transitions, a differentiable world model learns from 8-step autoregressive
windows, and the Actor and Critic train through 32-step imagined rollouts.

The current dataset contains 40 G1-retargeted LAFAN clips, approximately 2.45 hours
of motion, at a control rate of 50 Hz. Online training, checkpoint resume, and
fixed-start simulator evaluation are implemented. Tracking quality must be assessed
with complete motion evaluations.

The robot uses the vendored [RoboJuDo-Plus G1 assets](src/assets/robots/g1/README.md),
loading `g1_29dof_rev_1_0.xml` with collision meshes, shoulder cylinders, and four
5 mm radius spheres per foot. Training retains the mjlab PD control setup and
disables self-collisions while allowing full-body ground contacts. Motion conversion
uses the same robot. Checkpoints from the previous mjlab asset have a different
model fingerprint and require a new training run.

## Environment

From the repository root, create the environment and install the project with uv:

```bash
uv sync
source .venv/bin/activate
```

`uv sync` uses Python 3.11 from `.python-version` and installs the project in editable
mode. `uv.lock` records resolved dependency versions. Activate the environment in
each new terminal. All commands below use `python` and
`wandb` from the activated environment. Core dependencies include CUDA-enabled
PyTorch, mjlab, MuJoCo, MuJoCo Warp, and Warp; version declarations remain in
[setup.py](setup.py), referenced by [pyproject.toml](pyproject.toml). See the [installation guide](doc/setup_en.md) for details.
Check GPU availability in the selected Python environment:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

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
├── data/
│   └── convert_lafan_to_npz.py     # LAFAN CSV-to-NPZ conversion
└── offline/
    ├── collect_supermimic.py      # Replay collection or kinematic replay checks
    └── train_world_model.py       # Standalone model training and validation
```

Use the two scripts at the root for routine online training and evaluation.
The `offline/` scripts support diagnostics and offline experiments; online training
can start directly without running them. Example imports:

```python
from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.online_training import OnlineTrainer
```

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

### Reduced training workload

Use the default networks with 64 environments, 10 batches of 256 world-model windows,
and 64 imagined trajectories per update:

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --iterations 1000 \
  --output outputs/supermimic/online_training
```

Domain randomization, observation noise, and return-based reset sampling are enabled
by default. All motions are used unless `--clips` is specified; for debugging, use
`--clips walk1_subject1 run1_subject2`.

Robot self-collisions are disabled while full-body ground contacts remain enabled.
Training resets perturb the reference pose, root velocities and joint positions.
Foot friction and torso CoM are sampled once at startup; encoder bias is resampled
on each episode reset. Evaluation disables reset noise and domain randomization.
See the [environment configuration](doc/supermimic_foundation_zh.md) for ranges.

### Paper single-GPU training scale

The following command uses the network widths, batch sizes, environment count, and
training length in Tables A.4-A.5. It reads the converted manifest at
`src/assets/motions/g1/lafan1_npz/manifest.json` and trains the world model, Actor,
and Critic jointly from scratch on all 40 LAFAN clips. No standalone world-model
pretraining is required.

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

Two updates per five vector collection steps give an update-to-data ratio of `0.4`.
The world-model window is fixed at 8 steps. Defaults are a learning rate of `2e-4`,
discount `0.99`, GAE lambda `0.95`, and action exploration noise standard deviation
`0.1`. The learning rate follows a cosine decay to `0.03` times its initial value over
250,000 iterations, then stays constant. Evaluation and checkpoint intervals are
operational choices for this command, rather than paper hyperparameters. The script
also evaluates at initialization and at the end. The output directory must not exist.

Matching training scale does not establish exact reproduction of every implementation
detail. Real collection uses a 10-second episode limit; imagination proceeds
independently of that limit and retains value bootstrapping. World-model and Critic
losses sum over valid time steps and average over windows or trajectories. The full
workload requires substantially more memory than the reduced configuration, including
replay storage and the 32-step backward graph.

### Resume training

Network and batch settings are restored from the checkpoint. `--iterations` specifies
the total number of iterations, including those already completed:

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --resume outputs/supermimic/online_training/last.pt \
  --iterations 5000
```

New runs require a new output directory. Main outputs:

| File | Contents |
| --- | --- |
| `config.json` | Training settings, runtime versions, data and robot signatures |
| `metrics.jsonl` | Collection, model, Actor, Critic, and timing metrics |
| `simulator.log` | Simulator initialization and kernel output |
| `wandb_run.json`, `wandb/` | Optional W&B run identity and local tracking data |
| `last.pt` | Latest networks, optimizers, replay, sampler, and RNG states |
| `best.pt` | Checkpoint with the highest fixed-start simulator evaluation return |
| `iteration_*.pt` | Periodic training checkpoints |
| `evaluation*.json` | Per-motion returns, success counts, and failure reasons |

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

The same W&B flags can be appended to the paper-scale command. Optionally specify
`--wandb-entity YOUR_TEAM`. The terminal prints the run URL. W&B groups metrics under
`world_model/`, `policy/`, `critic/`, `collection/`, `gradients/`, `performance/`, and
`evaluation/`, with iteration as the horizontal axis. Evaluation includes normalized
return, success rate, completion, tracking errors, and per-motion bar charts and tables.
Collection metrics are sent only on collection iterations, so carried-forward values
are not counted twice. Model checkpoints and replay buffers stay in the output directory.
See the [W&B Run reference](https://docs.wandb.ai/ref/python/experiments/run/) for the SDK interfaces.

For a server without network access, record locally and sync later:

```bash
python scripts/train_supermimic.py \
  --device cuda:0 --iterations 1000 \
  --wandb --wandb-mode offline \
  --output outputs/supermimic/online_offline

wandb sync outputs/supermimic/online_offline/wandb/offline-run-*
```

When resuming training, pass `--wandb` again. `wandb_run.json` stores the run identity
and last logged iteration. Online runs continue with the saved ID when no checkpoint
rollback occurred. Offline resumes and rollbacks start a new run segment linked through
`parent_run_id`, keeping earlier curves intact. Logging flags can change on resume;
model and workload settings must still match the checkpoint. An already running process
uses its original logging code until it is restarted from a saved checkpoint.

## Policy evaluation

Load an online checkpoint and simulate each motion from its first frame, with domain
randomization and observation noise disabled:

```bash
python scripts/evaluate_supermimic.py \
  --device cuda:0 --checkpoint outputs/supermimic/online_training/best.pt \
  --output outputs/supermimic/online_policy_evaluation.json
```

Evaluation uses all training motions by default, until completion or physical failure.
Returns are normalized by the maximum possible return for the complete clip, so early
failure lowers the score. `--max-steps-per-clip` is for debugging; unfinished motions
are marked in the report. `best.pt` may still contain the initial policy; use the
evaluation report to determine whether training has improved tracking.

## Tests

```bash
SUPERMIMIC_TEST_DEVICE=cuda:0 \
python -m unittest discover -s tests -v
```

Tests cover conversion, motion loading, replay boundaries, simulation, differentiable
kinematics, BPTT, loss reduction, online updates, resume, and checkpoint inference.
Without a GPU, set `SUPERMIMIC_TEST_DEVICE=cpu`; CUDA-only tests will be skipped.

## Detailed guides

The following guides are currently in Chinese:

- [LAFAN data conversion](doc/lafan_conversion_zh.md)
- [Motion library, collection environment, and replay buffer](doc/supermimic_foundation_zh.md)
- [Standalone world-model training and validation](doc/supermimic_world_model_zh.md)
- [Online training and evaluation protocol](doc/supermimic_online_zh.md)

Standalone collection and model training support diagnostics and offline experiments.
Online training can start directly from scratch.
