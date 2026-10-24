# Training, checkpoints, evaluation and playback

[简体中文](training_and_playback_zh.md) | [Back to README](../README.md)

Run all commands from the repository root after [installation](setup_en.md) and motion conversion. The [preview checkpoint section](../README.md#reproduce-the-preview-checkpoint-iteration-10000) records the current snapshot, measured evaluation and its training command.

## Training variants

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
See the [environment configuration](supermimic_foundation_zh.md) for ranges.

### Paper single-GPU training scale

The following command uses the network widths, batch sizes, environment count, and
training length in Tables A.4-A.5. It reads the converted manifest at
`src/assets/motions/g1/lafan1_npz/manifest.json` and trains the world model, Actor,
and Critic jointly from scratch on all 40 LAFAN clips. No standalone world-model
pretraining is required.
To warm up the world model first, add `--warmup-steps 64 --wm-only-iters 500`:
the first 500 rounds after initial collection keep collecting and fitting the
world model, with Actor and Critic updates starting at round 501. Warmup rounds
count toward `--iterations`, and checkpoints preserve progress. The default is `0`.

CUDA training defaults to `--compile --amp bf16`. Core tensor computations use
`torch.compile`; only MLPs use BF16 autocast, while physical computations remain
FP32. Rewards and Critic targets are batched across imagined time steps. Reference
inputs default to 73 features per frame (730 per window). First updates compile
kernels; measure steady-state speed afterward. CPU uses eager FP32. Use
`--no-compile` or `--amp none` to isolate execution issues.
Use `--reference-dim 2680` to train the original 268-feature-per-frame full-body
representation, including its original scaling and absolute reference height;
`--reference-dim 730` selects the compact representation (default). Actor and Critic
use the selected layout throughout collection and imagination. For an FP32 eager
comparison with the old baseline, add `--amp none --no-compile`.
Resume automatically restores the reference dimension from a full checkpoint,
including old 2680-input checkpoints without schema metadata. Old checkpoints
without execution settings resume in eager FP32. A resume cannot change the layout;
use a new output directory to train a different architecture.

New runs default to `--world-prediction state_delta`: the world model predicts complete
one-step state increments, without adding a constant-velocity extrapolation. Actor
outputs remain normalized joint-position commands. Old checkpoints without this setting
load as `velocity_residual`, preserving their original predictions, including the preview
world model. Resume restores the saved mode and cannot switch it. To train the new
parameterization, start a new run; use `--world-prediction velocity_residual` for a legacy
baseline. Input statistics, hidden activations and loss scales retain their existing rules.

Optional input `RunningNorm` has independent switches, both off by default.
Append these flags to a new training command to enable both:

```bash
--actor-input-normalization --critic-input-normalization
```

Actor observations and goals use separate statistics; all history frames
share the observation statistics. Critic features use their own statistics.
Only real collection updates the moments; imagination, critic fitting, evaluation
and playback hold them fixed. Checkpoints and actor exports preserve the moments
for both reference layouts. Resume restores the switches automatically and cannot
toggle normalization on an existing run; older checkpoints default to disabled.

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

## Checkpoints and exports

New runs require a new output directory. Main outputs:

| File | Contents |
| --- | --- |
| `config.json` | Training settings, runtime versions, data and robot signatures |
| `metrics.jsonl` | Collection, model, Actor, Critic, and timing metrics |
| `simulator.log` | Simulator initialization and kernel output |
| `wandb_run.json`, `wandb/` | Optional W&B run identity and local tracking data |
| `last.pt` | Full resume state: networks, optimizers, replay, sampler, and RNG states |
| `policy_best.pt`, `world_model_best.pt` | Matching best Actor and world model: most successful motions, then highest normalized return |
| `policy_last.pt`, `world_model_last.pt` | Latest saved matching inference weights and metadata |
| `policy_iteration_*.pt`, `world_model_iteration_*.pt` | Paired inference history at checkpoint intervals, the final iteration, and every new best |
| `best_checkpoint.json` | Selection rule, evaluation scores, iteration and paths to the retained best pair |
| `iteration_*.pt` | Historical full snapshots, enabled with `--full-checkpoint-interval N` |
| `evaluation*.json` | Per-motion returns, success counts, and failure reasons |

By default, training updates one full `last.pt` and keeps paired inference history.
Policy files contain the Actor and validation metadata, without replay, world model,
Critic, or optimizers. Use them for evaluation and MuJoCo playback; resume requires a
full training checkpoint. Add `--full-checkpoint-interval 10000` to retain full
historical snapshots every 10000 iterations. No old files are automatically deleted.
Inference history includes world-model weights and can consume substantial disk space
over a long run; increase `--checkpoint-interval` to save less frequently.

Best selection uses fixed-start evaluation, including the initial policy. A 40/40
policy remains best over a later 39/40 policy even if the later return is higher.
Equal success counts compare normalized return; an exact tie keeps the earlier model.
`best_checkpoint.json` is published after both files are saved and points to the
numbered history, so it identifies the pair even if an alias update is interrupted.
For example, `--checkpoint-interval 1000` retains both
`policy_iteration_010000.pt` and `world_model_iteration_010000.pt`; a later evaluation
regression does not overwrite those files. Lightweight best/history files cannot
resume training because they omit the Critic, optimizers and replay.

Export an existing full checkpoint without retraining, a GPU, or motion data:

```bash
python scripts/export_supermimic_policy.py \
  --checkpoint outputs/supermimic/lafan_horizon_fixed/best.pt \
  --output outputs/supermimic/lafan_horizon_fixed/policy_best.pt
```

Export the world model separately with the same script:

```bash
python scripts/export_supermimic_policy.py \
  --component world-model \
  --checkpoint outputs/supermimic/lafan_horizon_fixed/last.pt \
  --output outputs/supermimic/lafan_horizon_fixed/world_model_last.pt
```

The default component is `policy`. World-model files use kind
`supermimic_world_model` and retain the complete `world` state dict, including
normalization buffers, kinematics and the termination head, plus model configuration,
robot/physics signatures, motion provenance and iteration. They contain no Actor,
Critic, replay, optimizers or RNG state and cannot resume training. Policy-only
files have no world weights to extract. New runs save both components automatically;
for older runs, export both from the same full checkpoint to obtain a matching pair.
Pass the world-model file to `play --world-model <path>` for a prediction overlay.

Run from the project root. Export preserves the source and memory-maps replay storage
without reading it into RAM. Existing `last.pt` and `iteration_*.pt` can be exported
the same way. Evaluation and playback accept either format with the same actor weights
and data/robot/physics checks. On resume, retained policies and evaluation reports up
to the resumed iteration are ranked by the new success-first rule. Older full snapshots
can supply missing components. If the best retained policy has no matching world model,
`best_checkpoint.json` records `world_model: null` and training prints a notice; it
cannot reconstruct weights that were never saved. A stale world-model best alias is
preserved as `world_model_previous_best*.pt`. Existing full snapshots are retained.
Export does not migrate legacy feature schemas or physics settings.
When `reference_schema` is missing, the Actor's `encoder.0.weight` input width
identifies `root_joint_73_v1` (730) or `full_body_268` (2680). Export prints the
inferred schema and records `reference_schema_inferred: true`. Evaluation and
playback automatically select the original full-body feature computation for
2680-input policies, preserving their weights. Unrecognized input widths are rejected.
An already running training process needs to restart from `last.pt` to use this layout.

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
  --device cuda:0 --checkpoint outputs/supermimic/online_training/policy_best.pt \
  --output outputs/supermimic/online_policy_evaluation.json \
  --csv outputs/supermimic/eval.csv
```

Evaluation uses all training motions by default, until completion or physical failure.
Training evaluations and headless evaluation run clips in parallel, with up to 64
environments by default (all 40 LAFAN1 clips fit in one batch). Use
`--eval-num-envs N` during training or `--num-envs N` in the evaluation script to
limit parallelism; `1` selects serial evaluation. Larger motion libraries run in
batches. Each clip stops contributing rewards and errors at its first completion
or failure, including when its environment automatically resets. Playback, viewer
and video use one environment. Reports record `evaluation_num_envs`; training also
reports `evaluation_wall_seconds`, including environment initialization.
Returns are normalized by the maximum possible return for the complete clip, so early
failure lowers the score. `--max-steps-per-clip` is for debugging; unfinished motions
are marked in the report. `policy_best.pt` may still contain the initial policy; use the
evaluation report to determine whether training has improved tracking.

Following the [reference evaluator](https://github.com/zita-ch/supermimic_vibed/blob/main/supermimic/evaluate.py),
the terminal prints `Evaluation return <score>`, for example `Evaluation return 0.824`
(an illustration of the format, not a measured result). Each clip's undiscounted reward
is divided by `5 * 0.02 * (frames - 1)`, then averaged equally across clips. JSON also
reports `normalized_return_framew` (weighted by full clip lengths), per-clip errors,
completion, `n_failed`, and `failed`. Debug limits and unexpected timeouts are marked
`INCOMPLETE` / `evaluation_complete: false`. Physical failures are evaluated outcomes;
their denominators still include the entire clip. CSV rows include checkpoint,
iteration, motion selection, and scores for comparison on the same motion set.

## Playback and world-model visualization

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint outputs/supermimic/online_training/policy_best.pt \
  --clips walk1_subject1 run1_subject2 --loop
```

Add a world-model comparison using the exported weights:

```bash
python scripts/play_supermimic.py \
  --device cuda:0 --checkpoint checkpoints/preview/policy_10000.pt \
  --world-model checkpoints/preview/world_model_10000.pt \
  --world-model-horizon 32 --clips walk1_subject1 run1_subject2 --loop
```

The default `--world-model-mode future` runs the policy on imagined observations
inside the world model. The orange robot shows the predicted pose **32 steps
(0.64 s) ahead** of real physics, with an orange root trajectory. The preview is
recomputed from the current real state/history every `--world-model-refresh 4`
steps; its cached trajectory advances between refreshes, keeping the same future
lead. Reference features are encoded in a batch to reduce preview overhead. Increase
the refresh interval if rendering slows. The lead shortens at the clip end.

Use `--world-model-mode compare` for the original same-time dynamics comparison:
the model consumes actual actions and resyncs every horizon steps. In this mode,
`--world-model-horizon 1` measures single-step error. `W` toggles the orange robot;
`G` toggles the blue reference. Prediction poses retain absolute position and heading.
Future mode labels its time lead and compares each displayed forecast only when
the real rollout reaches the matching target frame; it never scores a future pose
against the current pose. JSON records the mode and per-clip errors under
`world_model_comparison`. Nonfinite predictions are hidden until the next refresh.
Model files are checked against the motion manifest, selected clips, robot signature
and physics settings; prefer policy/world weights from the same iteration.

The same options work with `--video` and with `--viewer none` for diagnostics. World
comparisons use one environment and leave the physical rollout and evaluation return
unchanged. Mouse forces affect physics; the world model has no external-force input,
so dragging can cause its prediction to diverge from the real robot.

The solid robot executes the policy in MuJoCo Warp; the translucent blue robot shows
the current reference. Reference XY and heading are aligned as in the body tracking
reward by default; `--global-ghost` reveals absolute position and heading errors.
Space pauses, Right steps once, and `G` toggles the reference. Close the window to exit.
Use `--speed 0.5` for slow motion or `--no-ghost` to hide the reference. `--loop` repeats
selected clips, saving a report after each complete pass. Closing the window midway
does not save the interrupted pass. Playback disables height/tilt failure termination
by default, so falls continue until the clip ends for recovery inspection. Clip ends
still switch to the next motion. Add `--termination` to restore failure resets and
display their final pre-reset poses. `evaluate_supermimic.py --viewer native` also enables live viewing;
evaluation is headless by default.

Playback enables mouse forces by default, following the same native viewer path as
`AMP_mjlab`. Double-click a body on the solid robot, then hold Ctrl while dragging:
Ctrl + right drag applies a translational force; add Shift for the horizontal plane.
Ctrl + left drag applies torque. Dragging without Ctrl controls the camera.
Releasing clears the mouse force on the next control step; motion changes and window
shutdown also clear it. Pausing stops physics; Right steps once with the current drag.
See the [MuJoCo mouse controls](https://mujoco.readthedocs.io/en/stable/programming/samples.html#mouse).

Use `--no-perturbations` to disable mouse forces. The evaluator disables them by
default; opt in with `--viewer native --perturbations` for interactive testing.
Passes with applied mouse forces show `INTERACTIVE` and record
`evaluation_mode: interactive` plus `mouse_perturbation_steps` in JSON/CSV. Passes
without applied forces are `playback` when failure termination is disabled, and
`standard` when it is enabled. JSON/CSV also record `termination_enabled`; first
failure steps and reasons remain diagnostic, so reaching a clip's end after falling
does not count as tracking success. Evaluation retains failure termination by default;
use `--no-termination` explicitly for diagnostic playback.

On a server without a desktop, export the same evaluation to MP4 (requires `ffmpeg`):

```bash
MUJOCO_GL=egl python scripts/evaluate_supermimic.py \
  --device cuda:0 --checkpoint outputs/supermimic/online_training/policy_best.pt \
  --clips walk1_subject1 --video outputs/supermimic/walk.mp4 \
  --output outputs/supermimic/walk_evaluation.json
```

Selected clips are concatenated at 960×540, 50 FPS. Live viewing requires a desktop
display; MP4 export does not. Both paths validate checkpoint features, manifest,
robot signature, and physics settings. Known 730/2680-input policies are selected
automatically, including older checkpoints without `reference_schema`. Incompatible
physics configurations are still rejected.
Playback requires `--checkpoint` or `--policy`; evaluation without either explicitly
runs a zero-action baseline.

## Tests

```bash
SUPERMIMIC_TEST_DEVICE=cuda:0 \
python -m unittest discover -s tests -v
```

Tests cover conversion, motion loading, replay boundaries, simulation, differentiable
kinematics, BPTT, loss reduction, online updates, resume, and checkpoint inference.
Without a GPU, set `SUPERMIMIC_TEST_DEVICE=cpu`; CUDA-only tests will be skipped.

## Python API

```python
from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.online_training import OnlineTrainer
```
