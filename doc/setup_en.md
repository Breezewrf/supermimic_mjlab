# Installation

[简体中文](setup_zh.md) | [Project README](../README.md)

Run the commands below from the repository root. The recommended quick start
uses `uv sync` to create the environment and install the project. GPU training
requires an NVIDIA GPU, a compatible driver, and a CUDA-enabled PyTorch build.

## Quick start with uv

```bash
uv sync
source .venv/bin/activate
```

`uv sync` reads [pyproject.toml](../pyproject.toml), selects Python 3.11 via
`.python-version`, and installs the local project in editable mode. `uv.lock` locks
the resolved dependencies; use `uv sync --locked` to require the committed lockfile.
There is no separate environment-creation or editable-install step.

For W&B and Rich logging support:

```bash
uv sync --extra logging
```

Activate the environment with `source .venv/bin/activate` in each new terminal.
All subsequent commands use `python` and `wandb` from that environment.

The project version and direct dependency pins remain in [setup.py](../setup.py).
The dynamic metadata in pyproject.toml reads them directly, avoiding duplicate
version declarations. Synchronization uses uv.lock for the resolved dependency set.
If dependencies are already installed in a prepared environment, register the local
package without changing them:

```bash
uv pip install --no-deps -e .
```

The package distribution is named `supermimic_mjlab`; Python imports use
`src.tasks.supermimic`. The G1 XML and meshes are vendored under
[`src/assets/robots/g1`](../src/assets/robots/g1/README.md); mjlab supplies the
training PD actuators. No local Unitree SDK2 deployment simulator is required.
Model fingerprints are retained for provenance and do not block loading. Robot
joint/body ordering, topology, network dimensions and simulation settings must be
compatible; reproducing historical scores still requires the original setup.

## Verify the environment

```bash
python - <<'PYTHON'
from importlib.metadata import version
import torch

for package in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "scipy"):
    print(package, version(package))
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PYTHON
```

The runtime versions are also saved in training configuration and checkpoints.
For GPU training, confirm that CUDA is available in the same Python environment
used to launch training.

## Prepare motions and train

Motion datasets and training outputs are excluded from version control. Place the
retargeted G1 LAFAN CSV files under
`src/assets/motions/LAFAN1_Retargeting_Dataset/`, then follow the
[data conversion instructions](lafan_conversion_zh.md). Conversion creates the NPZ
clips and manifest used by training.

See the [README](../README.md#online-training) for the released checkpoint, its training configuration and playback.
Reduced and paper-scale workloads, checkpoints and evaluation are in the
[operational guide](training_and_playback_en.md). See its [tests section](training_and_playback_en.md#tests)
for regression checks. Tests generate small motion and replay fixtures in temporary
directories; real-dataset checks additionally require the converted LAFAN dataset.
