# Installation

[简体中文](setup_zh.md) | [Project README](../README.md)

Use Python 3.11 and run these commands from the repository root. GPU training
requires an NVIDIA GPU, a compatible driver, and a CUDA-enabled PyTorch build.

## Create an environment

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

Dependency versions are defined in [setup.py](../setup.py). Installation uses those
existing version pins; this guide does not prescribe a separate set of versions.
If dependencies are already installed in a prepared environment, register the local
package without changing them:

```bash
.venv/bin/python -m pip install --no-deps -e .
```

The package distribution is named `supermimic_mjlab`; Python imports use
`src.tasks.supermimic`. Robot definitions and meshes are provided by the installed
mjlab package. No local Unitree SDK2 deployment simulator is required.

## Verify the environment

```bash
.venv/bin/python - <<'PYTHON'
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

See the [README](../README.md#online-training) for reduced and paper-scale training
commands, resume, and policy evaluation. See the [tests section](../README.md#tests)
for regression checks. Tests generate small motion and replay fixtures in temporary
directories; real-dataset checks additionally require the converted LAFAN dataset.
