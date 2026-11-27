#!/usr/bin/env bash
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  echo "Run this launcher with bash: bash ${BASH_SOURCE[0]} motionbrick180 frozen cuda:0" >&2
  return 2
fi
set -euo pipefail

usage() {
  cat <<EOF
Usage: bash $0 DATASET MODE [DEVICE] [OUTPUT] [OPTIONS]
  DATASET: bones | lafan | motionbrick180
  MODE: frozen | online
Options:
  --stand-still-reference motionbrick|unitree  (default: motionbrick)
  --stand-still-weight VALUE                  (default: -1.0; 0 disables)
  --self-collisions | --no-self-collisions     (default: disabled)
  --self-collision-weight VALUE               (default: -1.0; 0 disables penalty)
  --actor-checkpoint PATH
  --world-checkpoint PATH
  --dry-run                                  Show configuration without training
EOF
}
reference=motionbrick
stand_weight=-1.0
collision=--no-self-collisions
collision_name=off
collision_weight=-1.0
actor_checkpoint=
world_checkpoint=
dry_run=false
positional=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stand-still-reference|--stand-still-weight|--self-collision-weight|--actor-checkpoint|--world-checkpoint)
      if [[ $# -lt 2 || -z "$2" ]]; then echo "Missing value for $1" >&2; exit 2; fi
      case "$1" in
        --stand-still-reference) reference=$2 ;;
        --stand-still-weight) stand_weight=$2 ;;
        --self-collision-weight) collision_weight=$2 ;;
        --actor-checkpoint) actor_checkpoint=$2 ;;
        --world-checkpoint) world_checkpoint=$2 ;;
      esac
      shift 2 ;;
    --self-collisions) collision=--self-collisions; collision_name=on; shift ;;
    --no-self-collisions) collision=--no-self-collisions; collision_name=off; shift ;;
    --dry-run) dry_run=true; shift ;;
    -h|--help) usage; exit 0 ;;
    --*) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    *) positional+=("$1"); shift ;;
  esac
done
if [[ ${#positional[@]} -lt 2 || ${#positional[@]} -gt 4 ]]; then usage >&2; exit 2; fi
case "$reference" in motionbrick|unitree) ;; *) echo "Choose motionbrick or unitree for stand-still reference" >&2; exit 2 ;; esac
dataset=${positional[0]}
mode=${positional[1]}
device=${positional[2]:-cuda:0}
inputs=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo=$inputs
while [[ ! -f "$repo/scripts/train_velocity.py" || ! -d "$repo/src/tasks/supermimic" ]]; do
  if [[ "$repo" == / ]]; then
    echo "Cannot find the SuperMimic repository above $inputs" >&2
    exit 1
  fi
  repo=$(dirname -- "$repo")
done
cd -- "$repo"

case "$mode" in
  frozen) freeze=--freeze-world-model ;;
  online) freeze=--no-freeze-world-model ;;
  *) echo "Choose frozen or online" >&2; exit 2 ;;
esac

root=outputs/velocity/g1-29dof
pretrained=outputs/supermimic/bones_seed_10k_womoanorib
storage=full
initial_evaluation_steps=250
case "$dataset" in
  bones)
    manifest=src/assets/motions/bone-seed-unisupermimic/manifest.json
    output="$root/bones_seed_10k_add_motion"
    if [[ "$mode" == frozen ]]; then
      output="$root/bones_seed_10k_frozen_add_motion"
    fi
    ;;
  lafan)
    manifest=src/assets/motions/g1/lafan1_npz/manifest.json
    output="$root/lafan1_bones_init_$mode"
    ;;
  motionbrick180)
    manifest=src/assets/motions/g1/MotionBrick_180/npz/manifest.json
    output="$root/motionbrick180_bones_init_$mode"
    storage=compact
    initial_evaluation_steps=0
    ;;
  *) echo "Choose bones, lafan or motionbrick180" >&2; exit 2 ;;
esac
actor_checkpoint=${actor_checkpoint:-$pretrained/policy_best.pt}
world_checkpoint=${world_checkpoint:-$pretrained/world_model_best.pt}
output="${output}_stand-${reference}-w${stand_weight}_collision-${collision_name}-w${collision_weight}"
output=${positional[3]:-$output}

if [[ ! -x .venv/bin/python ]]; then
  echo "Python is missing or not executable: $repo/.venv/bin/python" >&2
  exit 1
fi
if [[ ! -f "$inputs/velocity_config.json" ]]; then
  echo "Copy velocity_config.json alongside this launcher: $inputs" >&2
  exit 1
fi

.venv/bin/python -B - "$stand_weight" "$collision_weight" "$mode" "$collision_name" "$world_checkpoint" "$dry_run" <<'CHECK'
import math
import sys
from pathlib import Path
for value in sys.argv[1:3]:
    try:
        valid = math.isfinite(float(value)) and float(value) <= 0
    except ValueError:
        valid = False
    if not valid:
        raise SystemExit(f"Penalty weights must be finite and nonpositive: {value}")
mode, collision, world, dry = sys.argv[3:]
if mode == "frozen" and collision == "on" and dry != "true":
    import torch
    checkpoint = torch.load(Path(world), map_location="cpu", weights_only=True, mmap=True)
    if not checkpoint.get("simulation", {}).get("self_collisions", False):
        raise SystemExit("Frozen + self-collisions requires an adapted collision world checkpoint. Use online mode first, or pass --world-checkpoint with adapted weights.")
CHECK

scratch=$(mktemp -d /tmp/supermimic-velocity-clips.XXXXXX)
trap 'rm -rf "$scratch"' EXIT

.venv/bin/python -B - "$manifest" "$inputs/manifest_reviewed.json" "$dataset" "$scratch/clips.txt" <<'PY'
import json
import re
import sys
from pathlib import Path

manifest, reviewed_path = map(Path, sys.argv[1:3])
dataset, output = sys.argv[3:]
data = json.loads(manifest.read_text())
if data.get("schema_version") != 1 or data.get("fps") != 50 or data.get("quaternion_order") != "wxyz":
    raise SystemExit(f"Expected a schema 1, 50 Hz, wxyz manifest: {manifest}")
entries = {row["file"]: row["frames"] for row in data["clips"]}
if dataset == "bones":
    reviewed = json.loads(reviewed_path.read_text())
    if reviewed["robot"] != "g1_29dof" or not reviewed["clips"]:
        raise SystemExit("Expected reviewed G1 29DoF clips")
    for row in reviewed["clips"]:
        if entries.get(row["file"]) != row["frames"]:
            raise SystemExit(f"Reviewed clip differs from the dataset: {row['file']}")
    files = [
        "g1/npz/210531/walk_forward_amateur_001__A001.npz",
        "g1/npz/220713/jog_forward_loop_001__A021.npz",
        "g1/npz/230417/run_loop_180_R_003__A326.npz",
    ]
    files.extend(row["file"] for row in reviewed["clips"])
elif dataset == "lafan":
    files = [name for name in entries if re.fullmatch(
        r"(?:walk|run|sprint)\d+_subject\d+\.npz", name
    )]
else:
    files = list(entries)
files = list(dict.fromkeys(files))
if not files:
    raise SystemExit(f"No clips selected from {manifest}")
for name in files:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or path.suffix != ".npz":
        raise SystemExit(f"Expected a relative NPZ path: {name}")
    if name not in entries or not (manifest.parent / path).is_file():
        raise SystemExit(f"Selected motion is missing: {name}")
Path(output).write_text(
    "\n".join(Path(name).with_suffix("").as_posix() for name in files) + "\n"
)
print(f"{dataset}: {len(files)} reference clips", flush=True)
PY
mapfile -t clips < "$scratch/clips.txt"

command=(.venv/bin/python scripts/train_velocity.py \
  --robot g1_29dof --device "$device" --seed 0 --stage joint \
  "$collision" --self-collision-weight "$collision_weight" \
  "$freeze" --warmup-updates 500 --iterations 5000 \
  --actor-checkpoint "$actor_checkpoint" \
  --world-checkpoint "$world_checkpoint" \
  --manifest "$manifest" \
  --velocity-config "$inputs/velocity_config.json" \
  --num-envs 4096 --policy-batch-size 4096 --wm-batch-size 2048 \
  --capacity 1000 --warmup-steps 32 --collect-steps 5 --updates-per-collect 2 \
  --wm-batches 10 --horizon 32 --critic-epochs 5 --critic-minibatches 4 \
  --learning-rate 0.0002 --stand-still-reference "$reference" --stand-still-weight "$stand_weight" --motion-storage "$storage" \
  --initial-evaluation-steps "$initial_evaluation_steps" \
  --no-compile --checkpoint-interval 1000 --output "$output" --tensorboard
)
printf 'Stand-still: reference=%s weight=%s | Self-collisions: %s weight=%s | World: %s\n' "$reference" "$stand_weight" "$collision_name" "$collision_weight" "$mode"
printf 'Output: %s\n' "$output"
if [[ "$dry_run" == true ]]; then
  printf '%q ' "${command[@]}"
  printf '\nClips: %s selected from %s (passed through --clips)\n' "${#clips[@]}" "$manifest"
  if [[ "$mode" == frozen && "$collision_name" == on ]]; then
    echo "Requires --world-checkpoint trained with self-collisions; checked when training starts."
  fi
  exit 0
fi
"${command[@]}" --clips "${clips[@]}"
