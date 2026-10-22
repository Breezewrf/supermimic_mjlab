"""Batch-convert retargeted G1 LAFAN CSV files to mjlab tracking NPZ files.

CSV: root xyz (metres), root quaternion (xyzw by default), 29 joint angles
(radians) in CSV_JOINT_NAMES order; default input rate is 30 Hz.
Uses CPU MuJoCo kinematics; no GPU, renderer, or physics rollout is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CSV_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)
MOTION_FIELDS = (
  "joint_pos",
  "joint_vel",
  "body_pos_w",
  "body_quat_w",
  "body_lin_vel_w",
  "body_ang_vel_w",
)


def positive_fps(value: str) -> float:
  fps = float(value)
  if not math.isfinite(fps) or fps <= 0:
    raise argparse.ArgumentTypeError("FPS must be finite and positive")
  return fps


def normalize_quaternions(quaternions: np.ndarray) -> np.ndarray:
  norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
  if np.any(norms < 1e-8):
    raise ValueError("Root quaternion has zero or near-zero norm")
  quaternions = quaternions / norms
  # q and -q describe the same rotation. Make the sequence sign-continuous.
  flips = np.where(np.sum(quaternions[1:] * quaternions[:-1], axis=-1) < 0, -1, 1)
  signs = np.concatenate(([1], np.cumprod(flips)))
  return quaternions * signs[:, None]


def resample_motion(
  motion: np.ndarray,
  input_fps: float,
  output_fps: float,
  quaternion_order: str = "xyzw",
) -> np.ndarray:
  """Return xyz, wxyz, and joint angles on a uniform output time grid.

  Include the final input timestamp if it lies on the output grid. Never
  extrapolate or append an irregular last timestep. Joint angles interpolate
  linearly; orientations use shortest-path SLERP.
  """
  if motion.ndim != 2 or motion.shape[1] != 36 or len(motion) < 2:
    raise ValueError(
      f"Expected at least two rows and exactly 36 columns; got {motion.shape}"
    )
  if not np.isfinite(motion).all():
    raise ValueError("CSV contains NaN or infinite values")
  if not all(math.isfinite(f) and f > 0 for f in (input_fps, output_fps)):
    raise ValueError("FPS must be finite and positive")
  if quaternion_order not in ("xyzw", "wxyz"):
    raise ValueError("quaternion_order must be xyzw or wxyz")
  quats = motion[:, 3:7]
  if quaternion_order == "xyzw":
    quats = quats[:, [3, 0, 1, 2]]
  quats = normalize_quaternions(quats)
  count = math.floor((len(motion) - 1) * output_fps / input_fps + 1e-9) + 1
  if count < 2:
    raise ValueError("Motion is too short for the requested output FPS")
  phase = np.minimum(np.arange(count) * input_fps / output_fps, len(motion) - 1)
  left = np.floor(phase).astype(np.int64)
  right = np.minimum(left + 1, len(motion) - 1)
  blend = (phase - left)[:, None]
  result = (1 - blend) * motion[left] + blend * motion[right]
  qa, qb = quats[left], quats[right]
  dot = np.clip(np.sum(qa * qb, axis=-1), 0, 1)
  theta = np.arccos(dot)[:, None]
  denominator = np.maximum(np.sin(theta), 1e-12)
  spherical = (
    np.sin((1 - blend) * theta) / denominator * qa
    + np.sin(blend * theta) / denominator * qb
  )
  linear = (1 - blend) * qa + blend * qb
  interpolated = np.where((dot > 0.9995)[:, None], linear, spherical)
  result[:, 3:7] = normalize_quaternions(interpolated)
  return result


def load_robot(model_xml: Path | None = None):
  """Use the same mjlab G1 factory as the training task by default."""
  if model_xml is None:
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.entity import Entity

    robot = Entity(get_g1_robot_cfg())
    model = robot.spec.compile()
    joint_names, body_names = tuple(robot.joint_names), tuple(robot.body_names)
    source = "mjlab.asset_zoo.robots.get_g1_robot_cfg"
  else:
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    joint_names = tuple(
      model.joint(i).name
      for i in range(model.njnt)
      if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE
    )
    body_names = tuple(model.body(i).name for i in range(1, model.nbody))
    source = str(model_xml.resolve())
  return model, joint_names, body_names, source


def motion_to_arrays(
  model: mujoco.MjModel,
  joint_names: tuple[str, ...],
  body_names: tuple[str, ...],
  motion: np.ndarray,
  fps: float,
) -> dict[str, np.ndarray]:
  """Evaluate body-link origins and world velocities, matching EntityData.

  MuJoCo free-joint angular qvel is local. mj_differentiatePos computes it
  directly in the current pose's tangent frame; no world/local approximation
  is needed. cvel linear velocities are shifted from the subtree COM to the
  body-link origins, exactly as mjlab's body_link_lin_vel_w does.
  """
  if len(joint_names) != 29 or set(joint_names) != set(CSV_JOINT_NAMES):
    raise ValueError("Robot must have exactly the 29 named G1 joints")
  free_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
  if len(free_ids) != 1 or model.nq != 36 or model.nv != 35:
    raise ValueError("Expected one floating-base G1 (nq=36, nv=35)")
  joint_ids = np.array([model.joint(name).id for name in joint_names])
  if np.any(model.jnt_type[joint_ids] != mujoco.mjtJoint.mjJNT_HINGE):
    raise ValueError("G1 joints must be hinge joints")
  body_ids = np.array([model.body(name).id for name in body_names])
  root_body = int(model.jnt_bodyid[free_ids[0]])
  if len(body_ids) == 0 or body_ids[0] != root_body:
    raise ValueError("Body order must start with the floating root (pelvis)")
  if np.any(model.body_rootid[body_ids] != root_body):
    raise ValueError("All exported bodies must belong to the robot subtree")
  qadr, vadr = model.jnt_qposadr[joint_ids], model.jnt_dofadr[joint_ids]
  csv_order = [CSV_JOINT_NAMES.index(name) for name in joint_names]
  count = len(motion)
  qpos = np.tile(model.qpos0, (count, 1))
  root_qadr = int(model.jnt_qposadr[free_ids[0]])
  qpos[:, root_qadr : root_qadr + 7] = motion[:, :7]
  qpos[:, qadr] = motion[:, 7:][:, csv_order]
  qvel = np.empty((count, model.nv))
  forward, backward = np.zeros(model.nv), np.zeros(model.nv)
  for t in range(count):
    if t < count - 1:
      mujoco.mj_differentiatePos(model, forward, 1 / fps, qpos[t], qpos[t + 1])
    if t > 0:
      mujoco.mj_differentiatePos(model, backward, 1 / fps, qpos[t], qpos[t - 1])
    qvel[t] = (
      forward if t == 0 else -backward if t == count - 1 else (forward - backward) / 2
    )

  arrays = {
    "fps": np.array([fps], dtype=np.float32),
    "joint_pos": qpos[:, qadr].astype(np.float32),
    "joint_vel": qvel[:, vadr].astype(np.float32),
    "body_pos_w": np.empty((count, len(body_ids), 3), dtype=np.float32),
    "body_quat_w": np.empty((count, len(body_ids), 4), dtype=np.float32),
    "body_lin_vel_w": np.empty((count, len(body_ids), 3), dtype=np.float32),
    "body_ang_vel_w": np.empty((count, len(body_ids), 3), dtype=np.float32),
    "joint_names": np.asarray(joint_names),
    "body_names": np.asarray(body_names),
    "quaternion_order": np.asarray("wxyz"),
  }
  data = mujoco.MjData(model)
  for t in range(count):
    data.qpos[:] = qpos[t]
    data.qvel[:] = qvel[t]
    # Kinematics only: no collision solve, integration, GPU, or renderer.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    pos = data.xpos[body_ids]
    omega = data.cvel[body_ids, :3]
    arrays["body_pos_w"][t] = pos
    arrays["body_quat_w"][t] = data.xquat[body_ids]
    arrays["body_ang_vel_w"][t] = omega
    arrays["body_lin_vel_w"][t] = data.cvel[body_ids, 3:] + np.cross(
      omega, pos - data.subtree_com[root_body]
    )
  validate_arrays(arrays)
  return arrays


def validate_arrays(arrays: dict[str, np.ndarray]) -> None:
  count = len(arrays["joint_pos"])
  bodies = len(arrays["body_names"])
  for name in MOTION_FIELDS:
    shape = (
      (count, 29)
      if name.startswith("joint_")
      else (count, bodies, 4 if name == "body_quat_w" else 3)
    )
    if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
      raise ValueError(f"Invalid shape or nonfinite values in {name}")
  if not np.allclose(np.linalg.norm(arrays["body_quat_w"], axis=-1), 1, atol=1e-5):
    raise ValueError("Output body quaternions are not normalized")


def save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
  """Atomic replacement prevents an interrupted write leaving a partial NPZ."""
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
      temporary = Path(f.name)
      np.savez_compressed(f, **arrays)
    with np.load(temporary, allow_pickle=False) as saved:
      validate_arrays({name: saved[name] for name in (*MOTION_FIELDS, "body_names")})
    os.replace(temporary, path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-dir",
    type=Path,
    default=ROOT / "src/assets/motions/LAFAN1_Retargeting_Dataset",
  )
  parser.add_argument(
    "--output-dir", type=Path, default=ROOT / "src/assets/motions/g1/lafan1_npz"
  )
  parser.add_argument("--pattern", default="*.csv", help="Glob relative to input-dir")
  parser.add_argument("--input-fps", type=positive_fps, default=30.0)
  parser.add_argument("--output-fps", type=positive_fps, default=50.0)
  parser.add_argument("--quaternion-order", choices=("xyzw", "wxyz"), default="xyzw")
  parser.add_argument(
    "--model-xml", type=Path, help="Override mjlab G1 with a standalone G1 XML"
  )
  parser.add_argument(
    "--overwrite", action="store_true", help="Replace existing NPZ files"
  )
  args = parser.parse_args()
  sources = sorted(path for path in args.input_dir.glob(args.pattern) if path.is_file())
  if not sources:
    parser.error(f"No files match {args.input_dir / args.pattern}")
  destinations = [
    args.output_dir / path.relative_to(args.input_dir).with_suffix(".npz")
    for path in sources
  ]
  existing = [path for path in destinations if path.exists()]
  if existing and not args.overwrite:
    parser.error(f"Output already exists: {existing[0]}; use --overwrite to replace")
  model, joints, bodies, model_source = load_robot(args.model_xml)
  # Hash the compiled model, including the actual geometry and joint definitions.
  with tempfile.TemporaryDirectory() as directory:
    binary = Path(directory) / "robot.mjb"
    mujoco.mj_saveModel(model, str(binary), None)
    model_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
  entries = []
  print(
    f"Converting {len(sources)} clips: {args.input_fps:g} -> {args.output_fps:g} Hz; CPU MuJoCo",
    flush=True,
  )
  for index, (source, destination) in enumerate(zip(sources, destinations), 1):
    try:
      raw = np.loadtxt(source, delimiter=",", ndmin=2)
      motion = resample_motion(
        raw, args.input_fps, args.output_fps, args.quaternion_order
      )
      arrays = motion_to_arrays(model, joints, bodies, motion, args.output_fps)
      source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
      arrays.update(
        {
          "source_file": np.asarray(str(source.resolve())),
          "source_sha256": np.asarray(source_hash),
          "input_fps": np.array([args.input_fps], dtype=np.float32),
          "input_quaternion_order": np.asarray(args.quaternion_order),
          "model_source": np.asarray(model_source),
          "model_sha256": np.asarray(model_hash),
          "mujoco_version": np.asarray(mujoco.__version__),
          "schema_version": np.array([1], dtype=np.int32),
        }
      )
      save_npz(destination, arrays)
    except Exception as exc:
      raise RuntimeError(f"Failed converting {source}: {exc}") from exc
    entries.append(
      {
        "source": str(source.resolve()),
        "source_sha256": source_hash,
        "file": str(destination.relative_to(args.output_dir)),
        "input_frames": len(raw),
        "frames": len(motion),
        "duration_seconds": (len(motion) - 1) / args.output_fps,
      }
    )
    print(
      f"[{index}/{len(sources)}] {source.name}: {len(raw)} -> {len(motion)} frames",
      flush=True,
    )
  manifest = {
    "schema_version": 1,
    "input_fps": args.input_fps,
    "fps": args.output_fps,
    "input_quaternion_order": args.quaternion_order,
    "quaternion_order": "wxyz",
    "model_source": model_source,
    "model_sha256": model_hash,
    "mujoco_version": mujoco.__version__,
    "joint_names": joints,
    "body_names": bodies,
    "clips": entries,
  }
  # The manifest describes this invocation's selected clips, in deterministic order.
  manifest_path = args.output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
  hours = sum(entry["duration_seconds"] for entry in entries) / 3600
  print(
    f"Saved {len(entries)} clips ({hours:.3f} h) to {args.output_dir}; manifest: {manifest_path}"
  )


if __name__ == "__main__":
  main()
