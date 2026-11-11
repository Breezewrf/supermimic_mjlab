"""Filter/convert BONES-SEED G1 CSVs into 50 Hz NPZ motions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing
import os
import sys
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/data"))
from convert_lafan_to_npz import (
  CSV_JOINT_NAMES,
  load_robot,
  motion_to_arrays,
  resample_motion,
  save_npz,
)

from src.tasks.supermimic.config.morphology import robot_layout, tracked_bodies

FPS = 50.0
KEYWORDS = (
  "bed",
  "bike",
  "chair",
  "climb",
  "com_up_50cm",
  "sitting",
  "step_on",
  "seat",
  "table",
  "_sit_",
  "sit_",
  "ladder",
  "crutch",
  "_bed_",
  "_ride_",
  "scooter",
  "stepdown",
  "acrobatics_",
  "box_hspu",
  "cartwheel",
  "50cm_box_",
  "on_box",
  "fall_from",
  "handstand_ff_",
  "on_1m",
  "form_box",
  "off_1m",
  "230m",
  "jump_over_obstacle_",
  "lift_crate_come_up_",
  "jump_to_shoulder_roll",
  "kozak_dance",
  "stair",
  "handstand",
  "box_jump",
  "monkey_jump",
  "safety_roll",
  "box_dips",
  "walking_on_edge",
  "push_obstacle",
  "jump_off_50cm",
  "look_over_fence",
  "_box_",
  "box_",
)


def read_csv(path):
  """Read centimeter/degree CSV values and resample to meters/radians at 50 Hz."""
  with path.open(newline="") as f:
    reader = csv.DictReader(f)
    names = reader.fieldnames or []
    required = (
      ["Frame"]
      + [f"root_translate{x}" for x in "XYZ"]
      + [f"root_rotate{x}" for x in "XYZ"]
      + [f"{n}_dof" for n in CSV_JOINT_NAMES]
    )
    if len(names) != 36 or any(n not in names for n in required):
      raise ValueError("Unexpected CSV schema")
    raw = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
  if (
    raw.shape[1] != 36
    or len(raw) < 2
    or (not np.isfinite(raw).all())
    or (not np.all(np.diff(raw[:, 0]) == 1))
  ):
    raise ValueError("Invalid CSV frames or values")
  motion = np.empty((len(raw), 36))
  motion[:, :3] = raw[:, 1:4] / 100
  motion[:, 3:7] = Rotation.from_euler("xyz", raw[:, 4:7], degrees=True).as_quat()[
    :, [3, 0, 1, 2]
  ]
  motion[:, 7:] = np.deg2rad(raw[:, 7:])
  return (resample_motion(motion, 120, FPS, "wxyz"), len(raw))


def longest(mask):
  """Return the start, exclusive end, and length of the longest true run."""
  best = (0, 0, 0)
  start = None
  for frame, active in enumerate(np.r_[mask, False]):
    if active and start is None:
      start = frame
    elif not active and start is not None:
      if frame - start > best[2]:
        best = (start, frame, frame - start)
      start = None
  return best


def build_reference_states(model, motion, joint_names):
  """Map CSV joints into MuJoCo state order and differentiate adjacent poses."""
  free_joint = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
  root_address = model.jnt_qposadr[free_joint]
  joint_addresses = np.array(
    [model.jnt_qposadr[model.joint(j).id] for j in joint_names]
  )
  csv_joint_indices = [CSV_JOINT_NAMES.index(j) for j in joint_names]
  positions = np.tile(model.qpos0, (len(motion), 1))
  positions[:, root_address : root_address + 7] = motion[:, :7]
  positions[:, joint_addresses] = motion[:, 7:][:, csv_joint_indices]
  velocities = np.zeros((len(positions), model.nv))
  velocity_buffer = np.zeros(model.nv)
  for i in range(len(positions)):
    if i < len(positions) - 1:
      mujoco.mj_differentiatePos(
        model, velocity_buffer, 1 / FPS, positions[i], positions[i + 1]
      )
      forward_velocity = velocity_buffer.copy()
    if i:
      mujoco.mj_differentiatePos(
        model, velocity_buffer, 1 / FPS, positions[i], positions[i - 1]
      )
      backward_velocity = -velocity_buffer.copy()
    velocities[i] = (
      forward_velocity
      if i == 0
      else backward_velocity
      if i == len(positions) - 1
      else (forward_velocity + backward_velocity) / 2
    )
  return (positions, velocities)


def measure_geometry(model, positions, velocities, body_names):
  """Measure lowest links and foot distances during upright stationary frames."""
  data = mujoco.MjData(model)
  floor = model.geom("floor").id
  feet = {
    s: [
      g
      for g in range(model.ngeom)
      if model.geom_bodyid[g] == model.body(s + "_ankle_roll_link").id
      and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    for s in ("left", "right")
  }
  body_ids = [model.body(n).id for n in body_names]
  lowest_heights = []
  foot_distances = []
  for i in range(len(positions)):
    data.qpos[:] = positions[i]
    data.qvel[:] = velocities[i]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    lowest_heights.append(data.xpos[1:, 2].min())
    pelvis = model.body("pelvis").id
    is_upright = data.xmat[pelvis].reshape(3, 3)[2, 2] >= math.cos(math.radians(20))
    is_standing = 0.65 <= data.xpos[pelvis, 2] <= 1.0
    is_slow = np.linalg.norm(data.cvel[body_ids, 3:], axis=1).max() <= 0.15
    is_not_rotating = np.linalg.norm(data.cvel[body_ids, :3], axis=1).max() <= 0.5
    if is_upright and is_standing and is_slow and is_not_rotating:
      foot_distances.append(
        [
          min(mujoco.mj_geomDistance(model, data, g, floor, 10, None) for g in feet[s])
          for s in ("left", "right")
        ]
      )
    else:
      foot_distances.append([0, 0])
  return (lowest_heights, foot_distances)


def convert(source_path, output_path, model, joint_names, body_names):
  """Filter one motion and save its NPZ only when all feasibility checks pass."""
  (motion, input_frames) = read_csv(source_path)
  (positions, velocities) = build_reference_states(model, motion, joint_names)
  (lowest_heights, foot_distances) = measure_geometry(
    model, positions, velocities, body_names
  )
  lowest_heights = np.asarray(lowest_heights)
  foot_distances = np.asarray(foot_distances)
  elevated = longest(lowest_heights > 0.2)
  floating = longest((foot_distances[:, 0] > 0.02) & (foot_distances[:, 1] > 0.02))
  reasons = []
  if elevated[2] >= 30:
    reasons.append("elevated_support")
  if lowest_heights.min() < -0.12 or np.mean(lowest_heights < -0.08) > 0.8:
    reasons.append("ground_penetration")
  if floating[2] >= 25:
    reasons.append("floating_feet")
  result = {
    "status": "filtered" if reasons else "keep",
    "reason": reasons,
    "input_frames": input_frames,
    "frames": len(positions),
    "zmin_min_m": float(lowest_heights.min()),
    "zmin_below_minus_008_fraction": float(np.mean(lowest_heights < -0.08)),
    "elevated_frames": elevated[2],
    "floating_frames": floating[2],
  }
  if not reasons:
    arrays = motion_to_arrays(model, joint_names, body_names, motion, FPS)
    arrays.update(
      {
        "fps": np.array([FPS], np.float32),
        "source_file": np.asarray(source_path.as_posix()),
        "source_sha256": np.asarray(
          hashlib.sha256(source_path.read_bytes()).hexdigest()
        ),
        "input_fps": np.array([120], np.float32),
        "input_rotation_format": np.asarray("euler_xyz_degrees"),
        "schema_version": np.array([1], np.int32),
      }
    )
    save_npz(output_path, arrays)
  return result


_WORKER_CONTEXT = None


def initialize_worker(root, output_path, robot, keywords):
  """Compile one independent MuJoCo model per process and reuse it across clips."""
  global _WORKER_CONTEXT
  model, joints, _, _ = load_robot(model_xml=robot_layout(robot).xml, robot=robot)
  _WORKER_CONTEXT = root, output_path, keywords, model, joints, tracked_bodies(robot)


def process_clip(source_path):
  """Convert one clip; the parent alone commits decisions and removes CSVs."""
  root, output_path, keywords, model, joints, bodies = _WORKER_CONTEXT
  relative_path = source_path.relative_to(root).as_posix()
  try:
    stat = source_path.stat()
    keyword = next((k for k in keywords if k in source_path.stem.lower()), None)
    destination = Path("g1/npz") / Path(relative_path).relative_to(
      "g1/csv"
    ).with_suffix(".npz")
    if keyword:
      result = {"status": "filtered", "reason": ["name"], "keyword": keyword}
    else:
      result = convert(source_path, output_path / destination, model, joints, bodies)
      result["output"] = destination.as_posix() if result["status"] == "keep" else None
    result.update(source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns)
  except Exception as error:  # noqa: BLE001 - retain sources when individual clips fail
    result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
  result["source"] = relative_path
  return result


def process_clips(paths, workers, context):
  """Yield completed results with at most two queued tasks per worker."""
  if workers == 1:
    initialize_worker(*context)
    yield from map(process_clip, paths)
    return
  with ProcessPoolExecutor(
    max_workers=workers,
    mp_context=multiprocessing.get_context("spawn"),
    initializer=initialize_worker,
    initargs=context,
  ) as executor:
    iterator = iter(paths)
    pending = set()
    exhausted = False
    while pending or not exhausted:
      while not exhausted and len(pending) < workers * 2:
        path = next(iterator, None)
        if path is None:
          exhausted = True
        else:
          pending.add(executor.submit(process_clip, path))
      if pending:
        completed, pending = wait(pending, return_when=FIRST_COMPLETED)
        for future in completed:
          yield future.result()


def main():
  """Parse paths, resume recorded decisions, and write the loader manifest."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("input", type=Path)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument(
    "--override",
    action="store_true",
    help="Remove each source CSV after its decision is saved",
  )
  parser.add_argument("--robot", choices=("g1_29dof", "g1_23dof"), default="g1_29dof")
  parser.add_argument("--add-keywords", nargs="*", default=[])
  parser.add_argument(
    "--workers",
    type=int,
    default=min(8, os.cpu_count() or 1),
    help="Independent conversion processes (default: up to 8); use 1 for serial",
  )
  args = parser.parse_args()
  if args.workers < 1:
    parser.error("--workers must be at least 1")
  root = args.input.resolve()
  csv_root = root / "g1/csv"
  if not csv_root.is_dir():
    parser.error(f"Missing {csv_root}")
  output_path = (
    root
    if args.override
    else args.output_dir.resolve()
    if args.output_dir
    else root.with_name(root.name + "_filtered")
  )
  if output_path == root and (not args.override):
    parser.error("Use --override to write into the input dataset")
  output_path.mkdir(parents=True, exist_ok=True)
  state = output_path / ".bones_seed_filter.json"
  report = output_path / "filter_decisions.jsonl"
  config = {
    "source": str(root),
    "robot": args.robot,
    "fps_in": 120,
    "fps_out": FPS,
    "keywords": list(KEYWORDS) + args.add_keywords,
    "thresholds": {
      "z_high_m": 0.2,
      "z_high_frames": 30,
      "z_penetration_m": -0.12,
      "z_penetration_fraction_m": -0.08,
      "z_penetration_fraction": 0.8,
      "foot_distance_m": 0.02,
      "floating_frames": 25,
      "standing_height_m": [0.65, 1.0],
      "upright_deg": 20,
      "stationary_speed_mps": 0.15,
      "stationary_omega_rps": 0.5,
    },
  }
  previous = {}
  if state.exists():
    saved = json.loads(state.read_text())
    if saved["config"] != config:
      parser.error("Resume settings differ from existing state")
    if report.exists():
      with report.open() as f:
        for line in f:
          row = json.loads(line)
          previous[row["source"]] = row
  elif output_path != root and any(output_path.iterdir()):
    parser.error(f"Refusing non-empty output: {output_path}")
  else:
    state.write_text(json.dumps({"config": config}, ensure_ascii=False))
  paths = sorted(csv_root.rglob("*.csv"))
  results = dict(previous)
  keywords = tuple(k.lower() for k in config["keywords"])
  pending_paths = []
  for source_path in paths:
    relative_path = source_path.relative_to(root).as_posix()
    recorded = previous.get(relative_path)
    if recorded and recorded["status"] in ("keep", "filtered"):
      if (
        recorded["status"] == "keep"
        and not (output_path / recorded["output"]).is_file()
      ):
        parser.error(f"Missing output for {relative_path}")
      if args.override:
        stat = source_path.stat()
        if (
          stat.st_size != recorded["source_size"]
          or stat.st_mtime_ns != recorded["source_mtime_ns"]
        ):
          parser.error(f"Source changed since its recorded decision: {relative_path}")
        source_path.unlink()
    else:
      pending_paths.append(source_path)
  context = (root, output_path, args.robot, keywords)
  completed = process_clips(pending_paths, args.workers, context)
  for result in tqdm(
    completed, total=len(pending_paths), desc="Filter + convert", unit="clip"
  ):
    relative_path = result["source"]
    source_path = root / relative_path
    if args.override and result["status"] != "error":
      stat = source_path.stat()
      if (
        stat.st_size != result["source_size"]
        or stat.st_mtime_ns != result["source_mtime_ns"]
      ):
        result.update(status="error", error="source changed during conversion")
    results[relative_path] = result
    with report.open("a") as log:
      log.write(json.dumps(result, ensure_ascii=False) + "\n")
      log.flush()
      os.fsync(log.fileno())
    if args.override and result["status"] != "error":
      source_path.unlink()
  counts = {
    s: sum(result["status"] == s for result in results.values())
    for s in ("keep", "filtered", "error")
  }
  print(
    f"Kept {counts['keep']}; filtered {counts['filtered']}; errors {counts['error']}. Report: {report}"
  )
  if counts["error"]:
    raise SystemExit(1)
  manifest = {
    "schema_version": 1,
    "fps": FPS,
    "quaternion_order": "wxyz",
    "robot": args.robot,
    "input_fps": 120,
    "source_dataset": "BONES-SEED G1 CSV",
    "clips": [
      {"file": result["output"], "frames": result["frames"]}
      for (_, result) in sorted(results.items())
      if result["status"] == "keep"
    ],
  }
  tmp = output_path / "manifest.json.tmp"
  tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
  tmp.replace(output_path / "manifest.json")
  print(f"Manifest ready: {output_path / 'manifest.json'}")


if __name__ == "__main__":
  main()
