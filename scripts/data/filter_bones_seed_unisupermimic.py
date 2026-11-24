"""Filter BONES-SEED with Uni-SuperMimic rules and export SuperMimic NPZ motions."""

import mujoco
import numpy as np
from bones_seed_support import SUPPORT_KEYWORDS, WALL_JUMP_KEYWORD
from filter_bones_seed import FPS, build_reference_positions, main

# Rules from Motphys/Uni-SuperMimic e8da48f, src/unisupermimic/data/boneseed.py.
KEYWORDS = tuple(
  sorted(
    (
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
      "box_hspu",
      "50cm_box_",
      "on_box",
      "fall_from",
      "on_1m",
      "form_box",
      "off_1m",
      "230m",
      "jump_over_obstacle_",
      "lift_crate_come_up_",
      "stair",
      "box_jump",
      "box_dips",
      "walking_on_edge",
      "push_obstacle",
      "acrobatics_",
      "cartwheel",
      "handstand",
      "handstand_ff_",
      "safety_roll",
      "kozak_dance",
      "jump_to_shoulder_roll",
      "monkey_jump",
      "_box_",
      "box_",
      "_stair",
      "jump_off_50cm",
      "off_50cm",
      WALL_JUMP_KEYWORD,
      "fence",
      "look_over",
    )
  )
) + SUPPORT_KEYWORDS
THRESHOLDS = {
  "z_high_m": 0.20,
  "z_high_frames": 30,
  "z_penetration_m": -0.12,
  "z_penetration_fraction_m": -0.08,
  "z_penetration_fraction": 0.8,
  "foot_distance_m": 0.02,
  "floating_frames": 25,
  "standing_height_m": [0.60, 0.85],
  "upright_gz": 0.9,
  "stationary_pelvis_speed_mps": 0.1,
  "stationary_joint_velocity_rms_radps": 0.5,
}


def longest_run(mask):
  transitions = np.flatnonzero(np.diff(np.r_[False, mask, False]))
  return int((transitions[1::2] - transitions[::2]).max(initial=0))


def foot_sole_heights(model, data, poses, frames):
  foot_bodies = [model.body(f"{side}_ankle_roll_link").id for side in ("left", "right")]
  colliding = (model.geom_contype != 0) | (model.geom_conaffinity != 0)
  geoms = np.flatnonzero(np.isin(model.geom_bodyid, foot_bodies) & colliding)
  geom_types = model.geom_type[geoms]
  capsules = geom_types == int(mujoco.mjtGeom.mjGEOM_CAPSULE)
  spheres = geom_types == int(mujoco.mjtGeom.mjGEOM_SPHERE)
  if not len(geoms) or not (spheres | capsules).all():
    raise ValueError("Foot collision geometry must contain spheres or capsules")
  radius = model.geom_size[geoms, 0]
  half_length = model.geom_size[geoms, 1] * capsules
  heights = np.empty(len(frames))
  for index, frame in enumerate(frames):
    # Uni reconstructs foot poses from the float32 converted NPZ.
    data.qpos[:] = poses[frame].astype(np.float32)
    mujoco.mj_kinematics(model, data)
    axis_z = data.geom_xmat[geoms, 8]
    heights[index] = (
      data.geom_xpos[geoms, 2] - half_length * abs(axis_z) - radius
    ).min()
  return heights


def check_feasibility(model, motion, joint_names, _body_names):
  """Apply Uni's rule order to the selected G1 model and 50 Hz poses."""
  cfg = THRESHOLDS
  poses = build_reference_positions(model, motion, joint_names)
  pelvis = model.body("pelvis").id
  data = mujoco.MjData(model)
  zmin = np.empty(len(poses), dtype=np.float32)
  pelvis_z = np.empty_like(zmin)
  pelvis_quat = np.empty((len(poses), 4), dtype=np.float32)
  for i, pose in enumerate(poses):
    data.qpos[:] = pose
    mujoco.mj_kinematics(model, data)
    zmin[i] = data.xpos[1:, 2].min()
    pelvis_z[i] = data.xpos[pelvis, 2]
    pelvis_quat[i] = data.xquat[pelvis]

  elevated = longest_run(zmin > cfg["z_high_m"])
  zmin_min = float(zmin.min())
  penetration_fraction = float(np.mean(zmin < cfg["z_penetration_fraction_m"]))
  result = {
    "reason": [],
    "zmin_min_m": zmin_min,
    "zmin_below_minus_008_fraction": penetration_fraction,
    "elevated_frames": elevated,
    "floating_frames": None,
  }
  if elevated >= cfg["z_high_frames"]:
    result["reason"] = ["elevated_support"]
    return result
  # Compare on the float32 pose values, as Uni does on the converted NPZ.
  if (zmin < cfg["z_penetration_m"]).any() or penetration_fraction > cfg[
    "z_penetration_fraction"
  ]:
    result["reason"] = ["ground_penetration"]
    return result

  upright = (
    1 - 2 * (pelvis_quat[:, 1] ** 2 + pelvis_quat[:, 2] ** 2) > cfg["upright_gz"]
  )
  low, high = cfg["standing_height_m"]
  standing = (pelvis_z > low) & (pelvis_z < high)
  joint_addresses = [model.jnt_qposadr[model.joint(name).id] for name in joint_names]
  # The free root is the pelvis link origin, so its translation derivative is
  # the same world-frame pelvis velocity that Uni reads from its converted NPZ.
  pelvis_velocity = np.gradient(motion[:, :3], 1 / FPS, axis=0).astype(np.float32)
  joint_velocity = np.gradient(poses[:, joint_addresses], 1 / FPS, axis=0).astype(
    np.float32
  )
  still = (
    np.linalg.norm(pelvis_velocity, axis=1) < cfg["stationary_pelvis_speed_mps"]
  ) & (
    np.sqrt((joint_velocity**2).mean(axis=1))
    < cfg["stationary_joint_velocity_rms_radps"]
  )
  gate = upright & standing & still
  result["floating_frames"] = 0
  if longest_run(gate) >= cfg["floating_frames"]:
    frames = np.flatnonzero(gate)
    floating = gate.copy()
    floating[frames] = (
      foot_sole_heights(model, data, poses, frames) > cfg["foot_distance_m"]
    )
    result["floating_frames"] = longest_run(floating)
    if result["floating_frames"] >= cfg["floating_frames"]:
      result["reason"] = ["floating_feet"]
  return result


if __name__ == "__main__":
  main(
    keywords=KEYWORDS,
    thresholds=THRESHOLDS,
    feasibility=check_feasibility,
    ruleset="unisupermimic-e8da48f",
    description=__doc__,
  )
