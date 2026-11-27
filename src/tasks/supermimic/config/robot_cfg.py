"""G1 physical configuration shared by simulation and model fingerprints."""

import re
from copy import deepcopy
from functools import partial
from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from .morphology import robot_layout

G1_XML = Path(__file__).resolve().parents[3] / "assets/robots/g1/g1_29dof_rev_1_0.xml"
FOOT_COLLISION_EXPR = r"^(left|right)_foot[1-4]_collision$"
SELF_COLLISION_FORCE_THRESHOLD = 10.0
COLLISION_SUBSTEPS = 4


def self_collision_excludes(robot: str = "g1_29dof"):
  """Unitree G1 exclusions, using each variant's actual wrist body names."""
  wrist = "wrist_roll_rubber_hand" if robot == "g1_23dof" else "wrist_pitch_link"
  return tuple(
    (f"{side}_elbow_link", f"{side}_{wrist}") for side in ("left", "right")
  ) + tuple(("pelvis", f"{side}_hip_roll_link") for side in ("left", "right"))


def get_supermimic_spec(
  robot: str = "g1_29dof", *, self_collisions: bool = False
) -> mujoco.MjSpec:
  """Adapt the vendored RoboJuDo-Plus asset to the mjlab training scene."""
  xml = G1_XML if robot == "g1_29dof" else robot_layout(robot).xml
  spec = mujoco.MjSpec.from_file(str(xml))
  update_assets(spec.assets, xml.parent / "meshes", spec.meshdir)
  # The scene supplies its own terrain; Entity supplies the training PD motors.
  for geom in list(spec.worldbody.geoms):
    spec.delete(geom)
  for light in list(spec.lights):
    spec.delete(light)
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  # Source geoms are unnamed. Name them without changing their geometry or
  # enabling visual-only meshes, so collision filters and friction DR can select
  # the original collision shapes consistently.
  for body in spec.bodies:
    foot_index = 0
    for index, geom in enumerate(body.geoms):
      if geom.contype or geom.conaffinity:
        if body.name in ("left_ankle_roll_link", "right_ankle_roll_link"):
          foot_index += 1
          side = body.name.split("_", 1)[0]
          geom.name = f"{side}_foot{foot_index}_collision"
        else:
          geom.name = f"{body.name}_{index}_collision"
      else:
        geom.name = f"{body.name}_{index}_visual"
  if self_collisions:
    existing = {frozenset((e.bodyname1, e.bodyname2)) for e in spec.excludes}
    for first, second in self_collision_excludes(robot):
      if spec.body(first) is None or spec.body(second) is None:
        raise ValueError(f"Self-collision exclusion body missing: {first}, {second}")
      if frozenset((first, second)) not in existing:
        spec.add_exclude(bodyname1=first, bodyname2=second)
  return spec


def get_supermimic_robot_cfg(robot: str = "g1_29dof", *, self_collisions=False):
  """Retain ground contacts; optionally enable self contacts with exclusions."""
  config = get_g1_robot_cfg()
  layout = robot_layout(robot)
  config.spec_fn = (
    get_supermimic_spec
    if layout.num_joints == 29 and not self_collisions
    else partial(get_supermimic_spec, robot, self_collisions=self_collisions)
  )
  config.articulation = deepcopy(config.articulation)
  if layout.num_joints == 23:
    names = [j.name for j in config.spec_fn().joints]
    actuators = []
    for actuator in config.articulation.actuators:
      actuator.target_names_expr = tuple(
        p for p in actuator.target_names_expr if any(re.fullmatch(p, n) for n in names)
      )
      if actuator.target_names_expr:
        actuators.append(actuator)
    config.articulation.actuators = tuple(actuators)
  config.collisions = (
    CollisionCfg(
      geom_names_expr=(".*_collision",),
      contype=int(self_collisions),
      conaffinity=1,
      condim={FOOT_COLLISION_EXPR: 3, ".*_collision": 1},
      priority={FOOT_COLLISION_EXPR: 1},
    ),
  )
  return config
