"""G1 physical configuration shared by simulation and model fingerprints."""

from copy import deepcopy
from pathlib import Path

import mujoco
from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

G1_XML = Path(__file__).resolve().parents[3] / "assets/robots/g1/g1_29dof_rev_1_0.xml"
FOOT_COLLISION_EXPR = r"^(left|right)_foot[1-4]_collision$"


def get_supermimic_spec() -> mujoco.MjSpec:
  """Adapt the vendored RoboJuDo-Plus asset to the mjlab training scene."""
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  update_assets(spec.assets, G1_XML.parent / "meshes", spec.meshdir)
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
  return spec


def get_supermimic_robot_cfg():
  """Retain full-body ground contacts while disabling robot self contacts."""
  config = get_g1_robot_cfg()
  config.spec_fn = get_supermimic_spec
  config.articulation = deepcopy(config.articulation)
  config.collisions = (
    CollisionCfg(
      geom_names_expr=(".*_collision",),
      contype=0,
      conaffinity=1,
      condim={FOOT_COLLISION_EXPR: 3, ".*_collision": 1},
      priority={FOOT_COLLISION_EXPR: 1},
    ),
  )
  return config
