"""Minimal mjlab scene for collection; task signals are shared tensor functions."""

import re

import mujoco
from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from .config.robot_cfg import FOOT_COLLISION_EXPR, get_supermimic_robot_cfg
from .config.task_cfg import FoundationConfig


class FoundationMujocoCfg(MujocoCfg):
  def apply(self, model: mujoco.MjModel) -> None:
    super().apply(model)
    # Observations are reconstructed from synchronized physical state, including
    # explicit observation noise. No MuJoCo built-in sensors are consumed here.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_SENSOR


def make_env_cfg(config: FoundationConfig) -> ManagerBasedRlEnvCfg:
  foot = SceneEntityCfg("robot", geom_names=FOOT_COLLISION_EXPR)
  torso = SceneEntityCfg("robot", body_names=("torso_link",))
  # Constant ranges also expand model fields in debug/evaluation mode, ensuring
  # the context always comes from the actual applied model parameters.
  mu = (0.3, 1.2) if config.randomization else (1.0, 1.0)
  bias = (-0.01, 0.01) if config.randomization else (0.0, 0.0)
  com = {0: (-0.025, 0.025), 1: (-0.05, 0.05), 2: (-0.05, 0.05)}
  if not config.randomization:
    com = {i: (0.0, 0.0) for i in range(3)}
  robot_cfg = get_supermimic_robot_cfg(config.robot)
  joint_names = [j.name for j in robot_cfg.spec_fn().joints]
  return ManagerBasedRlEnvCfg(
    seed=config.seed,
    scene=SceneCfg(
      terrain=TerrainEntityCfg(terrain_type="plane"),
      entities={"robot": robot_cfg},
      num_envs=config.num_envs,
    ),
    actions={
      "joint_pos": JointPositionActionCfg(
        entity_name="robot",
        actuator_names=(".*",),
        scale={
          p: value
          for p, value in G1_ACTION_SCALE.items()
          if any(re.fullmatch(p, name) for name in joint_names)
        },
        use_default_offset=True,
      )
    },
    events={
      "foot_friction": EventTermCfg(
        func=dr.geom_friction,
        mode="startup",
        params={
          "asset_cfg": foot,
          "ranges": mu,
          "operation": "abs",
          "shared_random": True,
        },
      ),
      "encoder_bias": EventTermCfg(
        func=dr.encoder_bias, mode="reset", params={"bias_range": bias}
      ),
      "torso_com": EventTermCfg(
        func=dr.body_com_offset,
        mode="startup",
        params={"asset_cfg": torso, "ranges": com, "operation": "add"},
      ),
    },
    sim=SimulationCfg(
      # Falling and dancing need full-body ground contacts and adequate capacity.
      nconmax=128,
      njmax=1024,
      mujoco=FoundationMujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=config.episode_length_s,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="pelvis",
    ),
  )
