"""Small deterministic motions and replay snapshots, independent of local runs."""

import json
from pathlib import Path

import numpy as np
import torch

from src.tasks.supermimic.config.morphology import RobotLayout
from src.tasks.supermimic.config.task_cfg import STEP_DT
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.mdp.rewards import tracking_reward
from src.tasks.supermimic.mdp.state import observe
from src.tasks.supermimic.models.kinematics import G1Kinematics


@torch.no_grad()
def make_motions(directory: Path, metadata: dict) -> tuple[Path, list[str]]:
  """Write two kinematically consistent G1 clips with different root speeds."""
  directory.mkdir(parents=True, exist_ok=True)
  layout = RobotLayout(len(metadata["joint_names"]))
  kinematics = G1Kinematics(metadata, tuple(metadata["body_names"]))
  names = ["fixture_walk", "fixture_run"]
  entries = []
  for clip, name in enumerate(names):
    frames = 65 + clip * 8
    state = torch.zeros(frames, layout.state_dim)
    state[:, 2], state[:, 3] = 0.8, 1.0
    state[:, layout.q] = metadata["default_q"].float()
    state[:, layout.v.start] = 0.05 * (clip + 1)
    state[:, 0] = torch.arange(frames) * STEP_DT * state[:, layout.v.start]
    bodies = kinematics(state)
    np.savez(
      directory / f"{name}.npz",
      joint_pos=state[:, layout.q].numpy(),
      joint_vel=state[:, layout.qd].numpy(),
      **{key: value.numpy() for key, value in bodies.items()},
      fps=np.array([50.0], dtype=np.float32),
      quaternion_order="wxyz",
      joint_names=np.array(metadata["joint_names"]),
      body_names=np.array(metadata["body_names"]),
    )
    entries.append({"file": f"{name}.npz", "frames": frames})
  manifest = directory / "manifest.json"
  manifest.write_text(
    json.dumps(
      {"schema_version": 1, "fps": 50, "quaternion_order": "wxyz", "clips": entries}
    )
  )
  return manifest, names


@torch.no_grad()
def make_replay(directory: Path, metadata: dict) -> tuple[Path, Path, list[str]]:
  """Write labeled logged-action trajectories with independent episodes per clip.

  This fixture tests data handling and supervised fitting. It is a kinematic
  sequence, not evidence of a physically stable controller.
  """
  manifest, names = make_motions(directory, metadata)
  layout = RobotLayout(len(metadata["joint_names"]))
  kinematics = G1Kinematics(metadata)
  library = MotionLibrary(manifest, kinematics.joint_names, kinematics.body_names)
  num_envs, episode_length = 4, 48
  replay = ReplayBuffer(num_envs, capacity=episode_length * 2)
  motion = torch.tensor([0, 1, 0, 1])
  context = torch.zeros(num_envs, layout.context_dim)
  context[:, 0] = torch.tensor([0.6, 0.8, 1.0, 1.2])
  default_q = metadata["default_q"].float()
  rng = torch.Generator().manual_seed(71)
  for episode in range(2):
    previous_action = torch.zeros(num_envs, layout.num_joints)
    for step in range(episode_length):
      frame = torch.full((num_envs,), step, dtype=torch.long)
      reference, next_reference = (
        library.get_frame(motion, frame),
        library.get_frame(motion, frame + 1),
      )

      def physical_state(reference_frame):
        return torch.cat(
          (
            reference_frame["body_pos_w"][:, 0],
            reference_frame["body_quat_w"][:, 0],
            reference_frame["joint_pos"],
            reference_frame["body_lin_vel_w"][:, 0],
            reference_frame["body_ang_vel_w"][:, 0],
            reference_frame["joint_vel"],
          ),
          dim=-1,
        )

      state, next_state = physical_state(reference), physical_state(next_reference)
      action = torch.randn(num_envs, layout.num_joints, generator=rng) * 0.02
      reward, _ = tracking_reward(
        next_state,
        kinematics(next_state),
        next_reference,
        action,
        previous_action,
        metadata["joint_limits"].float(),
      )
      observation = observe(state, previous_action, default_q, context)
      next_observation = observe(next_state, action, default_q, context)
      replay.append(
        {
          "state": state,
          "next_state": next_state,
          "action": action,
          "previous_action": previous_action,
          "observation": observation,
          "next_observation": next_observation,
          "clean_observation": observation,
          "next_clean_observation": next_observation,
          "context": context,
          "reward": reward,
          "motion_id": motion,
          "frame_id": frame,
          "episode_id": torch.full_like(frame, episode),
          "episode_step": frame,
          "terminated": torch.zeros(num_envs, dtype=torch.bool),
          "time_out": torch.full((num_envs,), step == episode_length - 1),
          "motion_end": torch.zeros(num_envs, dtype=torch.bool),
        }
      )
      previous_action = action
  snapshot = directory / "replay.pt"
  torch.save({"replay": replay.state_dict(), "clips": names}, snapshot)
  return manifest, snapshot, names
