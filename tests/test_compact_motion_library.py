"""Compact reference reconstruction, bounded cache and training gradients."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.tasks.supermimic.config.morphology import tracked_bodies
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.mdp.velocity_commands import VelocityCommands
from src.tasks.supermimic.config.velocity_cfg import VelocityConfig
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.models.policy import MotionActor, MotionCritic
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.rl.imagination import imagine, frozen_parameters
from src.tasks.supermimic.rl.world_model_training import rollout, world_model_loss
from tests.supermimic_fixtures import make_motions, make_replay


class CompactMotionTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(4)

  def test_variants_boundaries_cache_and_root(self):
    for robot in ("g1_29dof", "g1_23dof"):
      metadata = robot_metadata(robot)
      with tempfile.TemporaryDirectory() as folder:
        manifest, _ = make_motions(Path(folder), metadata)
        args = (manifest, tuple(metadata["joint_names"]), tracked_bodies(robot))
        full = MotionLibrary(*args)
        for capacity in (0, 1, 200):
          compact = MotionLibrary(*args, storage="compact", fk_cache_frames=capacity)
          motion = torch.tensor([[0, 1, 0], [1, 0, 1]])
          frame = torch.tensor([[63, 71, 0], [4, 8, 4]])
          expected = full.get_future(motion, frame, 12)
          torch.testing.assert_close(
            compact.compact_root_pos, full.root_fields()[0], rtol=0, atol=0
          )
          for _ in range(2):
            actual = compact.get_future(motion, frame, 12)
            for key in expected:
              if key in ("joint_pos", "joint_vel"):
                torch.testing.assert_close(actual[key], expected[key].half().float())
              else:
                torch.testing.assert_close(
                  actual[key], expected[key], atol=2e-4, rtol=2e-4
                )
              self.assertFalse(actual[key].requires_grad)
          torch.testing.assert_close(
            compact.root_state(motion, frame),
            full.root_state(motion, frame),
            atol=3e-5,
            rtol=2e-5,
          )
          torch.testing.assert_close(
            VelocityCommands(VelocityConfig(), compact).reference_commands,
            VelocityCommands(VelocityConfig(), full).reference_commands,
            atol=3e-5,
            rtol=2e-5,
          )
          self.assertFalse(compact.data)
          self.assertEqual(compact.compact_root_pos.dtype, torch.float32)
          self.assertEqual(compact.compact_rest.dtype, torch.float16)
          self.assertEqual(
            compact.compact_rest.shape[1], 10 + 2 * len(metadata["joint_names"])
          )
        compact = MotionLibrary(*args, storage="compact", fk_cache_frames=200)
        with patch.object(
          compact.kinematics, "forward", wraps=compact.kinematics.forward
        ) as fk:
          compact.get_frame(motion, frame)
          count = fk.call_count
          compact.get_frame(motion, frame)
          self.assertEqual(fk.call_count, count)

  def test_world_update_and_actor_gradient(self):
    metadata = robot_metadata()
    with tempfile.TemporaryDirectory() as folder:
      manifest, snapshot, _ = make_replay(Path(folder), metadata)
      library = MotionLibrary(
        manifest,
        tuple(metadata["joint_names"]),
        tracked_bodies(),
        storage="compact",
        fk_cache_frames=16,
      )
      replay = ReplayBuffer.from_state_dict(
        torch.load(snapshot, weights_only=True)["replay"]
      )
      world = WorldModel(metadata, (16,))
      actor = MotionActor(metadata["default_q"], (16,))
      critic = MotionCritic(metadata["default_q"], (16,))
      batch = replay.sample_windows(4, 8)
      loss, _ = world_model_loss(rollout(world, batch, library), batch)
      loss.backward()
      torch.optim.Adam(world.parameters()).step()
      world.zero_grad(set_to_none=True)
      starts = replay.sample_starts(4, horizon=4, clip_lengths=library.lengths)
      with frozen_parameters(world, critic):
        result = imagine(
          actor, world, critic, library, starts, horizon=4, sensor_noise=False
        )
        (-result["objective"]).backward()
      self.assertTrue(
        any(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.parameters())
      )
      self.assertTrue(all(p.grad is None for p in world.parameters()))

  def test_fk_chunk_limit(self):
    from src.tasks.supermimic.data.motion_library import ReferenceFKCache
    from src.tasks.supermimic.models.kinematics import G1Kinematics

    metadata = robot_metadata()
    fk = G1Kinematics(metadata)
    states = torch.zeros(8200, 71)
    states[:, 3] = 1
    states[:, 7:36] = metadata["default_q"]
    cache = ReferenceFKCache(fk, 32, "cpu")
    with patch.object(fk, "forward", wraps=fk.forward) as call:
      result = cache.get(torch.arange(8200), states[:, :3], states[:, 3:].half())
      self.assertEqual(call.call_count, 2)
      self.assertTrue(
        all(args.args[0].shape[0] <= 8192 for args in call.call_args_list)
      )
    expected = fk(torch.cat((states[:, :3], states[:, 3:].half().float()), dim=-1))
    for key in expected:
      torch.testing.assert_close(result[key], expected[key])
