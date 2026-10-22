"""Regressions for reset scoring, replay sampling, diagnostics and ablations."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.tasks.supermimic.config.task_cfg import TRACKED_BODIES
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.mdp.reset_sampler import ResetSampler
from src.tasks.supermimic.mdp.state import (
  add_sensor_noise,
  observe,
  quat_mul,
  rotvec_to_quat,
)
from src.tasks.supermimic.mdp.terminations import soft_termination
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.models.policy import MotionActor, MotionCritic
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.imagination import frozen_parameters, imagine
from src.tasks.supermimic.rl.training_logger import MetricWindow
from src.tasks.supermimic.rl.world_model_training import (
  rollout,
  world_model_loss,
)
from tests.supermimic_fixtures import make_replay
from tests.test_supermimic_foundation import transition
from tests.test_training_logger import record


class ResetScoreTests(unittest.TestCase):
  def test_perfect_tracking_is_uniform_across_timeouts_and_short_clip_tails(
    self,
  ):
    sampler = ResetSampler(torch.tensor([10001]), mode="return", ema=1.0)
    before = sampler.probabilities().clone()
    for start, steps in ((0, 500), (500, 500), (9990, 10), (9999, 1)):
      sampler.update_episode(
        0,
        torch.arange(start, start + steps),
        torch.full((steps,), 0.1),
        terminated=False,
      )
    torch.testing.assert_close(sampler.probabilities(), before, atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(
      sampler.returns, torch.full_like(sampler.returns, sampler.max_return)
    )

  def test_early_failure_is_scored_against_possible_remaining_length(self):
    sampler = ResetSampler(torch.tensor([10001]), mode="return", ema=1.0)
    sampler.update_episode(0, torch.arange(5), torch.full((5,), 0.1), terminated=True)
    self.assertLess(float(sampler.returns[0] / sampler.max_return), 0.05)
    self.assertGreater(
      float(sampler.probabilities()[0]), float(sampler.probabilities()[1])
    )
    torch.testing.assert_close(
      sampler.returns[1:],
      torch.full_like(sampler.returns[1:], sampler.max_return),
    )

  def test_shared_start_bin_aggregates_episodes_before_ema(self):
    sampler = ResetSampler(torch.tensor([1001]), mode="return", ema=0.2)
    sampler.update_episodes(
      torch.tensor([0, 0]),
      torch.tensor([0, 10]),
      torch.tensor([0.0, 0.0]),
      torch.tensor([0.0, 0.0]),
      torch.tensor([1, 1]),
      torch.tensor([True, True]),
    )
    self.assertAlmostEqual(float(sampler.returns[0]), 8.0)
    self.assertEqual(sampler.history_updates, 1)
    self.assertAlmostEqual(float(sampler.probabilities().sum()), 1.0, places=6)

  def test_legacy_sampler_scores_are_not_silently_reused(self):
    sampler = ResetSampler(torch.tensor([101]))
    state = sampler.state_dict()
    del state["schema_version"]
    with self.assertRaisesRegex(ValueError, "Legacy"):
      sampler.load_state_dict(state)


class ReplaySamplingTests(unittest.TestCase):
  def test_dense_sampling_does_not_enumerate_buffer_and_remains_uniform(self):
    replay = ReplayBuffer(4, 64, seed=9)
    for step in range(64):
      replay.append(transition(step, num_envs=4))
    with patch.object(replay, "_candidates", side_effect=AssertionError("full scan")):
      windows = replay.sample_windows(4096, 8, require_full=True)
      starts = replay.sample_starts(64)
    self.assertTrue(windows["mask"].all())
    self.assertEqual(starts["history"].shape, (64, 10, 93))
    counts = torch.bincount(windows["state"][:, 0, 0].long(), minlength=57).float()
    self.assertLess(float((counts - counts.mean()).abs().max() / counts.mean()), 0.5)

  def test_sparse_and_empty_full_windows_keep_boundary_semantics(self):
    replay = ReplayBuffer(1, 16)
    for step in range(8):
      replay.append(transition(step, done="terminated" if step == 7 else None))
    batch = replay.sample_windows(20, 8, require_full=True)
    self.assertTrue((batch["state"][:, 0, 0] == 0).all())
    self.assertTrue(batch["terminated"][:, -1].all())
    with self.assertRaises(ValueError):
      replay.sample_windows(1, 9, require_full=True)


class ModelFixTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(4)
    cls.metadata = robot_metadata()
    folder = tempfile.TemporaryDirectory()
    cls.addClassCleanup(folder.cleanup)
    manifest, snapshot, _ = make_replay(Path(folder.name), cls.metadata)
    cls.replay_state = torch.load(snapshot, weights_only=True)["replay"]
    cls.library = MotionLibrary(
      manifest, tuple(cls.metadata["joint_names"]), TRACKED_BODIES
    )

  def replay(self):
    return ReplayBuffer.from_state_dict(self.replay_state)

  def test_orientation_distinguishes_yaw_error_but_is_joint_rotation_invariant(
    self,
  ):
    batch = self.replay().sample_starts(2)
    state, action, context = (
      batch["state"],
      batch["previous_action"],
      batch["context"],
    )
    reference = self.library.get_frame(batch["motion_id"], batch["frame_id"])[
      "body_quat_w"
    ][:, 0]
    default = self.metadata["default_q"].float()
    yaw = rotvec_to_quat(torch.tensor([[0.0, 0.0, torch.pi / 2]])).expand(2, -1)
    turned = state.clone()
    turned[:, 3:7] = quat_mul(yaw, state[:, 3:7])
    base = observe(state, action, default, context, reference)
    mismatch = observe(turned, action, default, context, reference)
    together = observe(turned, action, default, context, quat_mul(yaw, reference))
    self.assertGreater(float((base[:, 64:70] - mismatch[:, 64:70]).abs().max()), 0.9)
    torch.testing.assert_close(together, base)
    torch.testing.assert_close(base[:, -29:], action)
    noisy = add_sensor_noise(base)
    torch.testing.assert_close(noisy[:, -29:], action)
    self.assertLessEqual(
      float((noisy[:, 64:70] - base[:, 64:70]).abs().max()), 0.050001
    )

  def test_analytic_termination_matches_threshold_and_has_boundary_gradient(
    self,
  ):
    batch = self.replay().sample_starts(1)
    reference = self.library.get_frame(batch["motion_id"], batch["frame_id"])
    state = batch["state"].clone()
    self.assertLess(float(soft_termination(state, reference)), 1e-4)
    state[:, 2] += 0.25
    state.requires_grad_()
    probability = soft_termination(state, reference)
    self.assertAlmostEqual(float(probability.detach()), 0.5, places=4)
    probability.sum().backward()
    self.assertGreater(float(state.grad[:, 2]), 10.0)
    state = state.detach()
    state[:, 2] += 0.1
    self.assertGreater(float(soft_termination(state, reference)), 0.99)

  def test_fast_world_loss_and_gradients_equal_full_heads(self):
    batch = self.replay().sample_windows(4, 8)
    for mode in ("learned", "analytic"):
      world = WorldModel(self.metadata, (16,), termination_mode=mode)
      gradients = []
      losses = []
      for diagnostics in (True, False):
        world.zero_grad(set_to_none=True)
        with patch.object(
          world.kinematics, "forward", wraps=world.kinematics.forward
        ) as fk:
          pred = rollout(world, batch, self.library, diagnostics=diagnostics)
        if not diagnostics:
          fk.assert_not_called()
        loss, parts = world_model_loss(pred, batch, analytic_losses=False)
        loss.backward()
        losses.append(loss.detach())
        gradients.append(
          [
            p.grad.detach().clone() if p.grad is not None else None
            for p in world.parameters()
          ]
        )
        if mode == "analytic":
          self.assertEqual(float(parts["termination"]), 0.0)
          self.assertTrue(
            all(p.grad is None for p in world.termination_head.parameters())
          )
      torch.testing.assert_close(*losses)
      for a, b in zip(*gradients):
        if a is None:
          self.assertIsNone(b)
        else:
          torch.testing.assert_close(a, b)

  def test_relative_orientation_imagination_feedback_and_objective_decomposition(
    self,
  ):
    starts = self.replay().sample_starts(2)
    library = copy.copy(self.library)
    library.data = dict(library.data)
    library.data["body_quat_w"] = library.data["body_quat_w"].clone()
    rotation = torch.zeros(len(library.data["body_quat_w"]), 3)
    rotation[:, 2] = torch.arange(len(rotation)) * 0.03
    library.data["body_quat_w"][:, 0] = rotvec_to_quat(rotation)
    ref = library.get_frame(starts["motion_id"], starts["frame_id"])
    actor = MotionActor(self.metadata["default_q"], (16,), True)
    world = WorldModel(self.metadata, (16,), "analytic", True)
    critic = MotionCritic(self.metadata["default_q"], (16,))
    obs = observe(
      starts["state"],
      starts["previous_action"],
      actor.default_q,
      starts["context"],
      ref["body_quat_w"][:, 0],
    )
    starts["history"] = obs[:, None].repeat(1, 10, 1)
    with frozen_parameters(world, critic):
      trace = imagine(
        actor,
        world,
        critic,
        library,
        starts,
        horizon=4,
        sensor_noise=False,
      )
      trace["objective"].backward()
    torch.testing.assert_close(
      trace["objective"],
      trace["reward_objective"] + trace["bootstrap_objective"],
    )
    torch.testing.assert_close(trace["history"][:, -1, -29:], trace["actions"][-1])
    self.assertTrue(
      all(
        torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None
      )
    )
    # The final observation must use the successor reference, just like the simulator.
    final_ref = library.get_frame(starts["motion_id"], starts["frame_id"] + 4)
    expected = observe(
      trace["states"][:, -1],
      trace["actions"][-1],
      actor.default_q,
      starts["context"],
      final_ref["body_quat_w"][:, 0],
    )
    torch.testing.assert_close(trace["history"][:, -1], expected)




class CollectionMetricTests(unittest.TestCase):
  def test_update_only_window_has_no_carried_episode_counts_or_reward_terms(self):
    window = MetricWindow()
    item = record(collected=False)
    item.update(
      episode_steps_sum=20,
      episode_returns_sum=1.0,
      normal_timeouts=1,
      real_body_position=0.03,
    )
    window.add(item)
    summary = window.summary()
    self.assertIsNone(summary["real_body_position"])
    self.assertIsNone(summary["mean_episode_steps"])
    self.assertEqual(summary["normal_timeouts"], 0)

  def test_episode_and_reward_terms_count_each_collection_once(self):
    window = MetricWindow()
    first = record()
    first.update(
      episode_steps_sum=20,
      episode_returns_sum=1.0,
      normal_timeouts=1,
      normal_motion_ends=0,
      real_body_position=0.01,
    )
    window.add(first)
    window.add(first | {"iteration": 2, "collected_this_iteration": False})
    window.add(
      first | {"iteration": 3, "episode_steps_sum": 40, "real_body_position": 0.03}
    )
    summary = window.summary()
    self.assertEqual(summary["mean_episode_steps"], 15.0)
    self.assertEqual(summary["normal_timeouts"], 2)
    self.assertAlmostEqual(summary["real_body_position"], 0.02)


if __name__ == "__main__":
  unittest.main()
