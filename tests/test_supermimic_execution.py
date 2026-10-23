"""Numerical and gradient checks for batched and compiled training kernels."""

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import TRACKED_BODIES
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.replay_buffer import ReplayBuffer
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.models.policy import (
  CRITIC_DIM,
  GOAL_DIM,
  MotionActor,
  MotionCritic,
  reference_features,
)
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.execution import TrainingKernels
from src.tasks.supermimic.rl.imagination import frozen_parameters, gae_targets, imagine
from src.tasks.supermimic.rl.world_model_training import rollout, world_model_loss
from tests.supermimic_fixtures import make_replay


def sequential_objective(actor, world, critic, library, starts, horizon=4, gamma=0.99):
  """Independent pre-batching computation, with the same new reference features."""
  state, history = starts["state"].detach(), starts["history"].detach()
  context = starts["context"]
  future = library.get_future(starts["motion_id"], starts["frame_id"], horizon + 10)
  survival = state.new_ones(len(state))
  objective = torch.zeros_like(survival)
  rewards, values, continuation = [], [], []
  for t in range(horizon):
    reference = {key: value[:, t : t + 10] for key, value in future.items()}
    with torch.no_grad():
      values.append(critic(state, reference, context))
    action = actor(history, reference)
    state = world(state, action, context)
    heads = world.task_heads(
      state,
      action,
      context,
      {key: value[:, t + 1] for key, value in future.items()},
      history[:, -1, -29:],
    )
    objective = objective + gamma**t * survival * heads["reward"]
    survive = 1 - heads["termination_probability"]
    survival = survival * survive
    history = torch.cat((history[:, 1:], heads["clean_observation"][:, None]), 1)
    rewards.append(heads["reward"])
    continuation.append(survive)
  terminal = critic(
    state, {key: value[:, horizon:] for key, value in future.items()}, context
  )
  values.append(terminal.detach())
  return {
    "objective": (objective + gamma**horizon * survival * terminal).mean(),
    "rewards": torch.stack(rewards, 1),
    "values": torch.stack(values, 1),
    "continuation": torch.stack(continuation, 1),
    "history": history,
  }


class ExecutionTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(4)
    cls.metadata = robot_metadata()
    folder = tempfile.TemporaryDirectory()
    cls.addClassCleanup(folder.cleanup)
    manifest, snapshot, _ = make_replay(Path(folder.name), cls.metadata)
    cls.library = MotionLibrary(
      manifest, tuple(cls.metadata["joint_names"]), TRACKED_BODIES
    )
    cls.replay = ReplayBuffer.from_state_dict(
      torch.load(snapshot, weights_only=True)["replay"]
    )

  def models(self, mode="analytic", relative=True, device="cpu", amp="none"):
    actor = MotionActor(self.metadata["default_q"], (16,), relative, amp).to(device)
    world = WorldModel(self.metadata, (16,), mode, relative, amp).to(device)
    critic = MotionCritic(self.metadata["default_q"], (16,), amp).to(device)
    torch.nn.init.normal_(world.dynamics[-1].weight, std=1e-4)
    return actor, world, critic

  def starts(self, actor):
    starts = self.replay.sample_starts(2, horizon=4, clip_lengths=self.library.lengths)
    if actor.relative_orientation:
      from src.tasks.supermimic.mdp.state import observe

      reference = self.library.get_frame(starts["motion_id"], starts["frame_id"])
      obs = observe(
        starts["state"],
        starts["previous_action"],
        actor.default_q,
        starts["context"],
        reference["body_quat_w"][:, 0],
      )
      starts["history"] = obs[:, None].expand(-1, 10, -1).clone()
    return starts

  def test_compact_reference_contains_root_and_joint_targets_only(self):
    ref = self.library.get_future(torch.tensor([0, 1]), torch.tensor([0, 3]))
    result = reference_features(ref, self.metadata["default_q"])
    self.assertEqual((GOAL_DIM, CRITIC_DIM), (730, 834))
    self.assertEqual(result.shape, (2, 730))
    frames = result.reshape(2, 10, 73)
    torch.testing.assert_close(frames[:, 0, :3], torch.zeros(2, 3))
    torch.testing.assert_close(frames[..., 15:44], ref["joint_pos"])
    torch.testing.assert_close(frames[..., 44:], 0.1 * ref["joint_vel"])
    # Changing redundant non-root FK targets must not change policy inputs.
    altered = {key: value.clone() for key, value in ref.items()}
    for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
      altered[key][..., 1:, :] = 123
    torch.testing.assert_close(
      result, reference_features(altered, self.metadata["default_q"])
    )

  def test_full_body_reference_retains_height_scaling_and_body_targets(self):
    actor = MotionActor(self.metadata["default_q"], (16,), reference_dim=2680)
    ref = self.library.get_future(torch.tensor([0, 1]), torch.tensor([0, 3]))
    frames = actor.reference_features(ref, actor.default_q).reshape(2, 10, 268)
    self.assertEqual(actor.encoder[0].in_features, 2680)
    torch.testing.assert_close(
      frames[..., :29], (ref["joint_pos"] - actor.default_q) / 0.3
    )
    torch.testing.assert_close(frames[..., 29:58], ref["joint_vel"] / 5)
    root = frames[..., 58:100].reshape(2, 10, 14, 3)[..., 0, :]
    torch.testing.assert_close(root[:, 0, :2], torch.zeros(2, 2), atol=1e-6, rtol=0)
    torch.testing.assert_close(root[..., 2], ref["body_pos_w"][..., 0, 2] / 0.3)
    shifted = {key: value.clone() for key, value in ref.items()}
    shifted["body_pos_w"][..., 2] += 0.3
    difference = (
      actor.reference_features(shifted, actor.default_q).reshape(2, 10, 268) - frames
    )
    torch.testing.assert_close(
      difference[..., 58:100].reshape(2, 10, 14, 3)[..., 2],
      torch.ones(2, 10, 14),
      atol=1e-6,
      rtol=1e-6,
    )

  def test_full_body_compiled_imagination_matches_sequential_actor_gradients(self):
    actor = MotionActor(self.metadata["default_q"], (16,), reference_dim=2680)
    critic = MotionCritic(self.metadata["default_q"], (16,), reference_dim=2680)
    world = WorldModel(self.metadata, (16,), "analytic")
    torch.nn.init.normal_(world.dynamics[-1].weight, std=1e-4)
    starts = self.starts(actor)
    self.assertEqual(critic.value[0].in_features, 2784)
    kernels = TrainingKernels(actor, world, critic, enabled=True, backend="aot_eager")
    with frozen_parameters(world, critic):
      expected = sequential_objective(actor, world, critic, self.library, starts)
      expected["objective"].backward()
      gradients = [p.grad.clone() for p in actor.parameters()]
      actor.zero_grad(set_to_none=True)
      actual = imagine(
        actor,
        world,
        critic,
        self.library,
        starts,
        horizon=4,
        sensor_noise=False,
        kernels=kernels,
      )
      actual["objective"].backward()
      for key, value in expected.items():
        torch.testing.assert_close(actual[key], value, atol=2e-6, rtol=2e-5)
      for parameter, gradient in zip(actor.parameters(), gradients):
        torch.testing.assert_close(parameter.grad, gradient, atol=2e-6, rtol=2e-4)

  def test_batched_rewards_values_and_actor_gradients_match_sequential(self):
    for mode, relative in (("learned", False), ("analytic", True)):
      actor, world, critic = self.models(mode, relative)
      starts = self.starts(actor)
      with self.subTest(mode=mode), frozen_parameters(world, critic):
        expected = sequential_objective(actor, world, critic, self.library, starts)
        expected["objective"].backward()
        gradients = [p.grad.clone() for p in actor.parameters()]
        actor.zero_grad(set_to_none=True)
        with patch.object(
          world.kinematics, "forward", wraps=world.kinematics.forward
        ) as fk:
          actual = imagine(
            actor, world, critic, self.library, starts, horizon=4, sensor_noise=False
          )
        self.assertEqual(fk.call_count, 1)
        self.assertEqual(fk.call_args.args[0].shape, (8, 71))
        for key, value in expected.items():
          torch.testing.assert_close(actual[key], value, atol=2e-6, rtol=2e-5)
        actual["objective"].backward()
        for parameter, gradient in zip(actor.parameters(), gradients):
          torch.testing.assert_close(parameter.grad, gradient, atol=2e-6, rtol=2e-4)

  def test_fullgraph_aot_kernels_preserve_outputs_gradients_and_checkpoint_keys(self):
    actor, world, critic = self.models()
    starts = self.starts(actor)
    keys = list(actor.state_dict())
    compiled = TrainingKernels(actor, world, critic, enabled=True, backend="aot_eager")
    with frozen_parameters(world, critic):
      expected = imagine(
        actor, world, critic, self.library, starts, horizon=2, sensor_noise=False
      )
      expected["objective"].backward()
      gradients = [p.grad.clone() for p in actor.parameters()]
      actor.zero_grad(set_to_none=True)
      actual = imagine(
        actor,
        world,
        critic,
        self.library,
        starts,
        horizon=2,
        sensor_noise=False,
        kernels=compiled,
      )
      actual["objective"].backward()
      torch.testing.assert_close(actual["objective"], expected["objective"])
      for p, gradient in zip(actor.parameters(), gradients):
        torch.testing.assert_close(p.grad, gradient, atol=2e-6, rtol=2e-4)
      torch.testing.assert_close(
        compiled.gae(actual["rewards"], actual["continuation"], actual["values"]),
        gae_targets(actual["rewards"], actual["continuation"], actual["values"]),
      )
    self.assertEqual(keys, list(actor.state_dict()))
    batch = self.replay.sample_windows(2, 8)
    for diagnostics in (False, True):
      world.zero_grad(set_to_none=True)
      prediction = rollout(
        world,
        batch,
        self.library,
        diagnostics=diagnostics,
        dynamics=compiled.world_rollout,
      )
      loss, parts = compiled.world_loss(prediction, batch)
      expected_loss, expected_parts = world_model_loss(
        prediction, batch, analytic_losses=False
      )
      torch.testing.assert_close(loss, expected_loss)
      for key in parts:
        torch.testing.assert_close(parts[key], expected_parts[key])
      loss.backward()
      self.assertTrue(
        all(
          torch.isfinite(p.grad).all() for p in world.parameters() if p.grad is not None
        )
      )
    features = actual["critic_features"]
    target = actual["values"][:, :-1]
    mask = actual["mask"]
    value_loss = compiled.critic_loss(
      compiled.critic_value(features).squeeze(-1), target, mask
    )
    value_loss.backward()
    self.assertTrue(any(p.grad is not None for p in critic.parameters()))

  def test_execution_defaults_and_amp_validation(self):
    self.assertTrue(OnlineConfig().compile)
    self.assertEqual(OnlineConfig().amp, "bf16")
    with self.assertRaisesRegex(ValueError, "amp"):
      OnlineConfig(amp="fp16")

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
  def test_cuda_bf16_mlp_keeps_physics_and_gradients_fp32(self):
    actor, world, _ = self.models(device="cuda", amp="bf16")
    if not torch.cuda.is_bf16_supported():
      self.skipTest("GPU does not support BF16")
    state = torch.zeros(2, 71, device="cuda")
    state[:, 3] = 1
    observed = []
    handle = world.dynamics[-1].register_forward_hook(
      lambda _, __, out: observed.append(out.dtype)
    )
    result = world(
      state, torch.zeros(2, 29, device="cuda"), torch.zeros(2, 33, device="cuda")
    )
    handle.remove()
    self.assertEqual(observed, [torch.bfloat16])
    self.assertEqual(result.dtype, torch.float32)
    result.square().sum().backward()
    self.assertTrue(
      all(
        p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all()
        for p in world.parameters()
        if p.grad is not None
      )
    )
    action = actor.forward_features(
      torch.zeros(2, 10, 99, device="cuda"), torch.zeros(2, 730, device="cuda")
    )
    self.assertEqual(action.dtype, torch.float32)

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
  def test_cuda_inductor_bf16_training_kernels(self):
    if not torch.cuda.is_bf16_supported():
      self.skipTest("GPU does not support BF16")
    actor, world, critic = self.models(device="cuda", amp="bf16")
    library = copy.copy(self.library)
    library.device = "cuda"
    library.lengths = library.lengths.cuda()
    library.offsets = library.offsets.cuda()
    library.data = {key: value.cuda() for key, value in library.data.items()}
    starts = {key: value.cuda() for key, value in self.starts(self.models()[0]).items()}
    kernels = TrainingKernels(actor, world, critic, enabled=True)
    with frozen_parameters(world, critic):
      eager = imagine(
        actor, world, critic, library, starts, horizon=2, sensor_noise=False
      )
      compiled = imagine(
        actor,
        world,
        critic,
        library,
        starts,
        horizon=2,
        sensor_noise=False,
        kernels=kernels,
      )
      torch.testing.assert_close(
        compiled["objective"], eager["objective"], atol=2e-3, rtol=2e-2
      )
      compiled["objective"].backward()
      self.assertTrue(
        all(
          torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None
        )
      )
    batch = {
      key: value.cuda() for key, value in self.replay.sample_windows(2, 8).items()
    }
    prediction = rollout(world, batch, library, dynamics=kernels.world_rollout)
    loss, _ = kernels.world_loss(prediction, batch)
    loss.backward()
    targets = kernels.gae(
      compiled["rewards"], compiled["continuation"], compiled["values"]
    )
    critic_loss = kernels.critic_loss(
      kernels.critic_value(compiled["critic_features"]).squeeze(-1),
      targets,
      compiled["mask"],
    )
    critic_loss.backward()
    self.assertTrue(torch.isfinite(loss) and torch.isfinite(critic_loss))
    self.assertEqual(compiled["states"].dtype, torch.float32)
    self.assertEqual(prediction["next_state"].dtype, torch.float32)
    self.assertFalse(any("_orig_mod" in key for key in actor.state_dict()))


if __name__ == "__main__":
  unittest.main()
