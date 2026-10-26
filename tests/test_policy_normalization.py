"""RunningNorm moments, frozen BPTT statistics, and normalized policy exports."""

import copy
import tempfile
import unittest
from pathlib import Path

import torch

from src.tasks.supermimic.config.task_cfg import TRACKED_BODIES
from src.tasks.supermimic.data.motion_library import MotionLibrary
from src.tasks.supermimic.data.world_model_data import file_sha256
from src.tasks.supermimic.models.kinematics import robot_metadata
from src.tasks.supermimic.models.normalization import RunningNorm
from src.tasks.supermimic.models.policy import MotionActor, MotionCritic
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.checkpoints import export_policy
from src.tasks.supermimic.rl.execution import TrainingKernels
from src.tasks.supermimic.rl.imagination import frozen_parameters, imagine
from src.tasks.supermimic.rl.inference import load_actor
from tests.supermimic_fixtures import make_motions
from tests.test_supermimic_online import make_starts


class RunningNormTests(unittest.TestCase):
  def test_streamed_population_moments_match_combined_data(self):
    data = torch.tensor(
      [[1.0, 4.0, 2.0], [3.0, 4.0, 6.0], [8.0, 4.0, 3.0], [-2.0, 4.0, 5.0]]
    )
    norm = RunningNorm(3)
    norm.update(data[:1])
    norm.update(data[1:])
    torch.testing.assert_close(norm.mean, data.double().mean(0))
    torch.testing.assert_close(norm.var, data.double().var(0, unbiased=False))
    self.assertEqual(int(norm.count), 4)
    expected = (data - data.mean(0)) / (data.std(0, unbiased=False) + norm.eps)
    torch.testing.assert_close(norm(data), expected)

  def test_forward_preserves_input_gradient_without_updating_statistics(self):
    norm = RunningNorm(2)
    norm.update(torch.tensor([[1.0, 2.0], [3.0, 6.0]]))
    before = {key: value.clone() for key, value in norm.state_dict().items()}
    values = torch.tensor([[2.0, 3.0]], requires_grad=True)
    norm(values).sum().backward()
    torch.testing.assert_close(values.grad, (1 / (norm.std + norm.eps))[None])
    for key, value in norm.state_dict().items():
      torch.testing.assert_close(value, before[key], rtol=0, atol=0)

  def test_denormalization_inverts_normalization_and_preserves_gradients(self):
    norm = RunningNorm(2, eps=1e-4)
    norm.update(torch.tensor([[1.0, 2.0], [3.0, 6.0]]))
    value = torch.tensor([[0.3, -0.2]], requires_grad=True)
    restored = norm.denormalize(value)
    torch.testing.assert_close(norm(restored), value)
    restored.sum().backward()
    torch.testing.assert_close(value.grad, (norm.std + norm.eps)[None])

  def test_eval_mode_and_sample_limit_freeze_updates(self):
    norm = RunningNorm(2, max_count=2)
    data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    norm.eval()
    norm.update(data)
    self.assertEqual(int(norm.count), 0)
    norm.train()
    norm.update(data)
    before = norm.mean.clone()
    norm.update(data + 100)
    self.assertEqual(int(norm.count), 2)
    torch.testing.assert_close(norm.mean, before)

  def test_nonfinite_samples_do_not_contaminate_statistics(self):
    norm = RunningNorm(2)
    norm.update(torch.ones(2, 2))
    before = {key: value.clone() for key, value in norm.state_dict().items()}
    for invalid in (float("nan"), float("inf")):
      with self.subTest(invalid=invalid), self.assertRaises(FloatingPointError):
        norm.update(torch.tensor([[1.0, invalid]]))
    for key, value in norm.state_dict().items():
      torch.testing.assert_close(value, before[key], rtol=0, atol=0)

  def test_large_finite_samples_do_not_overflow_variance(self):
    norm = RunningNorm(2)
    data = torch.tensor([[1e20, -1e20], [-1e20, 1e20]])
    norm.update(data)
    self.assertTrue(torch.isfinite(norm.var).all())
    self.assertTrue(torch.isfinite(norm.std).all())
    torch.testing.assert_close(norm(data), torch.tensor([[1.0, -1.0], [-1.0, 1.0]]))


class NormalizedPolicyTests(unittest.TestCase):
  def tearDown(self):
    # CPU/CUDA, both reference widths and normalization flags specialize the
    # shared kernel code differently. Keep test cases from exhausting Dynamo's
    # per-code-object cache; one training run uses a single configuration.
    torch._dynamo.reset()

  @classmethod
  def setUpClass(cls):
    torch.set_num_threads(4)
    cls.metadata = robot_metadata()
    folder = tempfile.TemporaryDirectory()
    cls.addClassCleanup(folder.cleanup)
    cls.manifest, _ = make_motions(Path(folder.name), cls.metadata)
    cls.library = MotionLibrary(
      cls.manifest,
      tuple(cls.metadata["joint_names"]),
      TRACKED_BODIES,
      clips=("fixture_walk",),
    )

  def models(self, reference_dim):
    actor = MotionActor(
      self.metadata["default_q"],
      (16,),
      reference_dim=reference_dim,
      input_normalization=True,
    )
    critic = MotionCritic(
      self.metadata["default_q"],
      (16,),
      reference_dim=reference_dim,
      input_normalization=True,
    )
    world = WorldModel(self.metadata, (16,), "analytic")
    torch.nn.init.normal_(world.dynamics[-1].weight, std=1e-4)
    return actor, critic, world

  def test_normalized_export_preserves_actions_and_freezes_statistics(self):
    for width in (730, 2680):
      with self.subTest(width=width), tempfile.TemporaryDirectory() as folder:
        actor, _, _ = self.models(width)
        starts = make_starts(self.library, actor.default_q)
        ref = self.library.get_future(starts["motion_id"], starts["frame_id"])
        goal = actor.reference_features(ref, actor.default_q)
        actor.update_normalization(starts["history"][:, -1], goal)
        saved = {
          "kind": "supermimic_online",
          "schema_version": 1,
          "reference_schema": actor.reference_schema,
          "config": {"actor_dims": (16,), "actor_input_normalization": True},
          "actor": actor.state_dict(),
          "robot_metadata": self.metadata,
          "simulation": {},
          "clips": self.library.names,
          "manifest_sha256": file_sha256(self.manifest),
        }
        source, output = Path(folder) / "full.pt", Path(folder) / "actor.pt"
        torch.save(saved, source)
        export_policy(source, output)
        loaded, _, _ = load_actor(output, self.manifest, "cpu")
        with torch.no_grad():
          torch.testing.assert_close(
            actor(starts["history"], ref),
            loaded(starts["history"], ref),
            rtol=0,
            atol=0,
          )
        loaded.update_normalization(starts["history"][:, -1] + 5, goal + 5)
        self.assertEqual(int(loaded.obs_norm.count), len(goal))
        self.assertEqual(int(loaded.goal_norm.count), len(goal))

  def test_compiled_normalized_bptt_preserves_gradients_and_statistics(self):
    for width in (730, 2680):
      with self.subTest(width=width):
        actor, critic, world = self.models(width)
        starts = make_starts(self.library, actor.default_q)
        ref = self.library.get_future(starts["motion_id"], starts["frame_id"])
        goal = actor.reference_features(ref, actor.default_q)
        actor.update_normalization(starts["history"][:, -1], goal)
        critic.update_normalization(
          critic.features(starts["state"], ref, starts["context"], goal)
        )
        before = [
          {key: value.clone() for key, value in module.state_dict().items()}
          for module in (actor.obs_norm, actor.goal_norm, critic.input_norm)
        ]
        kernels = TrainingKernels(
          actor, world, critic, enabled=True, backend="aot_eager"
        )
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
            kernels=kernels,
          )
          actual["objective"].backward()
        torch.testing.assert_close(
          actual["objective"], expected["objective"], rtol=2e-5, atol=2e-6
        )
        for parameter, gradient in zip(actor.parameters(), gradients):
          torch.testing.assert_close(parameter.grad, gradient, rtol=2e-4, atol=2e-6)
        for module, saved in zip(
          (actor.obs_norm, actor.goal_norm, critic.input_norm), before
        ):
          for key, value in module.state_dict().items():
            torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
        # Compiled calls must read statistics updated between iterations.
        actor.update_normalization(starts["history"][:, -1] + 0.1, goal + 0.1)
        with frozen_parameters(world, critic):
          eager = imagine(
            actor, world, critic, self.library, starts, horizon=2, sensor_noise=False
          )
          compiled = imagine(
            actor,
            world,
            critic,
            self.library,
            starts,
            horizon=2,
            sensor_noise=False,
            kernels=kernels,
          )
        torch.testing.assert_close(
          eager["objective"], compiled["objective"], rtol=2e-5, atol=2e-6
        )

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
  def test_cuda_bf16_compiled_full_body_normalization(self):
    if not torch.cuda.is_bf16_supported():
      self.skipTest("GPU does not support BF16")
    actor, critic, world = [model.cuda() for model in self.models(2680)]
    for model in (actor, critic, world):
      model.amp = "bf16"
    library = copy.copy(self.library)
    library.device = "cuda"
    library.lengths = library.lengths.cuda()
    library.offsets = library.offsets.cuda()
    library.data = {key: value.cuda() for key, value in library.data.items()}
    starts = make_starts(library, actor.default_q)
    ref = library.get_future(starts["motion_id"], starts["frame_id"])
    goal = actor.reference_features(ref, actor.default_q)
    actor.update_normalization(starts["history"][:, -1], goal)
    critic.update_normalization(
      critic.features(starts["state"], ref, starts["context"], goal)
    )
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
        compiled["objective"], eager["objective"], rtol=2e-2, atol=2e-3
      )
      compiled["objective"].backward()
    self.assertTrue(
      all(
        torch.isfinite(p.grad).all() for p in actor.parameters() if p.grad is not None
      )
    )
    for norm in (actor.obs_norm, actor.goal_norm, critic.input_norm):
      self.assertEqual(int(norm.count), len(goal))


if __name__ == "__main__":
  unittest.main()
