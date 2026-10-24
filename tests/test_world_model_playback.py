"""World-model synchronization, terminal frames, render isolation and checkpoint loading."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from queue import SimpleQueue
from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from src.tasks.supermimic.config.rl_cfg import OnlineConfig
from src.tasks.supermimic.config.task_cfg import FoundationConfig
from src.tasks.supermimic.data.world_model_data import file_sha256
from src.tasks.supermimic.mdp.state import observe
from src.tasks.supermimic.models.policy import MotionActor
from src.tasks.supermimic.models.world_model import WorldModel
from src.tasks.supermimic.rl.checkpoints import save_checkpoint, world_model_checkpoint
from src.tasks.supermimic.rl.evaluation import evaluate
from src.tasks.supermimic.rl.inference import load_world_model, validate_environment
from src.tasks.supermimic.rl.visualization import EvaluationScene, LiveViewer
from src.tasks.supermimic.rl.world_model_playback import (
  WorldModelFuturePreview,
  WorldModelPlayback,
)


def physical_state(x=0.0):
  state = torch.zeros(1, 71)
  state[:, 0], state[:, 2], state[:, 3] = x, 0.8, 1.0
  return state


class ToyDynamics:
  def __init__(self):
    self.actions, self.contexts = [], []

  def __call__(self, state, action, context):
    self.actions.append(action.clone())
    self.contexts.append(context.clone())
    predicted = state.clone()
    predicted[:, 0] += 2 * action[:, 0]
    return predicted

  def kinematics(self, state):
    return {"body_pos_w": state[:, None, :3]}


class WorldModelPlaybackTests(unittest.TestCase):
  def setUp(self):
    self.env = SimpleNamespace(
      num_envs=1,
      context=torch.ones(1, 33),
      foundation_config=FoundationConfig(num_envs=1),
      step_dt=0.02,
      library=SimpleNamespace(names=["one", "two"]),
    )
    self.world = ToyDynamics()

  def test_rollout_accumulates_error_and_resyncs_only_at_horizon_and_clip_start(self):
    observer = WorldModelPlayback(self.env, self.world, horizon=3)
    observer.on_frame(self.env, 0, 0, physical_state(), 0)
    errors = []
    for step in range(1, 5):
      observer.predict(torch.ones(1, 29))
      observer.on_frame(self.env, 0, step, physical_state(step), 0)
      errors.append(observer.metrics["root_position_m"])
      self.env.context.fill_(5)
    self.assertEqual(errors, [1, 2, 3, 1])
    self.assertEqual(observer.age, 1)
    self.assertEqual([float(c[0, 0]) for c in self.world.contexts], [1, 1, 1, 5])
    report = observer.report()
    self.assertEqual(report["motions"][0]["samples"], 4)
    self.assertEqual(report["motions"][0]["mean_errors"]["root_position_m"], 1.75)
    observer.on_frame(self.env, 1, 0, physical_state(10), 0)
    self.assertEqual(observer.age, 0)
    self.assertEqual(float(observer.state[0, 0]), 10)
    self.assertIsNone(observer.metrics)
    observer.begin_evaluation()
    self.assertEqual(observer.report()["motions"], [])

  def test_single_step_prediction_uses_pre_reset_successor_and_clipped_actions(self):
    self.env.foundation_config = FoundationConfig(action_clip=0.5)
    observer = WorldModelPlayback(self.env, self.world, horizon=1)
    observer.on_frame(self.env, 0, 0, physical_state(), 0)
    action = torch.ones(1, 29)
    observed = observer.policy(lambda history, reference: action)
    for step in (1, 2):
      self.assertIs(observed(None, None), action)
      observer.on_frame(self.env, 0, step, physical_state(step), 0)
      self.assertEqual(observer.metrics["root_position_m"], 0)
    torch.testing.assert_close(action, torch.ones_like(action))
    torch.testing.assert_close(self.world.actions[0], action * 0.5)

  def test_nonfinite_predictions_are_hidden_until_next_synchronization(self):
    calls = []
    original = self.world

    class Divergent(ToyDynamics):
      def __call__(self, state, action, context):
        calls.append(1)
        return (
          torch.full_like(state, float("nan"))
          if len(calls) == 1
          else original(state, action, context)
        )

    observer = WorldModelPlayback(self.env, Divergent(), horizon=2)
    observer.on_frame(self.env, 0, 0, physical_state(), 0)
    for step in (1, 2):
      observer.predict(torch.zeros(1, 29))
      observer.on_frame(self.env, 0, step, physical_state(), 0)
      self.assertIsNone(observer.state)
      self.assertIn("DIVERGED", observer.status())
    observer.predict(torch.zeros(1, 29))
    observer.on_frame(self.env, 0, 3, physical_state(), 0)
    self.assertEqual(observer.metrics["root_position_m"], 0)
    self.assertEqual(observer.report()["motions"][0]["nonfinite_steps"], 2)

  def test_prediction_visibility_toggle_does_not_advance_the_model(self):
    observer = WorldModelPlayback(self.env, self.world)
    viewer = object.__new__(LiveViewer)
    viewer.world_model = observer
    viewer.scene = SimpleNamespace(show_world_model=True)
    viewer.commands = SimpleQueue()
    viewer.commands.put(ord("W"))
    viewer._commands()
    self.assertFalse(viewer.scene.show_world_model)
    self.assertEqual(observer.age, 0)

  def test_orange_prediction_uses_absolute_pose_and_independent_render_buffers(self):
    joints = "".join(
      f'<body pos="0 0 {i * 0.02}"><joint name="j{i}"/><geom size=".01"/></body>'
      for i in range(29)
    )
    model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><body><freejoint/><geom size=".1"/>'
      + joints
      + "</body></worldbody></mujoco>"
    )
    data = mujoco.MjData(model)
    original = data.qpos.copy()
    reference = {
      "body_pos_w": torch.tensor([[[0.0, 0.0, 0.8]]]),
      "body_quat_w": torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
      "joint_pos": torch.zeros(1, 29),
    }
    self.env.sim = SimpleNamespace(mj_model=model, data=data)
    self.env.device = "cpu"
    self.env.robot = SimpleNamespace(
      indexing=SimpleNamespace(
        free_joint_q_adr=torch.arange(7),
        joint_q_adr=torch.arange(35, 6, -1),
        free_joint_v_adr=torch.arange(6),
        joint_v_adr=torch.arange(34, 5, -1),
      )
    )
    self.env.scene = SimpleNamespace(env_origins=torch.tensor([[4.0, 5.0, 0.0]]))
    self.env.library.get_frame = lambda *args: reference
    observer = WorldModelPlayback(self.env, self.world)
    observer.on_frame(self.env, 0, 0, physical_state(2), 0)
    observer.state[:, 7:36] = torch.linspace(-0.2, 0.2, 29)
    scene = EvaluationScene(self.env, ghost=False, world_model=observer)
    scene.update(self.env, 0, 0, physical_state(1))
    np.testing.assert_allclose(scene.data.qpos[:3], [5, 5, 0.8])
    np.testing.assert_allclose(scene.prediction_data.qpos[:3], [6, 5, 0.8])
    np.testing.assert_allclose(
      scene.prediction_data.qpos[scene.joint_adr], observer.state[0, 7:36]
    )
    rendered = mujoco.MjvScene(scene.model, maxgeom=1000)
    scene.add_ghost(rendered)
    self.assertGreater(rendered.ngeom, 0)
    for geom in rendered.geoms[: rendered.ngeom]:
      np.testing.assert_allclose(geom.rgba, [1, 0.55, 0.1, 0.5], atol=1e-6)
    np.testing.assert_array_equal(data.qpos, original)
    scene.show_world_model = False
    rendered.ngeom = 0
    scene.add_ghost(rendered)
    self.assertEqual(rendered.ngeom, 0)
    observer.state = None
    scene.update(self.env, 0, 0, physical_state(1))
    self.assertFalse(scene.prediction_ready)


class WorldModelPlaybackIntegrationTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    from src.tasks.supermimic.models.kinematics import robot_metadata

    torch.set_num_threads(2)
    cls.metadata = robot_metadata()

  def test_future_preview_rollout_time_alignment_and_physics_isolation(self):
    from src.tasks.supermimic.env import SuperMimicEnv
    from tests.supermimic_fixtures import make_motions

    with tempfile.TemporaryDirectory() as directory:
      manifest, _ = make_motions(Path(directory), self.metadata)
      env = SuperMimicEnv(
        FoundationConfig(
          manifest=manifest,
          num_envs=1,
          episode_length_s=1e9,
          relative_orientation=True,
          terminate_on_failure=False,
        ),
        device="cpu",
      )
      try:
        actor = MotionActor(
          self.metadata["default_q"], (8,), relative_orientation=True
        ).eval()
        world = WorldModel(self.metadata, (8,), relative_orientation=True).eval()
        with torch.no_grad():
          world.dynamics[-1].weight.normal_(std=0.003)
        preview = WorldModelFuturePreview(env, world, actor, horizon=3, refresh=2)
        env.reset(seed=0, options={"motion_id": 0, "frame_id": 0})
        initial = env.read_state().clone()
        history = env.policy_input()["history"].clone()
        expected = [initial]
        # Independent unbatched policy loop verifies future references and feedback.
        with torch.no_grad():
          state = initial
          for frame in range(4):
            reference = env.library.get_future(torch.tensor([0]), torch.tensor([frame]))
            action = actor(history, reference)
            state = world(state, action, env.context)
            following = env.library.get_frame(
              torch.tensor([0]), torch.tensor([frame + 1])
            )
            observation = observe(
              state,
              action,
              actor.default_q,
              env.context,
              following["body_quat_w"][:, 0],
            )
            history = torch.cat((history[:, 1:], observation[:, None]), 1)
            expected.append(state)
        preview.on_frame(env, 0, 0, initial, 0)
        torch.testing.assert_close(
          preview.cache, torch.stack(expected), atol=1e-6, rtol=1e-5
        )
        self.assertEqual(preview.target_frame, 3)
        torch.testing.assert_close(preview.state, expected[3], atol=1e-6, rtol=1e-5)
        self.assertIsNone(preview.metrics)  # Future poses are never scored against now.
        wrapped = preview.policy(actor)
        for frame in (1, 2):
          inputs = env.policy_input()
          with torch.no_grad():
            action = wrapped(inputs["history"], inputs["future_reference"])
          _, _, _, _, extras = env.step(action)
          preview.on_frame(env, 0, frame, extras["transition"]["next_state"], 0)
          torch.testing.assert_close(preview.history, env.policy_input()["history"])
          self.assertEqual(preview.target_frame, frame + 3)
          self.assertEqual(preview.anchor, 0 if frame == 1 else 2)
        baseline = evaluate(env, actor)
        preview.begin_evaluation()
        seen = []

        def on_frame(*args):
          preview.on_frame(*args)
          seen.append(
            (args[1], args[2], preview.target_frame, preview.metrics, preview.state)
          )

        compared = evaluate(env, wrapped, on_frame=on_frame)
        self.assertEqual(baseline, compared)
        for clip, frame, target, metrics, state in seen:
          self.assertEqual(target, min(frame + 3, int(env.library.lengths[clip]) - 1))
          self.assertEqual(metrics is not None, frame >= 3)
          if target == frame:
            self.assertIsNone(state)
        report = preview.report()
        self.assertEqual(report["mode"], "future")
        self.assertEqual([r["samples"] for r in report["motions"]], [62, 70])
        json.dumps(report, allow_nan=False)
      finally:
        env.close()

  def test_loader_reconstructs_world_without_actor_schema_and_checks_data(self):
    with tempfile.TemporaryDirectory() as directory:
      directory = Path(directory)
      manifest = directory / "manifest.json"
      manifest.write_text("{}")
      world = WorldModel(self.metadata, (8,)).eval()
      saved = {
        "kind": "supermimic_online",
        "schema_version": 1,
        "world": world.state_dict(),
        "robot_metadata": self.metadata,
        "config": asdict(OnlineConfig(world_dims=(8,))),
        "manifest_sha256": file_sha256(manifest),
        "clips": ["one"],
        "simulation": {},
      }
      path = directory / "world.pt"
      save_checkpoint(path, world_model_checkpoint(saved))
      loaded, _ = load_world_model(path, manifest, "cpu", ("one",))
      self.assertFalse(loaded.training)
      self.assertEqual(loaded.world_prediction, "state_delta")
      state = physical_state()
      state[:, 36] = 1.5
      action, context = torch.zeros(1, 29), torch.zeros(1, 33)
      torch.testing.assert_close(
        loaded(state, action, context), world(state, action, context)
      )
      # Historical files have identical tensor layouts but different prediction semantics.
      del saved["config"]["world_prediction"]
      save_checkpoint(path, world_model_checkpoint(saved))
      legacy, _ = load_world_model(path, manifest, "cpu")
      self.assertEqual(legacy.world_prediction, "velocity_residual")
      expected = state.clone()
      expected[:, 0] += 1.5 * 0.02
      torch.testing.assert_close(legacy(state, action, context), expected)
      for key, value in world.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value, atol=0, rtol=0)
      with self.assertRaisesRegex(ValueError, "clips"):
        load_world_model(path, manifest, "cpu", ("unknown",))
      manifest.write_text('{"changed": true}')
      with self.assertRaisesRegex(ValueError, "manifest"):
        load_world_model(path, manifest, "cpu")
      saved["kind"] = "supermimic_actor"
      save_checkpoint(path, saved)
      with self.assertRaisesRegex(ValueError, "policy-only"):
        load_world_model(path, manifest, "cpu")

  def test_world_comparison_leaves_real_physics_and_returns_unchanged(self):
    from src.tasks.supermimic.env import SuperMimicEnv
    from src.tasks.supermimic.rl.online_training import simulation_parameters
    from tests.supermimic_fixtures import make_motions

    with tempfile.TemporaryDirectory() as directory:
      manifest, _ = make_motions(Path(directory), self.metadata)
      env = SuperMimicEnv(
        FoundationConfig(manifest=manifest, num_envs=1, episode_length_s=1e9),
        device="cpu",
      )
      try:
        world = WorldModel(self.metadata, (8,)).eval()
        observer = WorldModelPlayback(env, world, horizon=3)
        baseline = evaluate(env, max_steps_per_clip=7)
        from src.tasks.supermimic.rl.evaluation import zero_policy

        compared = evaluate(
          env,
          observer.policy(zero_policy),
          max_steps_per_clip=7,
          on_frame=observer.on_frame,
        )
        self.assertEqual(baseline, compared)
        self.assertEqual(
          [row["samples"] for row in observer.report()["motions"]], [7, 7]
        )
        json.dumps(observer.report(), allow_nan=False)
        valid = {
          "robot_metadata": self.metadata,
          "simulation": simulation_parameters(env),
        }
        validate_environment(env, valid)
        with self.assertRaisesRegex(ValueError, "robot model"):
          validate_environment(env, {**valid, "robot_metadata": {"signature": "wrong"}})
        with self.assertRaisesRegex(ValueError, "simulation"):
          validate_environment(env, {**valid, "simulation": {}})
      finally:
        env.close()


if __name__ == "__main__":
  unittest.main()
