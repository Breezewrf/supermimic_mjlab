"""Metric semantics, terminal rendering, and W&B continuation behavior."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from src.tasks.supermimic.rl.training_logger import (
  ConsoleReporter,
  MetricWindow,
  WandbReporter,
  simulator_output,
)


def record(iteration=1, collected=True):
  return {
    "iteration": iteration,
    "environment_steps": 20,
    "replay_transitions": 20,
    "learning_rate": 2e-4,
    "seconds": 2.0,
    "world_model_seconds": 0.8,
    "policy_and_critic_seconds": 1.0,
    "collected_this_iteration": collected,
    "collected_transitions": 10,
    "physical_failures": 1,
    "episodes_ended": 2,
    "collection_seconds": 0.2,
    "real_reward_per_step": 0.03,
    "wm_loss": 0.8,
    "wm_state": 0.2,
    "wm_termination": 0.6,
    "wm_observation": 0.1,
    "wm_reward": 0.02,
    "actor_objective": 0.9,
    "imagined_reward": 0.04,
    "survival_at_horizon": 0.5,
    "critic_loss": 1.2,
    "wm_gradient_norm": 3.0,
    "actor_gradient_norm": 2.0,
    "critic_gradient_norm": 80.0,
    "peak_gpu_memory_gib": 26.9,
  }


def wm_only_record(iteration=1):
  row = record(iteration)
  row["training_phase"] = "wm_only"
  row["policy_and_critic_seconds"] = 0.0
  for key in (
    "actor_objective",
    "imagined_reward",
    "survival_at_horizon",
    "critic_loss",
    "actor_gradient_norm",
    "critic_gradient_norm",
  ):
    del row[key]
  return row


class MetricTests(unittest.TestCase):
  def test_wm_only_and_mixed_windows_do_not_invent_or_dilute_policy_metrics(self):
    window = MetricWindow()
    window.add(wm_only_record())
    warmup = window.summary()
    self.assertNotIn("actor_objective", warmup)
    self.assertEqual(warmup["training_phase"], "wm_only")
    stream = io.StringIO()
    console = ConsoleReporter(1000, stream=stream)
    console.training(warmup)
    self.assertIn("--", stream.getvalue())
    self.assertIn("WM-only warmup", stream.getvalue())
    joint = record(2, False) | {"training_phase": "joint"}
    window.add(joint)
    mixed = window.summary()
    self.assertEqual(mixed["actor_objective"], joint["actor_objective"])
    self.assertEqual(mixed["critic_loss"], joint["critic_loss"])
    self.assertEqual(mixed["policy_and_critic_seconds"], 0.5)
    self.assertEqual(mixed["training_phase"], "joint")
    console.training(mixed)
    self.assertIn("Joint training", stream.getvalue())

  def test_window_does_not_double_count_carried_collection_statistics(self):
    window = MetricWindow()
    window.add(record())
    window.add(record(2, False))
    third = record(3)
    third.update(
      collected_transitions=30,
      physical_failures=2,
      real_reward_per_step=0.07,
      environment_steps=50,
    )
    window.add(third)
    actual = window.summary()
    self.assertAlmostEqual(actual["real_reward_per_step"], 0.06)
    self.assertAlmostEqual(actual["failure_events_per_transition"], 3 / 40)
    self.assertAlmostEqual(actual["window_transitions_per_second"], 40 / 6)
    self.assertEqual(actual["environment_steps"], 50)
    self.assertEqual(actual["window_episodes_ended"], 4)

  def test_update_only_window_has_no_fake_collection_metrics(self):
    window = MetricWindow()
    window.add(record(collected=False))
    summary = window.summary()
    self.assertIsNone(summary["real_reward_per_step"])
    self.assertIsNone(summary["failure_events_per_transition"])
    self.assertEqual(summary["window_transitions_per_second"], 0)

  def test_table_output_and_json_compatibility(self):
    stream = io.StringIO()
    console = ConsoleReporter(1000, stream=stream)
    window = MetricWindow()
    window.add(record())
    console.training(window.summary())
    self.assertIn("WM loss", stream.getvalue())
    self.assertNotIn('"actor_objective"', stream.getvalue())
    raw = io.StringIO()
    ConsoleReporter(1000, "json", raw).training(record())
    self.assertEqual(json.loads(raw.getvalue()), record())

  def test_simulator_output_is_preserved_without_hiding_exceptions(self):
    simulator, terminal = io.StringIO(), io.StringIO()
    with patch("sys.stdout", terminal):
      with (
        self.assertRaisesRegex(RuntimeError, "visible failure"),
        simulator_output(simulator, False),
      ):
        print("Base Environment")
        raise RuntimeError("visible failure")
      print("training progress")
    self.assertIn("Base Environment", simulator.getvalue())
    self.assertEqual(terminal.getvalue(), "training progress\n")


class WandbTests(unittest.TestCase):
  def setUp(self):
    folder = tempfile.TemporaryDirectory()
    self.addCleanup(folder.cleanup)
    self.output = Path(folder.name)
    self.runs = []

    def init(**kwargs):
      run = MagicMock(
        spec=[
          "id",
          "project",
          "entity",
          "name",
          "url",
          "dir",
          "define_metric",
          "log",
          "finish",
        ]
      )
      run.id, run.project = kwargs["id"], kwargs["project"]
      run.entity, run.name = kwargs.get("entity") or "test-team", kwargs["name"]
      run.url = f"https://wandb.example/{run.entity}/{run.project}/runs/{run.id}"
      run.dir = str(self.output / "wandb" / run.id)
      self.runs.append((kwargs, run))
      return run

    sdk = SimpleNamespace(
      init=init,
      Settings=lambda **kwargs: kwargs,
      Table=lambda **kwargs: kwargs,
      plot=SimpleNamespace(bar=lambda *args, **kwargs: kwargs),
    )
    mock = patch.dict("sys.modules", {"wandb": sdk})
    mock.start()
    self.addCleanup(mock.stop)

  def test_training_only_logs_collection_when_it_happens(self):
    tracker = WandbReporter(self.output, {}, 0)
    tracker.training(record())
    first = tracker.run.log.call_args.args[0]
    self.assertEqual(first["collection/failure_events_per_transition"], 0.1)
    self.assertEqual(first["gradients/critic_preclip_max_norm"], 80)
    tracker.training(record(2, False))
    second = tracker.run.log.call_args.args[0]
    self.assertFalse(any(key.startswith("collection/") for key in second))
    self.assertEqual(second["train/iteration"], 2)
    tracker.close(exit_code=1)
    tracker.run.finish.assert_called_once_with(exit_code=1)

  def test_online_console_uses_url_property_without_get_url_method(self):
    console = MagicMock()
    tracker = WandbReporter(self.output, {}, 0, console=console)
    self.assertFalse(hasattr(tracker.run, "get_url"))
    console.message.assert_called_once_with(f"W&B online: {tracker.run.url}")

  def test_offline_console_uses_local_run_directory(self):
    console = MagicMock()
    tracker = WandbReporter(self.output, {}, 0, console=console, mode="offline")
    console.message.assert_called_once_with(f"W&B offline: {tracker.run.dir}")

  def test_online_console_falls_back_to_directory_when_url_is_unavailable(self):
    first = WandbReporter(self.output, {}, 0)
    first.run.url = None
    console = MagicMock()
    with patch.object(first.sdk, "init", return_value=first.run):
      tracker = WandbReporter(self.output, {}, 0, console=console)
    console.message.assert_called_once_with(f"W&B online: {tracker.run.dir}")

  def test_wm_only_round_logs_world_metrics_without_policy_metrics(self):
    tracker = WandbReporter(self.output, {}, 0)
    tracker.training(wm_only_record())
    metrics = tracker.run.log.call_args.args[0]
    self.assertEqual(metrics["world_model/loss"], 0.8)
    self.assertEqual(metrics["train/training_phase"], "wm_only")
    self.assertFalse(any(key.startswith(("policy/", "critic/")) for key in metrics))
    self.assertNotIn("gradients/actor_preclip_norm", metrics)
    self.assertNotIn("gradients/critic_preclip_max_norm", metrics)
    tracker.training(record(2))
    self.assertEqual(tracker.run.log.call_args.args[0]["policy/objective"], 0.9)

  def test_online_resume_and_rollback_branch(self):
    first = WandbReporter(self.output, {}, 0)
    first.training(record(10))
    resumed = WandbReporter(self.output, {}, 10)
    self.assertEqual(resumed.state["id"], first.state["id"])
    self.assertEqual(self.runs[-1][0]["resume"], "allow")
    rolled_back = WandbReporter(self.output, {}, 5)
    self.assertNotEqual(rolled_back.state["id"], first.state["id"])
    self.assertEqual(self.runs[-1][0]["config"]["parent_run_id"], first.state["id"])

  def test_offline_resume_starts_linked_segment(self):
    first = WandbReporter(self.output, {}, 0, mode="offline")
    first.training(record(10))
    resumed = WandbReporter(self.output, {}, 10, mode="offline")
    self.assertNotEqual(resumed.state["id"], first.state["id"])
    self.assertIsNone(self.runs[-1][0]["resume"])
    self.assertEqual(self.runs[-1][0]["config"]["parent_run_id"], first.state["id"])

  def test_evaluation_logs_scalars_and_motion_charts(self):
    tracker = WandbReporter(self.output, {}, 0)
    tracker.evaluation(
      {
        "iteration": 10,
        "training_environment_steps": 1000,
        "normalized_return": 0.5,
        "successes": 1,
        "motions": [
          {
            "motion": "a",
            "normalized_return": 0.8,
            "completion": 1.0,
            "success": True,
            "steps": 100,
            "failure_reason": None,
            "evaluation_limited": False,
            "mean_errors": {"root_position_error": 0.1},
          },
          {
            "motion": "b",
            "normalized_return": 0.2,
            "completion": 0.4,
            "success": False,
            "steps": 40,
            "failure_reason": "height_failure",
            "evaluation_limited": False,
            "mean_errors": {"root_position_error": 0.3},
          },
        ],
      }
    )
    payload = tracker.run.log.call_args.args[0]
    self.assertEqual(payload["evaluation/success_rate"], 0.5)
    self.assertAlmostEqual(payload["evaluation/root_position_error"], 0.2)
    self.assertEqual(len(payload["evaluation/motions"]["data"]), 2)
    self.assertIn("evaluation/per_motion_return", payload)


if __name__ == "__main__":
  unittest.main()
