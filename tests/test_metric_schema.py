"""Reward/error separation, units and collection-window semantics."""

import io
import unittest

from src.tasks.supermimic.rl.metric_schema import classified_metrics, metric_groups
from src.tasks.supermimic.rl.training_logger import MetricWindow
from src.tasks.supermimic.rl.velocity_logger import (
  VelocityConsoleReporter,
  VelocityMetricWindow,
)
from tests.test_training_logger import record as motion_record


def velocity_record(
  iteration=1, *, collected=True, count=10, tracking=0.02, error=0.193
):
  return {
    "iteration": iteration,
    "stage": "joint",
    "warmup_completed": 500,
    "learning_rate": 2e-4,
    "seconds": 1.0,
    "collected_this_iteration": collected,
    "collected_transitions": count,
    "velocity/linear_tracking": tracking,
    "velocity/yaw_tracking": 0.03,
    "velocity/action_smoothness": -0.01,
    "velocity/joint_limit": -0.001,
    "velocity/torso_sideways": -0.002,
    "velocity/torso_angular_velocity": -0.003,
    "velocity/stand_still": 0.0,
    "velocity/linear_velocity_error": error,
    "velocity/yaw_rate_error": 0.313,
    "velocity/physical_failure": 0.0,
    "motion/failure_rate": 0.0,
    "velocity/objective": 1.5,
    "velocity/alignment_loss": 0.1,
    "velocity/critic_loss": 0.2,
    "wm_loss": 0.005,
  }


class MetricSchemaTests(unittest.TestCase):
  def test_reward_terms_are_weighted_contributions_and_errors_keep_units(self):
    metrics = classified_metrics(velocity_record())
    self.assertEqual(metrics["reward/velocity/linear_tracking"], 0.02)
    self.assertEqual(metrics["reward/velocity/action_smoothness"], -0.01)
    self.assertAlmostEqual(metrics["reward/velocity/total_per_step"], 0.034)
    self.assertEqual(metrics["error/velocity/planar_velocity_m_per_s"], 0.193)
    self.assertEqual(metrics["error/velocity/yaw_rate_rad_per_s"], 0.313)
    self.assertFalse(
      any("error" in key for key in metrics if key.startswith("reward/"))
    )
    self.assertEqual(metrics["objective/velocity/return"], 1.5)
    self.assertEqual(metrics["loss/velocity/latent_alignment_mse"], 0.1)
    self.assertEqual(metrics["loss/critic/velocity"], 0.2)

  def test_legacy_motion_log_and_reward_prediction_error_are_classified(self):
    groups = metric_groups(
      {
        "real_reward_per_step": 0.08,
        "real_body_position": 0.02,
        "real_body_position_error": 0.1,
        "real_root_orientation_error": 0.2,
        "wm_reward": 0.006,
        "actor_objective": 2.0,
      }
    )
    self.assertEqual(groups["reward"]["motion/total_per_step"], 0.08)
    self.assertEqual(groups["reward"]["motion/body_position"], 0.02)
    self.assertEqual(groups["error"]["motion/body_position_m"], 0.1)
    self.assertEqual(groups["error"]["motion/root_orientation_rad"], 0.2)
    self.assertEqual(groups["error"]["world_model/reward_prediction"], 0.006)
    self.assertNotIn("world_model/reward_prediction", groups["reward"])
    self.assertEqual(groups["objective"]["motion/return"], 2.0)

  def test_update_only_record_does_not_duplicate_collection_metrics(self):
    groups = metric_groups(velocity_record(collected=False))
    self.assertNotIn("reward", groups)
    self.assertNotIn("failure", groups)
    self.assertNotIn("error", groups)
    self.assertIn("loss", groups)
    self.assertIn("objective", groups)

  def test_window_weights_real_rewards_and_errors_by_new_transitions(self):
    window = VelocityMetricWindow()
    window.add(velocity_record(count=10, tracking=0.02, error=0.2))
    window.add(velocity_record(2, collected=False, tracking=999, error=999))
    window.add(velocity_record(3, count=30, tracking=0.04, error=0.4))
    summary = window.summary()
    metrics = classified_metrics(summary)
    self.assertAlmostEqual(metrics["reward/velocity/linear_tracking"], 0.035)
    self.assertAlmostEqual(metrics["error/velocity/planar_velocity_m_per_s"], 0.35)
    self.assertAlmostEqual(summary["steps_per_second"], 40 / 3)

  def test_update_only_window_has_no_stale_reward_or_error(self):
    window = VelocityMetricWindow()
    window.add(velocity_record(collected=False))
    metrics = classified_metrics(window.summary())
    self.assertFalse(
      any(key.startswith(("reward/", "error/", "failure/")) for key in metrics)
    )
    self.assertIn("loss/critic/velocity", metrics)

  def test_motion_update_only_window_does_not_invent_failure_or_reward(self):
    window = MetricWindow()
    window.add(motion_record(collected=False))
    metrics = classified_metrics(window.summary())
    self.assertFalse(any(key.startswith(("reward/", "failure/")) for key in metrics))
    self.assertIn("loss/critic/motion", metrics)

  def test_console_labels_units_and_optimization_direction_with_plain_fallback(self):
    window = VelocityMetricWindow()
    window.add(velocity_record())
    for plain in (False, True):
      with self.subTest(plain=plain):
        stream = io.StringIO()
        reporter = VelocityConsoleReporter(100000, stream=stream)
        if plain:
          reporter.console = None
        reporter.training(window.summary())
        text = stream.getvalue()
        for label in (
          "Reward ↑",
          "Error ↓",
          "Loss ↓",
          "Objective ↑",
          "weighted",
          "m/s",
          "rad/s",
        ):
          self.assertIn(label, text)
        self.assertNotIn("Vel J", text)
