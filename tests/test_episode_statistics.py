"""Finished-episode aggregation and explicit reward normalization units."""

import unittest

import torch

from src.tasks.supermimic.data.episode_statistics import (
  EpisodeStatistics,
  EpisodeSummary,
)
from src.tasks.supermimic.rl.metric_schema import classified_metrics
from src.tasks.supermimic.rl.velocity_logger import VelocityMetricWindow


def report(count=1, error=0.2, total=0.08, steps=2):
  return {
    "velocity": {
      "count": count,
      "normalization_s": 10.0,
      "steps_sum": count * steps,
      "duration_s_sum": count * steps * 0.02,
      "terminated": 0,
      "time_out": count,
      "motion_end": 0,
      "metric_sums": {
        "reward_per_step": count * total,
        "linear_tracking": count * total,
      },
      "metric_mean_sums": {
        "reward_per_step": count * total / steps,
        "linear_tracking": count * total / steps,
        "linear_velocity_error": count * error,
      },
    }
  }


class EpisodeStatisticsTests(unittest.TestCase):
  def test_unfinished_episodes_are_not_published_and_partial_reset_is_local(self):
    stats = EpisodeStatistics(2, "cpu")
    tasks = {"velocity": torch.arange(2)}
    metrics = {
      "velocity/linear_tracking": torch.tensor([0.04, 0.02]),
      "velocity/linear_velocity_error": torch.tensor([0.2, 0.4]),
    }
    stats.add(metrics, tasks, torch.tensor([0.04, 0.02]))
    no = torch.zeros(2, dtype=torch.bool)
    self.assertEqual(
      stats.finish(tasks, no, torch.ones(2, dtype=torch.long), no, no, no, 0.02, 10.0),
      {},
    )
    done = torch.tensor([True, False])
    finished = stats.finish(
      tasks, done, torch.ones(2, dtype=torch.long), no, done, no, 0.02, 10.0
    )
    self.assertEqual(finished["velocity"]["count"], 1)
    self.assertEqual(finished["velocity"]["time_out"], 1)
    self.assertAlmostEqual(finished["velocity"]["metric_sums"]["reward_per_step"], 0.04)
    stats.clear(torch.tensor([0]))
    self.assertEqual(float(stats.sums["velocity/reward_per_step"][0]), 0)
    self.assertAlmostEqual(float(stats.sums["velocity/reward_per_step"][1]), 0.02)
    # Finished reports own their data even after the accumulators are cleared.
    self.assertAlmostEqual(finished["velocity"]["metric_sums"]["reward_per_step"], 0.04)

  def test_episode_means_use_episode_counts_and_reward_rate_uses_nominal_seconds(self):
    episodes = EpisodeSummary()
    episodes.add(report(count=1, error=0.2, total=0.04, steps=1))
    episodes.add(report(count=3, error=0.6, total=0.12, steps=3))
    metrics = classified_metrics(
      {
        "statistics_scope": "completed_episodes",
        "episode_statistics": episodes.summary(),
      }
    )
    self.assertAlmostEqual(metrics["error/velocity/planar_velocity_m_per_s"], 0.5)
    self.assertAlmostEqual(metrics["reward/velocity/total_per_second"], 0.01)
    self.assertAlmostEqual(metrics["reward/velocity/total_per_step"], 0.04)
    self.assertAlmostEqual(metrics["episode/velocity/return_mean"], 0.1)
    self.assertEqual(metrics["episode/velocity/completed"], 4)
    self.assertNotIn("reward/motion/total_per_second", metrics)

  def test_window_ignores_carried_reset_summaries_and_preserves_episode_weights(self):
    window = VelocityMetricWindow()
    base = {
      "iteration": 1,
      "stage": "joint",
      "warmup_completed": 1,
      "learning_rate": 2e-4,
      "seconds": 1.0,
      "collected_transitions": 10,
      "collected_this_iteration": True,
      "statistics_scope": "completed_episodes",
      "episode_statistics": report(),
    }
    window.add(base)
    window.add(base | {"iteration": 2, "collected_this_iteration": False})
    window.add(
      base | {"iteration": 3, "episode_statistics": report(count=3, error=0.6)}
    )
    metrics = classified_metrics(window.summary())
    self.assertEqual(metrics["episode/velocity/completed"], 4)
    self.assertAlmostEqual(metrics["error/velocity/planar_velocity_m_per_s"], 0.5)
    self.assertEqual(metrics["reset/velocity/time_out_count"], 4)

  def test_no_reset_produces_no_reward_error_or_failure_measurement(self):
    metrics = classified_metrics(
      {"statistics_scope": "completed_episodes", "episode_statistics": {}}
    )
    self.assertEqual(metrics["episode/velocity/completed"], 0)
    self.assertFalse(
      any(key.startswith(("reward/", "error/", "failure/")) for key in metrics)
    )
    carried = classified_metrics(
      {
        "statistics_scope": "completed_episodes",
        "episode_statistics": report(),
        "collected_this_iteration": False,
      }
    )
    self.assertNotIn("episode/velocity/completed", carried)

  def test_failure_reason_counts_are_exclusive(self):
    stats = EpisodeStatistics(2, "cpu")
    tasks = {"velocity": torch.arange(2)}
    stats.add({}, tasks, torch.ones(2))
    yes = torch.ones(2, dtype=torch.bool)
    result = stats.finish(
      tasks, yes, torch.ones(2), torch.tensor([True, False]), yes, yes, 0.02, 10.0
    )["velocity"]
    self.assertEqual(
      (result["terminated"], result["motion_end"], result["time_out"]), (1, 1, 0)
    )
