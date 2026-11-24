"""Write local scalar events from the shared training metric schema."""

import math

from .metric_schema import classified_metrics


class TensorBoardReporter:
  def __init__(self, output, *, purge_step=None):
    from torch.utils.tensorboard import SummaryWriter

    self.path = output / "tensorboard"
    self.writer = SummaryWriter(log_dir=str(self.path), purge_step=purge_step)

  def _scalars(self, metrics, iteration):
    for tag, value in metrics.items():
      if isinstance(value, (int, float)) and math.isfinite(value):
        self.writer.add_scalar(tag, value, global_step=iteration)

  def training(self, record):
    metrics = classified_metrics(record)
    for key, tag in {
      "environment_steps": "train/environment_steps",
      "world_updates": "train/world_updates",
      "actor_updates": "train/actor_updates",
      "critic_updates": "train/critic_updates",
      "replay_transitions": "replay/transitions",
      "learning_rate": "optim/learning_rate",
      "seconds": "performance/iteration_seconds",
      "world_model_seconds": "performance/world_model_seconds",
      "policy_and_critic_seconds": "performance/policy_critic_seconds",
      "peak_gpu_memory_gib": "memory/torch_peak_allocated_gib",
    }.items():
      if key in record:
        metrics[tag] = record[key]
    self._scalars(metrics, record["iteration"])

  def window(self, record):
    throughput = record.get(
      "window_transitions_per_second", record.get("steps_per_second")
    )
    self._scalars(
      {"performance/window_transitions_per_second": throughput}, record["iteration"]
    )
    self.writer.flush()

  def evaluation(self, report, iteration):
    self._scalars(
      {
        f"evaluation/{key}": report[key]
        for key in (
          "normalized_return",
          "normalized_return_framew",
          "successes",
          "success_rate",
          "n_failed",
          "mean_completion",
          "evaluation_complete",
          "evaluation_num_envs",
          "evaluation_wall_seconds",
          "training_environment_steps",
        )
        if key in report
      },
      iteration,
    )
    self.writer.flush()

  def close(self):
    self.writer.close()
