"""Per-environment accumulators published only when real episodes finish."""

import torch


class EpisodeStatistics:
  def __init__(self, num_envs, device):
    self.num_envs, self.device = num_envs, device
    self.sums = {}

  def clear(self, ids):
    for values in self.sums.values():
      values[ids] = 0

  def add(self, metrics, task_ids, reward):
    values = dict(metrics)
    for task, ids in task_ids.items():
      values[f"{task}/reward_per_step"] = reward[ids]
    for key, value in values.items():
      if key.endswith("/physical_failure"):
        continue
      task = key.split("/", 1)[0]
      if key not in self.sums:
        self.sums[key] = torch.zeros(self.num_envs, device=self.device)
      self.sums[key][task_ids[task]] += value

  def finish(
    self, task_ids, done, steps, terminated, time_out, motion_end, dt, normalization_s
  ):
    report = {}
    for task, ids in task_ids.items():
      ids = ids[done[ids]]
      if not len(ids):
        continue
      lengths = steps[ids].float()
      keys = [key for key in self.sums if key.startswith(task + "/")]
      sums = torch.stack([self.sums[key][ids] for key in keys])
      # Transfer all completed-episode metrics together, not scalar-by-scalar.
      packed = torch.stack((sums.sum(-1), (sums / lengths).sum(-1))).cpu()
      report[task] = {
        "count": len(ids),
        "steps_sum": int(lengths.sum()),
        "duration_s_sum": float(lengths.sum()) * dt,
        "terminated": int(terminated[ids].sum()),
        "time_out": int((time_out[ids] & ~terminated[ids] & ~motion_end[ids]).sum()),
        "motion_end": int((motion_end[ids] & ~terminated[ids]).sum()),
        "normalization_s": normalization_s,
        "metric_sums": {
          key.split("/", 1)[1]: float(packed[0, i]) for i, key in enumerate(keys)
        },
        "metric_mean_sums": {
          key.split("/", 1)[1]: float(packed[1, i]) for i, key in enumerate(keys)
        },
      }
    return report


class EpisodeSummary:
  """Merge reset events with episode counts as the statistical weights."""

  def __init__(self):
    self.tasks = {}

  def add(self, report):
    for task, row in report.items():
      if task not in self.tasks:
        self.tasks[task] = {
          "normalization_s": row["normalization_s"],
          "metric_sums": {},
          "metric_mean_sums": {},
        }
      target = self.tasks[task]
      if target["normalization_s"] != row["normalization_s"]:
        raise ValueError("Cannot mix different episode reward normalizations")
      for key, value in row.items():
        if key in ("metric_sums", "metric_mean_sums"):
          for name, number in value.items():
            target[key][name] = target[key].get(name, 0.0) + number
        elif key != "normalization_s":
          target[key] = target.get(key, 0) + value

  def summary(self):
    return {
      task: {
        key: dict(value) if isinstance(value, dict) else value
        for key, value in row.items()
      }
      for task, row in self.tasks.items()
    }
