"""Classify training metrics without confusing task rewards with physical errors.

Reward terms are already weighted and dt-scaled in the MDP. Errors retain their
physical units. Objectives are imagined discounted returns, not real step rewards.
"""

MOTION_REWARDS = (
  "body_position",
  "body_orientation",
  "body_linear_velocity",
  "body_angular_velocity",
  "root_position",
  "root_orientation",
  "action_smoothness",
  "joint_limit",
)
VELOCITY_REWARDS = (
  "linear_tracking",
  "yaw_tracking",
  "action_smoothness",
  "joint_limit",
  "torso_sideways",
  "torso_angular_velocity",
  "stand_still",
)
MOTION_ERRORS = {
  "body_position_error": "body_position_m",
  "body_orientation_error": "body_orientation_rad",
  "root_position_error": "root_position_m",
  "root_orientation_error": "root_orientation_rad",
}
VELOCITY_ERRORS = {
  "linear_velocity_error": "planar_velocity_m_per_s",
  "yaw_rate_error": "yaw_rate_rad_per_s",
}
UPDATE_METRICS = {
  "wm_loss": "loss/world_model/total",
  "wm_motion_loss": "loss/world_model/motion",
  "wm_velocity_loss": "loss/world_model/velocity",
  "wm_state": "loss/world_model/state",
  "wm_termination": "loss/world_model/termination",
  "wm_observation": "error/world_model/observation_prediction",
  "wm_reward": "error/world_model/reward_prediction",
  "wm_gradient_norm": "gradient/world_model/preclip_norm",
  "actor_objective": "objective/motion/return",
  "actor_reward_objective": "objective/motion/reward_return",
  "actor_bootstrap_objective": "objective/motion/bootstrap_value",
  "critic_loss": "loss/critic/motion",
  "actor_gradient_norm": "gradient/motion/actor_preclip_norm",
  "critic_gradient_norm": "gradient/motion/critic_preclip_max_norm",
  "imagined_reward": "diagnostic/motion/imagined_reward_per_step",
  "survival_at_horizon": "diagnostic/motion/imagined_survival",
}
TASK_UPDATE_METRICS = {
  "objective": "objective/{task}/return",
  "reward_objective": "objective/{task}/reward_return",
  "bootstrap_objective": "objective/{task}/bootstrap_value",
  "alignment_loss": "loss/{task}/latent_alignment_mse",
  "critic_loss": "loss/critic/{task}",
  "actor_gradient_norm": "gradient/{task}/actor_preclip_norm",
  "survival": "diagnostic/{task}/imagined_survival",
  "warmup_skipped": "diagnostic/{task}/warmup_skipped",
}


def is_collection_metric(key):
  if key.startswith("real_"):
    return True
  task, _, name = key.partition("/")
  if task not in ("motion", "velocity"):
    return False
  return name in (
    *MOTION_REWARDS,
    *VELOCITY_REWARDS,
    *MOTION_ERRORS,
    *VELOCITY_ERRORS,
    "failure_rate",
    "physical_failure",
    "action_change",
  )


def classified_metrics(record, *, include_collection=None):
  """Accept historical flat logs as well as current training records."""
  result = {
    target: record[key]
    for key, target in UPDATE_METRICS.items()
    if key in record and record[key] is not None
  }
  for task in ("motion", "velocity"):
    for key, target in TASK_UPDATE_METRICS.items():
      if f"{task}/{key}" in record:
        result[target.format(task=task)] = record[f"{task}/{key}"]
  if include_collection is None:
    include_collection = record.get("collected_this_iteration", True)
    if "window_transitions_per_second" in record:
      include_collection = record.get("real_reward_per_step") is not None
  if not include_collection:
    return result

  for task, terms, errors in (
    ("motion", MOTION_REWARDS, MOTION_ERRORS),
    ("velocity", VELOCITY_REWARDS, VELOCITY_ERRORS),
  ):
    values = {}
    for term in terms:
      key = f"{task}/{term}"
      if key not in record and task == "motion":
        key = f"real_{term}"
      if key in record and record[key] is not None:
        values[term] = record[key]
        result[f"reward/{task}/{term}"] = record[key]
    if len(values) == len(terms):
      result[f"reward/{task}/total_per_step"] = sum(values.values())
    for key, label in errors.items():
      source = f"{task}/{key}"
      if source not in record and task == "motion":
        source = f"real_{key}"
      if source in record and record[source] is not None:
        result[f"error/{task}/{label}"] = record[source]
  if "real_reward_per_step" in record and record["real_reward_per_step"] is not None:
    result["reward/motion/total_per_step"] = record["real_reward_per_step"]
  if "velocity/action_change" in record:
    result["diagnostic/velocity/action_change_squared"] = record[
      "velocity/action_change"
    ]
  for source, task in (
    ("velocity/physical_failure", "velocity"),
    ("motion/failure_rate", "motion"),
  ):
    if source in record:
      result[f"failure/{task}/events_per_transition"] = record[source]
  if "physical_failures" in record and record.get("collected_transitions", 0):
    result["failure/motion/events_per_transition"] = (
      record["physical_failures"] / record["collected_transitions"]
    )
  if record.get("failure_events_per_transition") is not None:
    result["failure/motion/events_per_transition"] = record[
      "failure_events_per_transition"
    ]
  return result


def metric_groups(record):
  groups = {}
  for path, value in classified_metrics(record).items():
    group, _, name = path.partition("/")
    groups.setdefault(group, {})[name] = value
  return groups
