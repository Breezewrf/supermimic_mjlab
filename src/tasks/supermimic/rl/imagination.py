"""Differentiable on-policy imagination, soft survival and detached GAE targets."""

from contextlib import contextmanager

import torch
from torch import Tensor

from ..mdp.state import add_sensor_noise, observe


@contextmanager
def frozen_parameters(*modules):
  """Freeze weights, preserving state/action Jacobians and the caller's flags."""
  parameters = [(p, p.requires_grad) for module in modules for p in module.parameters()]
  try:
    for parameter, _ in parameters:
      parameter.requires_grad_(False)
    yield
  finally:
    for parameter, enabled in parameters:
      parameter.requires_grad_(enabled)


@torch.no_grad()
def gae_targets(
  rewards: Tensor,
  continuation: Tensor,
  values: Tensor,
  gamma: float = 0.99,
  lam: float = 0.95,
) -> Tensor:
  target = values[:, -1]
  targets = []
  for t in range(rewards.shape[1] - 1, -1, -1):
    target = rewards[:, t] + gamma * continuation[:, t] * (
      (1 - lam) * values[:, t + 1] + lam * target
    )
    targets.append(target)
  return torch.stack(list(reversed(targets)), 1)


def critic_loss(values: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
  """Equation (3): sum valid time steps, then average over trajectories."""
  error = torch.where(mask, values - targets, 0.0)
  # FP32 intermediate squares/reductions can overflow even when the final batch
  # mean fits FP32. Keep the same objective and gradients, using FP64 arithmetic
  # until the scalar is formed; genuinely unrepresentable losses still fail.
  return error.double().square().sum(dim=1).mean().to(values.dtype)


def imagination_step(actor, world, state, history, goal, context, reference_quat):
  action = actor.forward_features(history, goal)
  next_state = world(state, action, context)
  observation = observe(
    next_state,
    action,
    actor.default_q,
    context,
    reference_quat if world.relative_orientation else None,
  )
  return next_state, action, observation


def reward_terms(world, state, action, context, reference, previous_action):
  prediction = world.task_heads(state, action, context, reference, previous_action)
  return prediction["reward"], prediction["termination_probability"]


def imagine(
  actor,
  world,
  critic,
  library,
  starts: dict[str, Tensor],
  horizon: int = 32,
  gamma: float = 0.99,
  generator: torch.Generator | None = None,
  sensor_noise: bool = True,
  kernels=None,
) -> dict:
  if horizon < 1:
    raise ValueError("Imagination horizon must be positive")
  state, history = starts["state"].detach(), starts["history"].detach()
  context, motion, frame = (
    starts["context"],
    starts["motion_id"],
    starts["frame_id"],
  )
  # A simulator collection timeout is a truncation, not a physical failure.
  # Imagination follows the reference independently of the real episode clock.
  # Match the reference's inclusive clip-end bound: f + H <= last frame.
  if (frame + horizon >= library.lengths[motion]).any():
    raise ValueError("An imagined start must have a full reference horizon")
  # Future goal features pad at the last frame, including at the bootstrap state.
  future = library.get_future(motion, frame, horizon + 10)
  # These windows are views: (batch, H+1, 10, ...). Reference-only feature
  # construction is shared by actor and critic and is batched across time.
  windows = {
    key: value.unfold(1, 10, 1).movedim(-1, 2) for key, value in future.items()
  }
  if actor.reference_dim != critic.reference_dim:
    raise ValueError("Actor and critic reference dimensions differ")
  goal_fn = kernels.goals if kernels is not None else actor.reference_features
  with torch.no_grad():
    goals = goal_fn(windows, actor.default_q)
  first_action = history[:, -1, -29:]
  actions, states = [], [state]
  for t in range(horizon):
    step_args = (
      state,
      history,
      goals[:, t],
      context,
      future["body_quat_w"][:, t + 1, 0],
    )
    if kernels is None:
      next_state, action, observation = imagination_step(actor, world, *step_args)
    else:
      next_state, action, observation = kernels.imag_step(*step_args)
    if sensor_noise:
      observation = add_sensor_noise(observation, generator)
    history = torch.cat((history[:, 1:], observation[:, None]), 1)
    state = next_state
    actions.append(action)
    states.append(state)
  states = torch.stack(states, 1)
  executed = torch.stack(actions, 1)
  previous = torch.cat((first_action[:, None], executed[:, :-1]), 1)
  contexts = context[:, None].expand(-1, horizon + 1, -1)
  next_reference = {
    key: value[:, 1 : horizon + 1].flatten(0, 1) for key, value in future.items()
  }
  reward_args = (
    states[:, 1:].flatten(0, 1),
    executed.flatten(0, 1),
    contexts[:, 1:].flatten(0, 1),
    next_reference,
    previous.flatten(0, 1),
  )
  if kernels is None:
    rewards, failure = reward_terms(world, *reward_args)
  else:
    rewards, failure = kernels.reward(*reward_args)
  rewards = rewards.reshape(len(state), horizon)
  continuation = 1 - failure.reshape(len(state), horizon)
  alive = continuation.cumprod(1)
  survival = alive[:, -1]
  before = torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), 1)
  discount = gamma ** torch.arange(horizon, device=state.device, dtype=state.dtype)
  reward_objective = (discount * before * rewards).sum(1)
  feature_fn = kernels.critic_features if kernels is not None else critic.features
  value_fn = kernels.critic_targets if kernels is not None else critic.forward_features
  # Detached, batched target values must not train the actor or retain its graph.
  with torch.no_grad():
    features = feature_fn(states.detach(), windows, contexts, goals)
    values = value_fn(features).squeeze(-1)
  terminal_reference = {
    key: value[:, horizon : horizon + 10] for key, value in future.items()
  }
  # Keep the terminal bootstrap differentiable through the final physical state.
  terminal_features = feature_fn(state, terminal_reference, context, goals[:, -1])
  bootstrap_fn = (
    kernels.critic_bootstrap if kernels is not None else critic.forward_features
  )
  terminal_value = bootstrap_fn(terminal_features).squeeze(-1)
  bootstrap_objective = gamma**horizon * survival * terminal_value
  objective = reward_objective + bootstrap_objective
  return {
    "objective": objective.mean(),
    "reward_objective": reward_objective.mean(),
    "bootstrap_objective": bootstrap_objective.mean(),
    "rewards": rewards,
    "continuation": continuation,
    "values": values,
    "critic_features": features[:, :-1],
    "mask": torch.ones_like(rewards, dtype=torch.bool),
    "survival": survival,
    "states": states,
    "actions": actions,
    "history": history,
  }
