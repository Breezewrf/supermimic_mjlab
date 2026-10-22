"""Differentiable on-policy imagination, soft survival and detached GAE targets."""

from contextlib import contextmanager

import torch
from torch import Tensor

from ..mdp.state import add_sensor_noise
from ..models.policy import reference_features


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
  return error.square().sum(dim=1).mean()


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
) -> dict:
  if horizon < 1:
    raise ValueError("Imagination horizon must be positive")
  state, history = starts["state"].detach(), starts["history"].detach()
  context, motion, frame = starts["context"], starts["motion_id"], starts["frame_id"]
  # Include the final state's full future window for the terminal bootstrap.
  future = library.get_future(motion, frame, horizon + 10)
  # A simulator collection timeout is a truncation, not a physical failure.
  # Imagination follows the reference independently of the real episode clock.
  budget = library.lengths[motion] - 1 - frame
  if (budget <= 0).any():
    raise ValueError("An imagined start must have a successor")
  survival = state.new_ones(len(state))
  objective = state.new_zeros(len(state))
  rewards, continuations, values, features, masks = [], [], [], [], []
  actions, states = [], [state]
  for t in range(horizon):
    reference = {key: value[:, t : t + 10] for key, value in future.items()}
    goal = reference_features(reference, actor.default_q)
    critic_input = critic.features(state, reference, context, goal)
    features.append(critic_input.detach())
    with torch.no_grad():
      values.append(critic.value(critic_input.detach()).squeeze(-1))
    active = t < budget
    action = torch.where(active[:, None], actor.forward_features(history, goal), 0.0)
    predicted_state = world(state, action, context)
    next_state = torch.where(active[:, None], predicted_state, state)
    prediction = world.task_heads(
      next_state,
      action,
      context,
      {key: value[:, t + 1] for key, value in future.items()},
      history[:, -1, -29:],
    )
    reward = prediction["reward"] * active
    continuation = (1 - prediction["termination_probability"]) * ((t + 1) < budget)
    objective = objective + gamma**t * survival * reward
    survival = survival * continuation
    observation = prediction["clean_observation"]
    if sensor_noise:
      observation = add_sensor_noise(observation, generator)
    appended = torch.cat((history[:, 1:], observation[:, None]), 1)
    history = torch.where(active[:, None, None], appended, history)
    state = next_state
    rewards.append(reward)
    continuations.append(continuation)
    masks.append(active)
    actions.append(action)
    states.append(state)
  terminal_reference = {
    key: value[:, horizon : horizon + 10] for key, value in future.items()
  }
  terminal_value = critic(state, terminal_reference, context)
  objective = objective + gamma**horizon * survival * terminal_value
  values.append(terminal_value.detach())
  return {
    "objective": objective.mean(),
    "rewards": torch.stack(rewards, 1),
    "continuation": torch.stack(continuations, 1),
    "values": torch.stack(values, 1),
    "critic_features": torch.stack(features, 1),
    "mask": torch.stack(masks, 1),
    "survival": survival,
    "states": torch.stack(states, 1),
    "actions": actions,
    "history": history,
  }
