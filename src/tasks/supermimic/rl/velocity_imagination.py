"""Differentiable velocity rollouts with shared command scheduling and task heads."""

import torch

from ..mdp.state import add_sensor_noise, observe
from ..mdp.velocity import velocity_failure, velocity_reward


def velocity_step(actor, encoder, world, state, history, command, context):
  latent = encoder(actor.normalize_history(history), command)
  action = actor.decode(history, latent)
  next_state = world(state, action, context)
  observation = observe(next_state, action, actor.default_q, context)
  return next_state, action, observation, latent


def velocity_signals(world, state, command, action, previous_action, cfg):
  reward, _ = velocity_reward(
    state,
    world.kinematics(state),
    command,
    action,
    previous_action,
    world.kinematics.joint_limits,
    world.kinematics.default_q,
    cfg,
  )
  return reward, velocity_failure(state, cfg)["termination_probability"]


class VelocityKernels:
  def __init__(self, actor, encoder, world, cfg, *, enabled=False, backend="inductor"):
    def build(fn):
      return (
        torch.compile(fn, backend=backend, fullgraph=True, dynamic=False)
        if enabled
        else fn
      )

    self.step = build(
      lambda state, history, command, context: velocity_step(
        actor, encoder, world, state, history, command, context
      )
    )
    self.signals = build(
      lambda state, command, action, previous: velocity_signals(
        world, state, command, action, previous, cfg
      )
    )


def imagine_velocity(
  actor,
  encoder,
  world,
  critic,
  commands,
  starts,
  *,
  horizon=32,
  gamma=0.99,
  generator=None,
  sensor_noise=True,
  control_step=0,
  reference_only=False,
  kernels=None,
):
  if horizon < 1:
    raise ValueError("Imagination horizon must be positive")
  cfg = commands.cfg
  sequence = commands.sequence(
    starts, horizon, generator, control_step, reference_only=reference_only
  )
  state, history = starts["state"].detach(), starts["history"].detach()
  context = starts["context"].detach()
  previous = history[:, -1, -29:]
  states, actions, latents, rewards, continuation, features, values = (
    [state],
    [],
    [],
    [],
    [],
    [],
    [],
  )
  alignment_targets = torch.zeros(len(state), horizon, 32, device=state.device)
  alignment_mask = sequence["command_source"][:, :horizon]
  # Only real reference-source goals are encoded; random commands have no teacher.
  with torch.no_grad():
    if alignment_mask.any():
      reference = commands.library.get_future(
        sequence["reference_motion"][:, :horizon][alignment_mask],
        sequence["reference_frame"][:, :horizon][alignment_mask],
        10,
      )
      goal = actor.reference_features(reference, actor.default_q)
      alignment_targets[alignment_mask] = actor.encode_motion(goal)
  for t in range(horizon):
    command = sequence["command"][:, t]
    with torch.no_grad():
      f = critic.features(state, command, previous, context)
      features.append(f)
      values.append(critic.forward_features(f))
    if kernels is None:
      next_state, action, observation, latent = velocity_step(
        actor, encoder, world, state, history, command, context
      )
      reward, failure = velocity_signals(
        world, next_state, command, action, previous, cfg
      )
    else:
      next_state, action, observation, latent = kernels.step(
        state, history, command, context
      )
      reward, failure = kernels.signals(next_state, command, action, previous)
    if sensor_noise:
      observation = add_sensor_noise(observation, generator)
    history = torch.cat((history[:, 1:], observation[:, None]), 1)
    state, previous = next_state, action
    states.append(state)
    actions.append(action)
    latents.append(latent)
    rewards.append(reward)
    continuation.append(1 - failure)
  terminal = critic(state, sequence["command"][:, -1], previous, context)
  values.append(terminal.detach())
  rewards, continuation = torch.stack(rewards, 1), torch.stack(continuation, 1)
  alive = continuation.cumprod(1)
  before = torch.cat((torch.ones_like(alive[:, :1]), alive[:, :-1]), 1)
  discount = gamma ** torch.arange(horizon, device=state.device, dtype=state.dtype)
  reward_objective = (discount * before * rewards).sum(1)
  bootstrap_objective = gamma**horizon * alive[:, -1] * terminal
  mse = (torch.stack(latents, 1) - alignment_targets).square().mean(-1)
  alignment = (mse * alignment_mask).sum() / alignment_mask.sum().clamp_min(1)
  return {
    "objective": (reward_objective + bootstrap_objective).mean(),
    "reward_objective": reward_objective.mean(),
    "bootstrap_objective": bootstrap_objective.mean(),
    "alignment_loss": alignment,
    "alignment_samples": alignment_mask.sum(),
    "rewards": rewards,
    "continuation": continuation,
    "values": torch.stack(values, 1),
    "critic_features": torch.stack(features, 1),
    "mask": torch.ones_like(rewards, dtype=torch.bool),
    "survival": alive[:, -1],
    "states": torch.stack(states, 1),
    "actions": torch.stack(actions, 1),
    "history": history,
    "commands": sequence["command"],
  }
