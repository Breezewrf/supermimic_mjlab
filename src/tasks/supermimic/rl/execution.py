"""Compile tensor kernels without wrapping modules or changing checkpoint keys."""

import torch

from .imagination import critic_loss, gae_targets, imagination_step, reward_terms
from .world_model_training import rollout_states, world_model_loss


class TrainingKernels:
  def __init__(self, actor, world, critic, *, enabled=False, backend="inductor"):
    self.enabled = enabled

    def build(function):
      if not enabled:
        return function
      options = {"triton.cudagraphs": False} if backend == "inductor" else None
      return torch.compile(
        function, backend=backend, fullgraph=True, dynamic=False, options=options
      )

    self.goals = build(actor.reference_features)
    self.world_rollout = build(
      lambda state, actions, context, mask: rollout_states(
        world, state, actions, context, mask
      )
    )
    self.world_loss = build(
      lambda prediction, batch: world_model_loss(
        prediction, batch, analytic_losses=False
      )
    )
    self.imag_step = build(
      lambda state, history, goal, context, quaternion: imagination_step(
        actor, world, state, history, goal, context, quaternion
      )
    )
    self.reward = build(
      lambda state, action, context, reference, previous: reward_terms(
        world, state, action, context, reference, previous
      )
    )
    self.critic_features = build(critic.features)
    # Keep separate entry points for detached targets, state-only gradients and
    # critic weight gradients. Their shapes and autograd guards differ; sharing
    # one entry point exhausts Dynamo's specialization cache across trainers.
    self.critic_targets = build(lambda features: critic.forward_features(features))
    self.critic_bootstrap = build(lambda features: critic.forward_features(features))
    self.critic_value = build(lambda features: critic.forward_features(features))
    self.gae = build(gae_targets)
    self.critic_loss = build(critic_loss)
