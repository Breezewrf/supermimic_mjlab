"""Mixed precision for MLPs only; physical computations keep their input dtype."""

import torch
from torch import Tensor, nn


def run_mlp(network: nn.Module, inputs: Tensor, amp: str) -> Tensor:
  if amp == "bf16" and inputs.is_cuda:
    with torch.autocast("cuda", dtype=torch.bfloat16):
      return network(inputs).float()
  return network(inputs)
