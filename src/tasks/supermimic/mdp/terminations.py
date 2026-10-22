"""Physical failure criteria shared by simulation and imagination."""

from torch import Tensor

from .state import quat_inverse_apply


def failure_terms(state: Tensor, reference: dict[str, Tensor]) -> dict[str, Tensor]:
  gravity = state.new_tensor([0, 0, -1])
  height = (state[..., 2] - reference["body_pos_w"][..., 0, 2]).abs()
  tilt = (
    quat_inverse_apply(state[..., 3:7], gravity)[..., 2]
    - quat_inverse_apply(reference["body_quat_w"][..., 0, :], gravity)[..., 2]
  ).abs()
  return {
    "height_error": height,
    "tilt_error": tilt,
    "height_failure": height > 0.25,
    "tilt_failure": tilt > 0.8,
  }
