"""Run a policy in real physics with a live MuJoCo reference comparison.

Uses the same checkpoint checks, fixed starts, and rewards as the
headless evaluator. Mouse forces are enabled by default: double-click a body and
Ctrl-drag to perturb it. Use --no-perturbations to disable interactive forces,
--loop for continuous playback, or --video for MP4 export.
Height/tilt failure resets are disabled by default; --termination restores them.
Use --world-model <weights.pt> for a future policy rollout (default: +32 steps).
Use --world-model-mode compare for same-time dynamics-error comparison.
"""

from evaluate_supermimic import main

if __name__ == "__main__":
  main(playback=True)
