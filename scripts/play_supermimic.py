"""Run a policy in real physics with a live MuJoCo reference comparison.

Uses the same checkpoint checks, fixed starts, rewards and termination as the
headless evaluator. Use --loop for continuous playback or --video for MP4 export.
"""

from evaluate_supermimic import main

if __name__ == "__main__":
  main(playback=True)
