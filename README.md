# Learning to Drive a Simplified F1 Car in MuJoCo

This repository studies continuous control for autonomous racing in a simplified MuJoCo environment. The first milestone is a trustworthy track geometry layer: centerline sampling, progress, heading error, lateral error, and off-track detection.

## Current Status

- Python package scaffold
- Closed-track representation
- Oval and technical track configs
- Unit tests for progress, lateral error, heading error, and off-track checks
- Track plotting script
- Minimal MuJoCo/Gymnasium environment with continuous steering/throttle/brake actions
- Heuristic baseline controller that completes the oval track

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
racesim-plot-track --config configs/tracks/oval.yaml --output results/oval_track.png
racesim-smoke-mujoco
racesim-rollout --controller heuristic --steps 1800
racesim-evaluate --controller heuristic --episodes 5 --output results/eval_heuristic.json
```

## Near-Term Build Order

1. Validate track math with tests and plots.
2. Add the simplest MuJoCo car scene.
3. Wrap simulator in a Gymnasium-style environment.
4. Implement the heuristic controller before RL.
5. Add PPO training only after the baseline completes laps.
