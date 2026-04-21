# Learning to Drive a Simplified F1 Car in MuJoCo

This repository studies continuous control for autonomous racing in a simplified MuJoCo environment. The first milestone is a trustworthy track geometry layer: centerline sampling, progress, heading error, lateral error, and off-track detection.

## Current Status

- Python package scaffold
- Closed-track representation
- Oval and technical track configs
- Unit tests for progress, lateral error, heading error, and off-track checks
- Track plotting script
- Minimal MuJoCo/Gymnasium environment with continuous steering/throttle/brake actions
- Bicycle-style vehicle dynamics with front steering, drivetrain split, tire-force proxies, and action smoothing
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
racesim-evaluate --controller heuristic --episodes 1 --record-trajectory --output results/eval_heuristic_trajectory.json
racesim-plot-rollout --input results/eval_heuristic_trajectory.json --output-dir results/plots
racesim-sweep-vehicle --episodes 3
racesim-render-rollout --controller heuristic --steps 1800
racesim-keyboard-drive
```

## Analysis Outputs

The rollout plotting command produces:

- `trajectory.png`
- `speed_vs_progress.png`
- `controls_vs_progress.png`
- `lateral_error_vs_progress.png`

The vehicle sweep compares `rwd`, `fwd`, and `awd` under low/nominal/high grip scales and writes CSV/JSON summaries to `results/`.

## Inspecting The Simulator

The current car is a simplified direct-force/yaw MuJoCo body, not yet a tire/contact model. See [docs/simulation.md](docs/simulation.md) for what is worth trusting now, what is intentionally simplified, and how to manually drive or render rollouts.

On macOS, the keyboard viewer uses MuJoCo's `mjpython`; `racesim-keyboard-drive` will relaunch itself with it when available. Driving uses `I/K`, `J/U`, and `F/G` instead of WASD because MuJoCo reserves WASD for built-in viewer shortcuts. The default camera is a dynamic chase camera behind the car.

## Near-Term Build Order

1. Validate track math with tests and plots.
2. Add the simplest MuJoCo car scene.
3. Wrap simulator in a Gymnasium-style environment.
4. Implement the heuristic controller before RL.
5. Add PPO training only after the baseline completes laps.
