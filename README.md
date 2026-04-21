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
- Centerline and racing-line heuristic baselines that complete the oval track
- Track catalog with spline-based layouts for curriculum/generalization
- Reset randomization for progress, lateral offset, heading error, and initial speed

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
racesim-evaluate --controller centerline --episodes 1 --record-trajectory --output results/eval_centerline_trajectory.json
racesim-evaluate --controller racing_line --episodes 1 --record-trajectory --output results/eval_racing_line_trajectory.json
racesim-compare-rollouts --input results/eval_centerline_trajectory.json results/eval_racing_line_trajectory.json --output-dir results/comparison
racesim-sweep-vehicle --episodes 3
racesim-smoke-tracks --controller racing_line --steps 1200 --output results/track_smoke.csv
racesim-render-rollout --controller heuristic --steps 1800
racesim-make-track-visual --track configs/tracks/technical.yaml --output assets/mjcf/technical_track.xml --model-name technical_track_visuals
racesim-keyboard-drive
```

## PPO Training

Start with the oval-only PPO config:

```bash
racesim-train-ppo --config configs/train_ppo_oval.yaml
```

Watch live eval rollouts in another terminal:

```bash
racesim-live-dashboard --watch results/ppo_oval/live/latest_rollout.json
```

Evaluate a trained policy:

```bash
racesim-evaluate-policy --model results/ppo_oval/final_model.zip --config configs/env.yaml --episodes 5 --deterministic --record-trajectory --output results/eval_ppo_oval.json
```

Available training configs:

- `configs/train_ppo_oval.yaml`
- `configs/train_ppo_easy.yaml`
- `configs/train_ppo_curriculum.yaml`

## Analysis Outputs

The rollout plotting command produces:

- `trajectory.png`
- `speed_vs_progress.png`
- `controls_vs_progress.png`
- `lateral_error_vs_progress.png`

The vehicle sweep compares `rwd`, `fwd`, and `awd` under low/nominal/high grip scales and writes CSV/JSON summaries to `results/`.

The comparison plotting command overlays recorded controller trajectories and writes a compact summary table. The technical track config/world are available through `configs/env_technical.yaml`; the current heuristics do not complete it reliably yet, which makes it a useful held-out challenge.

The track catalog lives in [configs/track_catalog.yaml](configs/track_catalog.yaml). Use `racesim-smoke-tracks` to quickly see which tracks are easy, intermediate, or failure cases for a controller.

## Inspecting The Simulator

The current car is a simplified direct-force/yaw MuJoCo body, not yet a tire/contact model. See [docs/simulation.md](docs/simulation.md) for what is worth trusting now, what is intentionally simplified, and how to manually drive or render rollouts.

On macOS, the keyboard viewer uses MuJoCo's `mjpython`; `racesim-keyboard-drive` will relaunch itself with it when available. Driving uses `I/K`, `J/U`, and `F/G` instead of WASD because MuJoCo reserves WASD for built-in viewer shortcuts. The default camera is a dynamic chase camera behind the car.

## Near-Term Build Order

1. Validate track math with tests and plots.
2. Add the simplest MuJoCo car scene.
3. Wrap simulator in a Gymnasium-style environment.
4. Implement the heuristic controller before RL.
5. Add PPO training only after the baseline completes laps.
