# Learning to Drive a Simplified F1 Car in MuJoCo

This repository studies continuous control for autonomous racing in a simplified MuJoCo environment. The first milestone is a trustworthy track geometry layer: centerline sampling, progress, heading error, lateral error, and off-track detection.

## Current Status

- Python package scaffold
- Closed-track representation
- 14 track configs with matching MuJoCo visual worlds
- Unit tests for track geometry, reset randomization, vehicle dynamics, evaluation tools, and training helpers
- Track plotting, validation, rollout rendering, comparison, and smoke-test scripts
- Minimal MuJoCo/Gymnasium environment with continuous steering/throttle/brake actions
- Bicycle-style vehicle dynamics with front steering, drivetrain split, load transfer, smooth tire saturation, combined tire-force limits, brake bias, aero downforce, and action smoothing
- Vehicle presets for touring, kart, and formula-style setups
- Centerline and racing-line heuristic baselines for smoke testing
- Track catalog with spline-based layouts for curriculum/generalization, plus harder held-out layouts
- Reset randomization for progress, lateral offset, heading error, initial speed, and grip

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check .
racesim-validate-tracks
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
racesim-physics-benchmarks --output results/physics_benchmarks.json
racesim-eval-suite --output results/baselines.md
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
- `configs/train_ppo_staged_progress_1m.yaml`
- `configs/train_ppo_nominal_beefy_1m.yaml`
- `configs/train_ppo_blind_grip_095_105_beefy_1m.yaml`

The staged-progress 1M config uses Stable-Baselines3's default `MlpPolicy`
network: separate actor and critic MLPs with `[64, 64]` hidden layers and Tanh
activations. The nominal and blind-grip configs use larger separate actor and
critic networks with `[256, 256, 128]` hidden layers and request `device: mps`
for Apple Silicon training. They use four subprocess environments and save both
`final_model.zip` and the best live-eval checkpoint as `best_model.zip`. The
blind-grip config samples `grip_scale` uniformly from `0.95` to `1.05` at reset,
but does not add grip to the observation, so the policy must infer grip from
vehicle behavior.

## Analysis Outputs

The rollout plotting command produces:

- `trajectory.png`
- `speed_vs_progress.png`
- `controls_vs_progress.png`
- `lateral_error_vs_progress.png`

The vehicle sweep compares `rwd`, `fwd`, and `awd` under low/nominal/high grip scales and writes CSV/JSON summaries to `results/`.

The physics benchmark command runs deterministic open-loop acceleration, braking, steady-turning, skidpad, step-steer, slalom, braking-turn, throttle-exit, and repeatability checks. These are telemetry baselines rather than claims of real vehicle fidelity.

The eval suite command runs tests, lint, catalog validation, controller smoke tests, default controller evals, physics telemetry, and the saved PPO policy matrix when the model artifact is present. It writes the tracked baseline report at [results/baselines.md](results/baselines.md).

The comparison plotting command overlays recorded controller trajectories and writes a compact summary table. The technical, street-circuit, grand-prix, kartplex, and endurance layouts are useful held-out challenge cases for controllers and learned policies.

The track catalog lives in [configs/track_catalog.yaml](configs/track_catalog.yaml). Use `racesim-smoke-tracks` to quickly see which tracks are easy, intermediate, or failure cases for a controller. At a 1200-step racing-line smoke horizon, most shorter tracks complete, while larger layouts such as grand prix and endurance usually need longer horizons or a stronger policy.

## Inspecting The Simulator

The current car is still a simplified free-body MuJoCo model, not a wheel/contact tire simulation. The environment applies forces from a bicycle-style tire proxy with axle loads, combined tire limits, and vehicle presets. See [docs/simulation.md](docs/simulation.md) for what is worth trusting now, what is intentionally simplified, and how to manually drive or render rollouts.

On macOS, the keyboard viewer uses MuJoCo's `mjpython`; `racesim-keyboard-drive` will relaunch itself with it when available. Driving uses `I/K`, `J/U`, and `F/G` instead of WASD because MuJoCo reserves WASD for built-in viewer shortcuts. The default camera is a dynamic chase camera behind the car, and `H` toggles the selected autopilot controller. The default autopilot is `racing_line`; use `--autopilot centerline` for the slower centerline follower.

## Near-Term Build Order

1. Keep the track catalog and MuJoCo visual worlds validated together.
2. Treat physics benchmarks as regression telemetry while tuning the simplified tire model.
3. Improve controller coverage on the hard catalog tracks before expanding curricula further.
4. Re-run policy evaluation after physics or reward changes so old PPO artifacts do not become misleading.
5. Add richer tire/contact modeling only after the current proxy is well characterized.
