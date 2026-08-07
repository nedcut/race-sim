# Learning to Drive a Simplified F1 Car in MuJoCo

This repository studies continuous control for autonomous racing in a simplified MuJoCo environment. The first milestone is a trustworthy track geometry layer: centerline sampling, progress, heading error, lateral error, and off-track detection.

## Status

- MuJoCo / Gymnasium racing env with continuous steering, throttle, and brake
- Closed-track geometry, multi-track catalog, and matching MJCF visual worlds
- Bicycle-style vehicle dynamics (load transfer, tire saturation, drivetrain split, aero)
- Heuristic baselines (centerline / racing line), keyboard drive, evaluation and physics tools
- Optional PPO training via Stable-Baselines3 (`[rl]` extra)
- Unit tests, track validation, and baseline reporting under `results/baselines.md`
- Onboarding tools: `racesim-doctor`, `docs/getting-started.md`, optional quick PPO checkpoint under `artifacts/`

## First 10 minutes

**Prerequisites:** Python **3.11+**. Optional OpenGL / desktop display for the MuJoCo viewer (keyboard drive and some render scripts). Full platform notes: [docs/getting-started.md](docs/getting-started.md).

```bash
# Clone and enter the repo (always run CLIs from this root)
cd race-sim   # or your local path

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Editable install: dev tests + RL tooling
pip install -e ".[dev,rl]"
# If you use uv (repo includes uv.lock):
#   uv sync --extra dev --extra rl

# Environment check
racesim-doctor

# Smoke tests
pytest

# CLI help + baseline evaluate
racesim-keyboard-drive --help
racesim-evaluate --controller racing_line --episodes 1 --output results/eval_racing_line.json
```

Docs:

- [docs/getting-started.md](docs/getting-started.md) — install, macOS `mjpython`, Linux headless, working directory
- [docs/simulation.md](docs/simulation.md) — vehicle model and what claims are safe

Optional: if `artifacts/ppo_oval_quick.zip` exists (or after you train it — see below), evaluate the tiny demo policy:

```bash
racesim-evaluate-policy --model artifacts/ppo_oval_quick.zip --config configs/env.yaml --episodes 2 --deterministic
```

Generate that checkpoint (short PPO run; needs `[rl]`):

```bash
python scripts/train_quick_checkpoint.py
# or:
racesim-train-ppo --config configs/train_ppo_oval_quick.yaml
cp results/ppo_oval_quick/final_model.zip artifacts/ppo_oval_quick.zip
```

## Advanced CLI

```bash
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

For a minimal onboarding run (a few thousand timesteps into `artifacts/`):

```bash
python scripts/train_quick_checkpoint.py
racesim-evaluate-policy --model artifacts/ppo_oval_quick.zip --config configs/env.yaml --episodes 2 --deterministic
```

Available training configs:

- `configs/train_ppo_oval.yaml`
- `configs/train_ppo_oval_quick.yaml` (short demo train for `artifacts/ppo_oval_quick.zip`)
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

## License

MIT — see [LICENSE](LICENSE).
