# Getting Started

This guide covers platform setup, install extras, and common gotchas when running RaceSim from a fresh clone. Run all commands from the **repository root** so relative paths like `configs/env.yaml` and `assets/mjcf/...` resolve correctly.

## Prerequisites

- **Python 3.11+**
- Optional OpenGL / display for the MuJoCo interactive viewer (`racesim-keyboard-drive`, render scripts)
- For training: install the `rl` extra (`stable-baselines3`, `torch`)

```bash
pip install -e ".[dev,rl]"
# or, if you use uv (this repo has uv.lock):
uv sync --extra dev --extra rl
```

Sanity check:

```bash
racesim doctor
# or: racesim-doctor
pytest
```

If you need tools from outside the repo checkout, set:

```bash
export RACESIM_ROOT=/path/to/race-sim
```

## Repository working directory

Always launch CLIs and tests from the package root (the directory that contains `configs/`, `assets/`, and `pyproject.toml`). Config files and MJCF assets use repo-relative paths; running from another directory will fail with missing-file errors.

## macOS

Interactive keyboard driving uses MuJoCo’s native viewer. On macOS, Apple requires a special process entry point (`mjpython`) for GLFW viewer apps.

`racesim-keyboard-drive` **auto-relaunches** under `mjpython` when available so you do not need to remember the invocation. Pass `--no-reexec` only if you intentionally manage that yourself.

Install notes:

- Prefer a normal CPython 3.11+ venv; MuJoCo ships platform wheels for macOS.
- If the viewer fails to open, confirm `mjpython` is on `PATH` (it usually arrives with the `mujoco` package) and that your session has a graphical desktop (not pure SSH without display forwarding).

## Linux

You need a working OpenGL context for the interactive viewer:

- On a desktop machine, install system OpenGL / GLFW-related packages if the MuJoCo viewer fails at import or launch.
- On headless servers or CI, **skip the viewer**. Prefer non-GUI workflows:

  ```bash
  racesim-evaluate --controller racing_line --episodes 1 --output results/eval_racing_line.json
  racesim-physics-benchmarks --output results/physics_benchmarks.json
  racesim-rollout --controller heuristic --steps 400
  ```

  These use the physics and evaluation stack without opening a MuJoCo window.

Display diagnostic: `racesim-doctor` checks for a set `DISPLAY` (and related clues) and reports viewer readiness as a soft warning when display/OpenGL look unavailable.

## RL extra (`train-ppo`)

`racesim-train-ppo`, `racesim-evaluate-policy`, and Stable-Baselines3 PPO paths require the optional `rl` extras:

```bash
pip install -e ".[rl]"
# or
uv sync --extra rl
```

Without `torch` / `stable-baselines3`, environment rollouts and heuristic evaluation still work; PPO training and policy evaluation will fail at import.

Quick oval train for a tiny onboarding checkpoint:

```bash
python scripts/train_quick_checkpoint.py
# equivalent one-liner after editing timesteps if preferred:
racesim train -- --config configs/train_ppo_oval_quick.yaml
```

This writes `artifacts/ppo_oval_quick.zip` when training succeeds.

For an experimental quality stack (VecNormalize, schedules, richer obs) see
[docs/training.md](training.md). There is no completed high-quality trained
baseline yet.

```bash
racesim train -- --config configs/train_ppo_quality.yaml
```

## Gymnasium

```python
import racesim  # registers RaceSim-v0 on import
import gymnasium as gym

env = gym.make("RaceSim-v0")
# override: gym.make("RaceSim-v0", config="configs/env_s_curve.yaml")
# or: export RACESIM_CONFIG=configs/env_hairpin.yaml
```

Entry point registration is also declared in `pyproject.toml` under
`[project.entry-points."gymnasium.envs"]` so editable installs pick it up after
`pip install -e .`.

## Next steps

- [README.md](../README.md) — First 10 minutes and advanced CLI
- [docs/training.md](training.md) — PPO, normalize, BC warm-start
- [docs/simulation.md](simulation.md) — Current vehicle/physics model limits
- [docs/observation-and-action.md](observation-and-action.md) — Observation / action / info
- [docs/config-reference.md](config-reference.md) — Env YAML keys
- [docs/tracks.md](tracks.md) — Authoring tracks
