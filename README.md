# RaceSim

**RaceSim** is a MuJoCo-backed [Gymnasium](https://gymnasium.farama.org/) racing toolkit for continuous control and reinforcement learning research. It ships a closed-track geometry layer, a transparent bicycle dynamics model, baseline controllers, evaluation suites, and optional PPO training.

> Honest scope: this is a **control / RL environment**, not an F1 tire laboratory. See [docs/simulation.md](docs/simulation.md) and [docs/dynamics-roadmap.md](docs/dynamics-roadmap.md).

[![CI](https://github.com/nedcut/race-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/nedcut/race-sim/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-informational.svg)](CHANGELOG.md)

---

## Why RaceSim

| Need | Built-in answer |
|------|-----------------|
| Gymnasium racing env | `gym.make("RaceSim-v0")` / `RacingEnv` |
| Multi-track curricula | 14-track catalog + `TrackCurriculumEnv` |
| Baselines without training | Centerline / racing-line heuristics |
| Regression telemetry | Physics benchmarks + `racesim suite` |
| PPO research path | Stable-Baselines3 extras + live eval |

## Install

```bash
git clone https://github.com/nedcut/race-sim.git
cd race-sim

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -e ".[dev,rl]"
# or: uv sync --extra dev --extra rl

racesim doctor
pytest -q
```

Extras:

- `dev` — pytest, ruff, build
- `rl` — Stable-Baselines3, torch, tensorboard
- `all` — `dev` + `rl`

Platform notes (macOS `mjpython`, headless Linux): [docs/getting-started.md](docs/getting-started.md).

## Quick start

### Unified CLI

```bash
racesim version
racesim doctor
racesim evaluate -- --controller racing_line --episodes 1 --max-steps 800
racesim physics -- --enforce --benchmarks acceleration braking
racesim suite -- --quick
racesim keyboard -- --help
```

Legacy `racesim-*` entry points remain for scripts and CI.

### Python API

```python
import racesim
import gymnasium as gym
import numpy as np

env = gym.make("RaceSim-v0")  # config via RACESIM_CONFIG or config=...
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(
    np.array([0.0, 0.5, 0.0], dtype=np.float32)
)
env.close()

from racesim import RacingEnv, ClosedTrack, __version__
print(__version__, racesim.project_root())
```

### Examples

```bash
python examples/custom_controller.py
python examples/sb3_train_minimal.py --timesteps 10000  # needs [rl]
```

### Demo policy

A short PPO checkpoint is optional under `artifacts/ppo_oval_quick.zip` (undertrained smoke model).

```bash
racesim evaluate-policy -- --model artifacts/ppo_oval_quick.zip --episodes 1 --deterministic
# regenerate: python scripts/train_quick_checkpoint.py
```

## Features

- **Track layer** — progress, lateral/heading error, off-track, catalog validation  
- **Vehicle presets** — touring / kart / formula with chassis mass + aero parameters  
- **Dynamics telemetry** — slip angles, tire usage, load transfer, understeer score  
- **Eval tooling** — rollouts, plots, vehicle sweeps, multi-seed `configs/eval.yaml`  
- **Training** — curriculum multi-track env, live eval dashboard, portable `device: auto`  
- **Quality gates** — GitHub Actions CI, `--quick` suite, open-loop physics pad  

## Project layout

```text
src/racesim/     Core package (env, eval, training, CLI)
configs/         Env, vehicle, track, and train YAML
assets/mjcf/     MuJoCo car + track worlds
docs/            Manuals and model trust boundaries
examples/        Minimal integration samples
tests/           Pytest suite
artifacts/       Optional demo checkpoints
```

## Documentation

| Doc | Content |
|-----|---------|
| [getting-started.md](docs/getting-started.md) | Install, platforms, first commands |
| [observation-and-action.md](docs/observation-and-action.md) | Observation vector, actions, `info` |
| [config-reference.md](docs/config-reference.md) | Env YAML fields |
| [tracks.md](docs/tracks.md) | Track authoring |
| [simulation.md](docs/simulation.md) | Physics model + trust boundary |
| [dynamics-roadmap.md](docs/dynamics-roadmap.md) | Future dynamics work |
| [roadmap-to-1.0.md](docs/roadmap-to-1.0.md) | Product path: fun manual + race vs agents |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev workflow |
| [CHANGELOG.md](CHANGELOG.md) | Release notes |

## Development

```bash
pip install -e ".[dev,rl]"
ruff check .
pytest
racesim suite -- --quick
python -m build   # sdist + wheel
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Versioning

RaceSim follows [Semantic Versioning](https://semver.org/) for the public Python API (`racesim` package). Gym ID `RaceSim-v0` may evolve observation semantics only with a version bump and changelog entry.

## License

[MIT](LICENSE) © Ned Cutler
