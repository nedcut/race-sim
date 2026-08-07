# Contributing to RaceSim

Thanks for improving RaceSim. This guide covers the development loop for a clean, reviewable change.

## Development setup

```bash
git clone https://github.com/nedcut/race-sim.git
cd race-sim
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,rl]"
racesim doctor
```

Always run tools from the repository root (or set `RACESIM_ROOT`).

## Workflow

1. Create a focused branch from `main`.
2. Prefer small, single-purpose PRs.
3. Keep physics claims honest: update `docs/simulation.md` when behavior changes.
4. Regenerate or note baseline impact when dyn/reward behavior changes.

## Quality gates

```bash
ruff check .
ruff format --check .
pytest -q
racesim suite -- --quick
python -m build
```

CI runs these on pull requests. Local full suite:

```bash
racesim suite -- --output results/baselines.md
```

## Style

- Python 3.11+, type-friendly code
- Ruff line length 100
- Prefer absolute package imports (`from racesim...`)
- No silent behavior changes without tests

## Public API

Stable surface for external users (prefer these in examples/docs):

- `racesim.RacingEnv`, `racesim.ClosedTrack`
- `racesim.make_race_sim_env`, `racesim.register_racesim_envs`
- `racesim.__version__`, `racesim.project_root`, `racesim.resolve_resource`
- Gym ID `RaceSim-v0`
- Unified CLI: `racesim <subcommand>`

Internal modules under `racesim.eval`, `racesim.training`, etc. may change more freely.

## Reporting issues

Include:

- OS and Python version
- `racesim doctor` output
- Command that failed
- Whether you used the `[rl]` extra

## License

By contributing, you agree that your contributions are licensed under the MIT License.
