# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Optional observation tails: smoothed actuator command, grip scale, tire usage (`configs/env_rl_quality.yaml` → 25-D)
- Reward terms: `lap_complete`, `tire_usage`, `action_rate` (default 0 on classic envs)
- Experimental train pipeline: `VecNormalize`, linear LR schedules (`linear_3e-4`), checkpoints, versioned agent bundles
- Expert data collection (`racesim collect-expert`) and BC pretrain (`racesim bc-pretrain`) with observation normalization + low action `log_std`
- Quality train configs: `train_ppo_quality.yaml`, `train_ppo_quality_smoke.yaml` (not a completed quality model)
- Distinct `pretrained` (policy weights / BC) vs `resume` (full PPO + VecNormalize + timesteps)
- Fail-closed env YAML validation (unknown keys, non-finite numbers, range checks)
- `docs/training.md` training guide (trusted pickle/Cloudpickle checkpoints only)

### Changed
- `include_prev_action` observation tail is the smoothed actuator command, not the raw request
- Policy eval prefers `--bundle`; missing VecNormalize stats are an error when training used normalization
- Curriculum env enforces consistent observation shapes and implements `close()`
- Disjoint default seed ranges for train vs live eval vs final evaluation

## [0.2.0] - 2026-08-07

### Added
- Unified `racesim` CLI (`doctor`, `evaluate`, `physics`, `suite`, `train`, …)
- Package-relative resource resolution (`project_root()`, `RACESIM_ROOT`, packaged `_data`)
- Stable public exports: `RacingEnv`, `ClosedTrack`, `make_race_sim_env`, version helpers
- Wheel packages configs/assets via hatch force-include
- `py.typed` for typed installs
- Professional package metadata (classifiers, URLs, license, keywords)
- CONTRIBUTING guide and this changelog

### Changed
- Version bump from 0.1.0 research scaffold to 0.2.0 toolkit release
- README rewritten as a product / toolkit document
- Doctor defaults to resolved project root (not only CWD)

### Notes
- Legacy `racesim-*` scripts are preserved for script compatibility

## [0.1.0] - 2026-08-07

### Added
- Tire saturation physics, telemetry, physics benchmarks
- Onboarding: MIT license, doctor, getting-started, quick PPO artifact
- CI workflow, quick eval suite, open-loop benchmark pad
- Gymnasium `RaceSim-v0`, integration docs, examples, device auto-resolution
- Render mode, richer metrics, vehicle preset consolidation
- Chassis mass/inertia and aero drag with soft physics ranges

[0.2.0]: https://github.com/nedcut/race-sim/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/nedcut/race-sim/releases/tag/v0.1.0
