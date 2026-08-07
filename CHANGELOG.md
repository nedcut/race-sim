# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
