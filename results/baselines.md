# RaceSim Baselines

Generated: `2026-04-23T22:12:18+00:00`
Catalog: `configs/track_catalog.yaml`

## Quality Gates

- pytest: pass
- ruff: pass
- track validation: pass

## Default Oval Eval

| controller | completion | off-track | mean lap time | best lap time |
| --- | ---: | ---: | ---: | ---: |
| racing_line | 1.00 | 0.00 | 70.40 | 70.40 |
| heuristic | 1.00 | 0.00 | 72.40 | 72.40 |

## Catalog Smoke (5000 Steps)

### Racing Line

| track | status | steps | lap fraction | speed |
| --- | --- | ---: | ---: | ---: |
| oval | complete | 880 | 1.00 | 2.41 |
| s_curve | complete | 1056 | 1.00 | 2.42 |
| kidney | complete | 1209 | 1.00 | 2.40 |
| chicane | complete | 1158 | 1.00 | 2.41 |
| hairpin | complete | 1086 | 1.00 | 2.40 |
| sweepers | complete | 1189 | 1.00 | 2.60 |
| stop_go | complete | 994 | 1.00 | 2.40 |
| club | complete | 1237 | 1.00 | 2.41 |
| mini_monaco | complete | 1090 | 1.00 | 2.44 |
| technical | off | 1042 | 0.96 | 2.47 |
| grand_prix | complete | 1732 | 1.00 | 3.03 |
| street_circuit | complete | 1523 | 1.00 | 2.40 |
| kartplex | complete | 1106 | 1.00 | 2.41 |
| endurance | complete | 2003 | 1.00 | 4.11 |

### Heuristic

| track | status | steps | lap fraction | speed |
| --- | --- | ---: | ---: | ---: |
| oval | complete | 905 | 1.00 | 2.42 |
| s_curve | complete | 1103 | 1.00 | 2.41 |
| kidney | complete | 1228 | 1.00 | 2.40 |
| chicane | complete | 1169 | 1.00 | 2.42 |
| hairpin | complete | 1123 | 1.00 | 2.41 |
| sweepers | complete | 1249 | 1.00 | 2.41 |
| stop_go | complete | 1019 | 1.00 | 2.41 |
| club | complete | 1271 | 1.00 | 2.40 |
| mini_monaco | complete | 1128 | 1.00 | 2.41 |
| technical | off | 516 | 0.46 | 2.48 |
| grand_prix | complete | 1776 | 1.00 | 2.67 |
| street_circuit | complete | 1552 | 1.00 | 2.40 |
| kartplex | complete | 1131 | 1.00 | 2.40 |
| endurance | complete | 2150 | 1.00 | 3.68 |

## PPO Policy Matrix

Model: `results/ppo_blind_grip_095_105_beefy_1m/best_model.zip`

| track | completion | off-track | mean lap time |
| --- | ---: | ---: | ---: |
| oval | 1.00 | 0.00 | 19.68 |
| s_curve | 1.00 | 0.00 | 20.00 |
| kidney | 1.00 | 0.00 | 21.28 |
| chicane | 1.00 | 0.00 | 20.24 |
| hairpin | 1.00 | 0.00 | 21.52 |
| sweepers | 1.00 | 0.00 | 24.56 |
| stop_go | 1.00 | 0.00 | 20.80 |
| club | 1.00 | 0.00 | 22.32 |
| mini_monaco | 1.00 | 0.00 | 20.80 |
| technical | 0.00 | 1.00 | - |
| grand_prix | 1.00 | 0.00 | 31.04 |
| street_circuit | 1.00 | 0.00 | 27.36 |
| kartplex | 0.00 | 1.00 | - |
| endurance | 1.00 | 0.00 | 43.92 |

## Physics Benchmarks

These are telemetry baselines, not pass/fail thresholds yet.

- acceleration: final `23.32 m/s`, avg `2.43 m/s^2`, off-track `True`
- braking: stopped `True`, distance `7.69 m`, avg decel `4.84 m/s^2`
- steady turning: yaw rate `0.016 rad/s`, radius `717.9 m`, off-track `True`
- repeatability: deterministic `True`, max obs delta `0.0`

## Failure Cases

- `technical` via racing_line controller: off (reached lap 0.96)
- `technical` via heuristic controller: off (reached lap 0.46)
- `technical` via ppo policy: incomplete (completion 0.00, off-track 1.00)
- `kartplex` via ppo policy: incomplete (completion 0.00, off-track 1.00)

## Threshold Decision

Physics benchmarks stay telemetry-only for now. The only hard gates are tests, lint, and catalog geometry validation; controller and policy results are baseline numbers used to catch regressions during review.
