# RaceSim Baselines

Generated: `2026-05-21T15:29:46+00:00`
Catalog: `configs/track_catalog.yaml`

## Quality Gates

- pytest: pass
- ruff: pass
- track validation: pass

## Default Oval Eval

| controller | completion | off-track | mean lap time | best lap time |
| --- | ---: | ---: | ---: | ---: |
| racing_line | 1.00 | 0.00 | 68.08 | 68.08 |
| heuristic | 1.00 | 0.00 | 69.44 | 69.44 |

## Catalog Smoke (5000 Steps)

### Racing Line

| track | status | steps | lap fraction | speed |
| --- | --- | ---: | ---: | ---: |
| oval | complete | 851 | 1.00 | 2.41 |
| s_curve | complete | 1032 | 1.00 | 2.42 |
| kidney | complete | 1177 | 1.00 | 2.40 |
| chicane | complete | 1121 | 1.00 | 2.42 |
| hairpin | complete | 1058 | 1.00 | 2.44 |
| sweepers | complete | 1159 | 1.00 | 2.58 |
| stop_go | complete | 967 | 1.00 | 2.41 |
| club | complete | 1211 | 1.00 | 2.40 |
| mini_monaco | complete | 1063 | 1.00 | 2.44 |
| technical | complete | 997 | 1.00 | 2.41 |
| grand_prix | complete | 1698 | 1.00 | 3.01 |
| street_circuit | complete | 1465 | 1.00 | 2.40 |
| kartplex | complete | 1080 | 1.00 | 2.40 |
| endurance | complete | 1974 | 1.00 | 4.15 |

### Heuristic

| track | status | steps | lap fraction | speed |
| --- | --- | ---: | ---: | ---: |
| oval | complete | 868 | 1.00 | 2.41 |
| s_curve | complete | 1071 | 1.00 | 2.42 |
| kidney | complete | 1196 | 1.00 | 2.40 |
| chicane | complete | 1137 | 1.00 | 2.43 |
| hairpin | complete | 1087 | 1.00 | 2.42 |
| sweepers | complete | 1206 | 1.00 | 2.41 |
| stop_go | complete | 990 | 1.00 | 2.40 |
| club | complete | 1240 | 1.00 | 2.41 |
| mini_monaco | complete | 1096 | 1.00 | 2.41 |
| technical | complete | 1034 | 1.00 | 2.41 |
| grand_prix | complete | 1730 | 1.00 | 2.66 |
| street_circuit | complete | 1492 | 1.00 | 2.40 |
| kartplex | complete | 1101 | 1.00 | 2.40 |
| endurance | complete | 2107 | 1.00 | 3.67 |

## PPO Policy Matrix

Model: `results/ppo_blind_grip_095_105_beefy_1m/best_model.zip`

| track | completion | off-track | mean lap time |
| --- | ---: | ---: | ---: |
| oval | 1.00 | 0.00 | 19.52 |
| s_curve | 1.00 | 0.00 | 23.36 |
| kidney | 1.00 | 0.00 | 24.96 |
| chicane | 1.00 | 0.00 | 23.60 |
| hairpin | 1.00 | 0.00 | 24.80 |
| sweepers | 1.00 | 0.00 | 27.92 |
| stop_go | 1.00 | 0.00 | 24.00 |
| club | 1.00 | 0.00 | 26.16 |
| mini_monaco | 1.00 | 0.00 | 24.24 |
| technical | 1.00 | 0.00 | 19.04 |
| grand_prix | 1.00 | 0.00 | 35.44 |
| street_circuit | 1.00 | 0.00 | 30.24 |
| kartplex | 1.00 | 0.00 | 23.12 |
| endurance | 1.00 | 0.00 | 48.72 |

## Physics Benchmarks

These are telemetry baselines, not pass/fail thresholds yet.

| benchmark | steps | final speed | yaw rate | tire usage | notes |
| --- | ---: | ---: | ---: | ---: | --- |
| acceleration | 120 | 22.47 | - | - | off-track |
| braking | 23 | 0.14 | - | - | stopped=True |
| steady_turning | 160 | - | 0.94 | - | off-track |
| skidpad | 180 | 13.83 | 1.28 | 0.96 | - |
| step_steer | 180 | 12.70 | 1.46 | 0.97 | off-track |
| slalom | 180 | 12.35 | 0.57 | 0.91 | off-track |
| braking_turn | 180 | 0.08 | 0.20 | 0.88 | off-track |
| throttle_exit | 180 | 21.31 | 0.73 | 0.88 | off-track |
| repeatability | 80 | - | - | - | deterministic=True |

### Sanity Flags

- none

## Failure Cases

- none in this suite run

## Threshold Decision

Physics benchmarks stay telemetry-only for now. The only hard gates are tests, lint, and catalog geometry validation; controller and policy results are baseline numbers used to catch regressions during review.
