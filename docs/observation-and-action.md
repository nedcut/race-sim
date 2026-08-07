# Observation and Action Spaces

Default observation dimension is **19** when `observation.lookahead_distances` has four entries (the package default). Shape is always:

```text
11 + 2 * len(lookahead_distances)
```

Action space is always **3-D** continuous.

## Action space

| Index | Name     | Range     | Units / meaning                          |
|------:|----------|-----------|------------------------------------------|
| 0     | steering | `[-1, 1]` | Normalized left/right (−1 left, +1 right)|
| 1     | throttle | `[0, 1]`  | Drive command                            |
| 2     | brake    | `[0, 1]`  | Brake command                            |

Requested actions are clipped to these bounds each `step`.

### Action smoothing

Commands are not applied open-loop. Each physics substep updates a smoothed command vector:

```text
smoothed += clip(action − smoothed, ±rate * dt)
```

Rates come from vehicle/control config (`steer_rate`, `throttle_rate`, `brake_rate`), in units of **command units per second**. `dt` is the MuJoCo timestep (`model.opt.timestep`). Over one env step, smoothing runs `frame_skip` times.

The commanded steer angle is `smoothed_steering * max_steer_angle` (radians). Smoothed commands are exposed each step as `info["smoothed_action"]` (shape `(3,)`).

## Observation layout (default 19-D)

| Index | Name | Units / scale | Description |
|------:|------|---------------|-------------|
| 0 | `progress_fraction` | unitless `[0, 1)` (wraps with track) | Arc progress / track length |
| 1 | `lateral_error_norm` | unitless | Lateral error / half-width (−/+) |
| 2 | `heading_error_norm` | unitless | Heading error / π |
| 3 | `longitudinal_speed` | m/s | Body-frame forward speed |
| 4 | `lateral_speed` | m/s | Body-frame lateral speed |
| 5 | `yaw_rate` | rad/s | Yaw rate |
| 6 | `speed` | m/s | Horizontal speed magnitude |
| 7 | `outside_track` | `{0, 1}` | 1 if \|lateral error\| > half-width |
| 8 | `left_margin_norm` | unitless | Distance to left bound / full width |
| 9 | `right_margin_norm` | unitless | Distance to right bound / full width |
| 10 | `target_speed_norm` | unitless | Target speed / `reward.target_speed_max` |
| 11 | `lookahead_heading_0` | unitless | Heading to future point at distance 0, / π |
| 12 | `lookahead_curvature_0` | scaled | Signed curvature proxy at distance 0 |
| 13 | `lookahead_heading_1` | unitless | Same pair for distance 1 |
| 14 | `lookahead_curvature_1` | scaled | |
| 15 | `lookahead_heading_2` | unitless | Same pair for distance 2 |
| 16 | `lookahead_curvature_2` | scaled | |
| 17 | `lookahead_heading_3` | unitless | Same pair for distance 3 |
| 18 | `lookahead_curvature_3` | scaled | |

Default `observation.lookahead_distances` are **6, 12, 24, 40** meters along the centerline. Curvature features are

```text
wrap(heading_future − heading_current) / distance * curvature_scale
```

with default `curvature_scale: 20.0`. Adding or removing lookahead distances changes the total length (two features per distance).

## `info` dictionary keys

Returned from both `reset` and `step` (reward-related fields empty at reset).

| Key | Type | Meaning |
|-----|------|---------|
| `progress` | float | Absolute centerline arc length (m) |
| `lap_fraction` | float | Current lap progress / track length |
| `cumulative_lap_fraction` | float | Forward progress / track length (multi-lap) |
| `progress_delta` | float | Validated centerline progress this step (m) |
| `raw_progress_delta` | float | Unclipped geometric progress delta (m) |
| `progress_delta_clipped` | bool | Whether progress delta was capped |
| `position` | `(2,)` float | World XY (m) |
| `heading` | float | Yaw (rad) |
| `speed` | float | Horizontal speed (m/s) |
| `longitudinal_speed` | float | Body-frame forward speed (m/s) |
| `lateral_speed` | float | Body-frame lateral speed (m/s) |
| `yaw_rate` | float | Yaw rate (rad/s) |
| `smoothed_action` | `(3,)` float | Post-smoothing control command |
| `lateral_error` | float | Signed distance from centerline (m) |
| `heading_error` | float | Heading vs tangent (rad), may be `None`-like 0 |
| `target_speed` | float | Local speed target (m/s) |
| `grip_scale` | float | Episode grip multiplier |
| `tire_usage` | `{front, rear}` | Combined tire usage [0, 1] |
| `normal_loads` | `{front, rear}` | Axle normal loads (N) |
| `slip_angles` | `{front, rear}` | Slip angles (rad) |
| `tire_forces` | dict | Axle longitudinal/lateral/raw lateral forces |
| `yaw_torque` | float | Applied yaw torque |
| `steering_angle` | float | Actual steer angle (rad) after max_steer scale |
| `understeer_score` | float | Expected yaw rate − actual yaw rate |
| `off_track` | bool | Off-track with termination margin |
| `lap_complete` | bool | Cumulative progress ≥ `lap_target` laps |
| `no_progress_timeout` | bool | Stuck within progress window |
| `reward_terms` | dict | Per-term reward contributions (empty on reset) |

On curriculum / multi-track training wrappers, additional keys such as `track_name` may be present.
