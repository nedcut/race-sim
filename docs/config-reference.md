# Environment Config Reference

Env YAML files (under `configs/env*.yaml`) fully describe a race episode: track, MuJoCo model, vehicle, reward, termination, and optional reset randomization. Paths in the file are relative to the **repository root** when the config lives under `configs/`.

## Top-level structure

```yaml
track: configs/tracks/oval.yaml          # geometry YAML (required)
simulation: { ... }                      # timing and episode limits
model:
  xml: assets/mjcf/world.xml             # MuJoCo scene (required)
vehicle: configs/vehicles/touring.yaml   # optional; merged into control/tire_model
control: { ... }                         # overrides vehicle.control
tire_model: { ... }                      # overrides vehicle.tire_model
observation: { ... }                     # optional; defaults to 4 lookaheads
termination: { ... }
reset_randomization: { ... }
reward: { ... }
```

Example baseline: `configs/env.yaml`.

## `simulation`

| Key | Default | Meaning |
|-----|---------|---------|
| `dt` | (XML timestep) | When provided, sets `model.opt.timestep`; otherwise the XML timestep is kept |
| `frame_skip` | `1` | Physics substeps per env `step` |
| `max_episode_steps` | `3000` | Truncate after this many env steps |
| `initial_speed` | `0.0` | Reset speed along track tangent (m/s) |
| `lap_target` | `1.0` | Terminates when net signed forward progress ≥ this many laps |

## `vehicle` / `control` / `tire_model`

`vehicle` points at a vehicle YAML (e.g. `configs/vehicles/touring.yaml`) with optional `control` and `tire_model` sections. Env-level `control` and `tire_model` override the vehicle file.

**Control (bicycle proxy)** — common keys:

| Key | Meaning |
|-----|---------|
| `drivetrain` | `rwd` / `fwd` / `awd` |
| `max_steer_angle` | Max steer angle (rad) at action ±1 |
| `steer_rate` / `throttle_rate` / `brake_rate` | Action smoothing (units/s) |
| `max_drive_force` / `max_brake_force` | Longitudinal force scales (N) |
| `front_cornering_stiffness` / `rear_cornering_stiffness` | Lateral tire scales |
| `max_lateral_force` | Used when friction is not set explicitly |
| `yaw_damping` | Yaw damping torque coefficient |
| `linear_drag` / `rolling_resistance` | Resistive longitudinal terms |
| `wheelbase`, `center_of_mass_to_front`, `center_of_mass_to_rear` | Geometry (m) |

**Tire model** — e.g. `center_of_mass_height`, `tire_friction_coefficient`, `combined_slip_exponent`, `brake_front_bias`, aero downforce keys, slip-angle saturation parameters. See [simulation.md](simulation.md) for the physical interpretation.

## `observation`

| Key | Default | Meaning |
|-----|---------|---------|
| `lookahead_distances` | `(6, 12, 24, 40)` | Arc distances (m) for heading/curvature features |
| `curvature_scale` | `20.0` | Multiplier on signed curvature features |

Observation shape is `11 + 2 * len(lookahead_distances)`. See [observation-and-action.md](observation-and-action.md).

## `reward`

All terms are **weights** unless noted. Per-step reward is the sum of:

| Key | Effect |
|-----|--------|
| `progress` | `× progress_delta` (m) |
| `time` | constant negative cost per step |
| `lateral_error` | `× |lateral_error|` |
| `heading_error` | `× |heading_error|` |
| `boundary_margin` | quadratic shortfall below margin start |
| `boundary_margin_start` | margin distance (m) before shortfall grows |
| `progress_gate` | bonus per ordered forward gate crossed |
| `progress_gate_spacing` | `(0, 1]` is a lap fraction, `> 1` is meters, `≤ 0` disables gates. A gate fires only when moving forward onto the next gate in order |
| `speed_excess` | `× max(speed − target, 0)²` |
| `target_speed_max` / `target_speed_min` | target speed clamp (m/s) |
| `target_speed_curvature_gain` | map curvature → lower target speed |
| `no_progress` | penalty when stuck-timeout triggers |
| `off_track` | penalty on off-track termination |

## `termination`

| Key | Default | Meaning |
|-----|---------|---------|
| `off_track_margin` | `0.0` | Extra margin beyond track half-width for off-track |
| `no_progress_window_steps` | `0` | Window length; `0` disables no-progress truncate |
| `no_progress_min_delta` | `0.0` | Minimum cumulative meters required in the window |
| `max_progress_delta_factor` | `2.0` | Cap progress vs physical motion |
| `max_progress_delta_slack` | `0.5` | Additive slack (m) on the progress cap |

Episodes **terminate** on off-track or lap complete; they **truncate** on max steps or no-progress timeout.

## `reset_randomization`

| Key | Meaning |
|-----|---------|
| `enabled` | Default randomize when `reset` options omit `randomize` |
| `progress` | Random start along full track length |
| `lateral_offset` | Max lateral offset (m) |
| `heading_error` | Max heading offset (rad) |
| `speed_min` / `speed_max` | Random initial speed range (m/s) |
| `grip_min` / `grip_max` | Random grip_scale range |

`env.reset(options={...})` can force `progress`, `lateral_offset`, `heading_error`, `initial_speed`, `grip_scale`, or `randomize`.

## Gym / factory usage

```python
import racesim  # registers RaceSim-v0
import gymnasium as gym

env = gym.make("RaceSim-v0")  # configs/env.yaml
env = gym.make("RaceSim-v0", config="configs/env_s_curve.yaml")
# or: export RACESIM_CONFIG=configs/env_hairpin.yaml
```
