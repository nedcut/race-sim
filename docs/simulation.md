# Current Simulation Model

This project currently uses a simplified MuJoCo vehicle model. The car is a free chassis body with visual wheel geometry. The environment applies a bicycle-style tire-force proxy:

- front steering angle from the steering action
- front/rear lateral tire-force proxies from slip angles
- configurable drivetrain split: `rwd`, `fwd`, or `awd`
- throttle and brake longitudinal forces
- drag, rolling resistance, yaw damping, and action smoothing

That means the current car is useful for testing the environment loop, reward, progress metric, heuristic baseline, and evaluation tooling. It should feel more car-like than raw yaw torque, but it is not yet a full wheel/contact tire model.

The default MuJoCo world now includes an oval track surface, red boundaries, and centerline markers so the native viewer matches the Python-side track geometry used for reward and evaluation.

## What Is Worth Trusting Now

- Track-relative progress, lateral error, heading error, and lap completion.
- Whether a controller can follow the track without leaving it.
- Coarse behavior under different force, damping, drivetrain, and grip-like parameters.
- Baseline-vs-learned evaluation plumbing once RL is added.

## What Is Not Worth Claiming Yet

- Realistic tire slip.
- Suspension behavior.
- Physical wheel torque transfer through tire contacts.
- F1-like dynamics.
- A physically meaningful lap time.

## How To Inspect It

Run a deterministic baseline evaluation:

```bash
racesim-evaluate --controller heuristic --episodes 5
```

Render a top-down rollout:

```bash
racesim-render-rollout --controller heuristic --steps 1800
```

Try manual driving in MuJoCo's native viewer:

```bash
racesim-keyboard-drive
```

On macOS, MuJoCo's passive viewer must run under `mjpython`. The `racesim-keyboard-drive` command automatically relaunches itself with `mjpython` when it can find it. If you want to run it manually:

```bash
mjpython -m racesim.scripts.keyboard_drive
```

Keyboard controls are printed when the script starts. The driving keys avoid WASD because MuJoCo's viewer uses those for built-in shortcuts.

- `I` / `K`: throttle up/down
- `J` / `U`: brake up/down
- `F` / `G`: steer left/right
- `T`: center steering
- `Space`: zero steering/throttle/brake
- `H`: toggle heuristic autopilot
- `1`: chase camera
- `2`: top-down camera
- `3`: reset free camera
- `4`: fixed MuJoCo follow camera

If the viewer looks wrong, run the smoke check first:

```bash
racesim-smoke-mujoco
```

The compiled model should include hundreds of geoms because the visual track is part of the MuJoCo scene.
