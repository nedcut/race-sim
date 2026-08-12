# Dynamics Roadmap

RaceSim uses a simplified bicycle force proxy. This document keeps the scope honest and
maps a path for deeper vehicle models when they earn their place in the tool.

It covers *what* deeper dynamics would take, not *when* to build them. Sequencing is
governed by [roadmap-to-1.0.md](roadmap-to-1.0.md), which defers wheel-contact tire work
until racing is fun — so treat everything under **Later** as post-1.0 unless that roadmap
says otherwise.

## Now (shipping proxy)

- Planar free-body chassis driven by `xfrc_applied` body forces
- Front/rear lateral tire curve with tanh saturation + combined-force ellipse
- Longitudinal drive/brake force caps with soft longitudinal saturation
- Longitudinal load transfer and optional aero **downforce** (normal load only)
- Optional aero **drag** coefficient (quadratic, added to residual linear drag)
- Optional chassis mass/inertia override from vehicle YAML
- Track geometry layer for progress, lateral/heading error, termination

## Trust boundary

Worth trusting: track geometry, environment plumbing, relative controller comparisons,
deterministic open-loop telemetry regression.

Not worth claiming: Pacejka tire data, suspension, wheel contact friction, F1 lap times,
transfer of absolute cornering limits to the real world.

See also: [`simulation.md`](simulation.md).

## Near-term hardening

1. Flat-pad open-loop golden ranges (`racesim-physics-benchmarks --enforce`)
2. Chassis parameters as first-class vehicle assets (mass already YAML-driven)
3. Richer evaluation metrics (lateral-error aggregates) for control quality

## Later (explicit non-goals of the proxy era)

| Item | Why deferred |
|------|----------------|
| Magic Formula / Pacejka | Needs contact kinematics, slip ratio, and fitted coefficients |
| Four-wheel contact MuJoCo tires | Larger MJCF rewrite; regress all forces |
| Suspension / roll | Requires vertical DOFs and track 3D elevation |
| Engine map + gears | Forces become wheel-torque limited, not direct body force |
| Banking / curbs / multi-surface μ | Track representation expansion |

## Guidance

Prefer improving reward, observations, curricula, and evaluation gates until the bicycle
proxy is well characterized. Then introduce wheel-contact tires behind regression
thresholds so later dynamics work cannot silently break training baselines.
