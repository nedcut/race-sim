# Track authoring

Tracks are pure geometry YAML files under `configs/tracks/`. Env configs point at a track file plus a matching MuJoCo visual world. Use this workflow when adding or editing layouts.

## 1. Author centerline YAML

Two geometry kinds are supported.

### Polyline (free centerline)

```yaml
name: my_track
kind: polyline
width: 9.0
centerline:
  - [34.0, 0.0]
  - [28.0, -16.0]
  # ... closed loop points (last reconnects toward first)
```

Coordinates are meters in the track plane. Width is full track width (m).

### Rounded rectangle (parametrics)

```yaml
name: oval
kind: rounded_rectangle
length: 80.0
height: 36.0
radius: 14.0
width: 10.0
samples_per_corner: 32
samples_per_straight: 28
```

Save as `configs/tracks/<name>.yaml`. See existing files under `configs/tracks/` for samples (`oval`, `technical`, `hairpin`, …). Catalog notes: [configs/tracks/README.md](../configs/tracks/README.md).

## 2. Validate geometry

From the repo root:

```bash
# entire catalog
racesim-validate-tracks

# one or more YAMLs
racesim-validate-tracks --track configs/tracks/my_track.yaml
```

The validator builds a `ClosedTrack` and reports geometry issues (self-intersections, zero-length segments, etc.). Fix until the track prints `ok`.

Optional top-down plot:

```bash
racesim-plot-track --config configs/tracks/my_track.yaml --output results/my_track.png
```

## 3. Generate MuJoCo visuals

```bash
racesim-make-track-visual \
  --track configs/tracks/my_track.yaml \
  --output assets/mjcf/my_track.xml \
  --model-name my_track_visuals
```

Wire the generated (or hand-edited include) model into an env config:

```yaml
track: configs/tracks/my_track.yaml
model:
  xml: assets/mjcf/my_track.xml   # or a world XML that includes the track mesh
vehicle: configs/vehicles/touring.yaml
# simulation / reward / termination — copy from configs/env.yaml and tune
```

Prefer mirroring an existing `configs/env_*.yaml` so reward and termination stay consistent.

## 4. Catalog entry

Add the track to `configs/track_catalog.yaml`:

```yaml
tracks:
  my_track:
    env: configs/env_my_track.yaml
    track: configs/tracks/my_track.yaml
```

Document the intent in `configs/tracks/README.md`. Re-run:

```bash
racesim-validate-tracks
racesim-smoke-tracks --controller racing_line --steps 800
```

## Checklist

1. Centerline YAML under `configs/tracks/`
2. `racesim-validate-tracks --track ...` passes
3. `racesim-make-track-visual` (or equivalent world XML)
4. Env YAML with `track` + `model.xml`
5. Catalog + short README note
6. Smoke eval on the new env config
