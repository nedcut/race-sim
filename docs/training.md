# Training guide for RaceSim PPO and warm-start pipelines.

This path is **experimental**. There is no completed high-quality trained baseline yet. Packaging remains **0.2.0**; treat quality configs as a research stack, not a release artifact.

## Recommended path

1. **Sanity-check the simulator**
   ```bash
   racesim doctor
   racesim evaluate -- --controller racing_line --episodes 3
   ```
2. **(Optional) Collect expert data and BC warm-start**
   ```bash
   racesim collect-expert -- --config configs/env_rl_quality.yaml \
     --controller racing_line --episodes 30 --randomize-reset \
     --output results/expert/racing_line_quality.npz
   racesim bc-pretrain -- --data results/expert/racing_line_quality.npz \
     --train-config configs/train_ppo_quality.yaml \
     --epochs 40 --output results/bc/bc_quality
   ```
   BC fits `VecNormalize` observation stats from the expert arrays (when
   `normalize.enabled` is true), trains on **normalized** observations, writes
   `vecnormalize.pkl` next to the BC zip, and sets action `log_std` low
   (`-3.0`, near-deterministic) so the first PPO rollouts resemble the expert.
3. **Train PPO** (VecNormalize + linear LR + live eval)
   ```bash
   racesim train -- --config configs/train_ppo_quality.yaml
   # BC warm-start (policy weights only):
   #   pretrained: results/bc/bc_quality
   # Full checkpoint restore (optimizer, schedules, timesteps, VecNormalize):
   #   resume: results/ppo_quality/agent_bundle
   ```
   `--dry-run` validates config and builds envs; it does **not** write
   `train_config.json` or other artifacts.
4. **Evaluate the agent bundle**
   ```bash
   racesim evaluate-policy -- \
     --bundle results/ppo_quality/agent_bundle \
     --config configs/env_rl_quality.yaml \
     --episodes 10 --deterministic --seed-base 20000
   ```

The default legacy oval config (`configs/train_ppo_oval.yaml` + 19-D `configs/env.yaml`) still works for A/B comparisons. **Quality** configs use a 25-D observation (smoothed actuator command, grip, tire usage) and shaped rewards (`lap_complete`, `tire_usage`, `action_rate`).

## Agent bundles

Training writes `output_dir/agent_bundle/` containing:

| File | Role |
|------|------|
| `model.zip` | SB3 PPO checkpoint |
| `vecnormalize.pkl` | Running obs/reward stats (**required** when `normalize.enabled`) |
| `train_config.json` | Resolved train YAML |
| `observation_action_schema.json` | Observation/action dims and flags |
| `manifest.json` | Schema version, SHA-256s, env config hashes, git SHA, dependency versions, timesteps/seed |

Evaluation, the suite policy matrix, and live-eval “best” snapshots must load that bundle (or a zip **plus** its required normalizer). A missing or incompatible `VecNormalize` file is an error — never a silent raw-observation eval.

`--model path.zip` remains for legacy checkpoints. It fail-closes when a sibling `vecnormalize.pkl` / `*_vecnormalize.pkl` is required (nearby `train_config.json` with `normalize.enabled`, or stats sitting next to the zip). Prefer `--bundle`.

**Keyboard drive does not load agent bundles.** If you pass `--policy-model` for a checkpoint that has VecNormalize stats, keyboard exits instead of evaluating on raw observations. Use `racesim evaluate-policy -- --bundle PATH`.

The eval suite accepts `--policy-bundle` and will also treat a `--policy-model` directory with `manifest.json` as a bundle.

### Trusted checkpoints only

SB3 zips and `VecNormalize` pickles are **Cloudpickle / pickle** artifacts. Loading them can execute code. Only load checkpoints you trained or otherwise trust. Do not unpickle bundles from untrusted sources.

## `pretrained` vs `resume`

These keys are distinct; setting both is an error.

| Key | Restores | VecNormalize | `learn(reset_num_timesteps=...)` |
|-----|----------|--------------|----------------------------------|
| `pretrained` | Policy weights only into a **fresh** PPO (BC warm-start) | Load BC stats when `normalize.enabled`; do not start from empty RMS | `True` (new run) |
| `resume` | Full PPO zip (optimizer, schedules, buffers, timestep counters) | Must load saved stats when `normalize.enabled` | `False` |

`pretrained` does **not** pretend optimizer state, learning-rate schedules, or timestep counters were restored.

## Train config keys

| Key | Meaning |
|-----|---------|
| `seed` | Training seed (quality default `1000`) |
| `total_timesteps` | PPO env steps |
| `output_dir` | Checkpoints, TensorBoard, `agent_bundle/` |
| `normalize.enabled` | Wrap with SB3 `VecNormalize` (obs/reward) |
| `checkpoint_freq` | Timesteps between `checkpoints/ppo_*.zip` |
| `vecnormalize_save_freq` | Timesteps between VecNormalize snapshots |
| `pretrained` | BC / policy-weight warm-start path (zip or bundle) |
| `resume` | Full checkpoint path (zip or bundle) |
| `env.env_configs` | List of RacingEnv YAMLs (curriculum sample) |
| `env.n_envs` / `env.vec_env` | Parallel envs (`dummy` or `subproc`) |
| `ppo.learning_rate` | Float or `linear_3e-4` schedule |
| `ppo.policy_kwargs.net_arch` | Actor/critic MLP sizes |
| `eval.frequency` | Live eval period in **timesteps** (first eval at this step, not step 1) |
| `eval.seed_base` | Live/validation eval seeds (quality default `10000`) |
| `curriculum.stages` | Time-based probability shifts |

Parsed schedule helpers live in `racesim.training.schedules`.

CheckpointCallback still counts `n_calls`, so `checkpoint_freq` is converted with `freq // n_envs`. `SaveVecNormalizeCallback` and live eval use `num_timesteps`. YAML values are all in **timesteps**; with `n_envs=4` they must not silently disagree by 4×.

## Seeds

Keep these ranges disjoint so training noise is not reused at eval time:

| Role | Range | Where |
|------|-------|--------|
| Training | 1000–1999 | `seed` in train YAML |
| Live / validation | 10000–19999 | `eval.seed_base` |
| Final evaluation | 20000–29999 | `racesim evaluate-policy --seed-base` (CLI default `20000`) |

## Observation choice for RL

Default **19-D** (no extra flags) matches older checkpoints.

**Quality env** (`configs/env_rl_quality.yaml`) appends:

| Tail features | Dims | Flag |
|---------------|-----:|------|
| smoothed actuator command | 3 | `include_prev_action` |
| grip scale | 1 | `include_grip_scale` |
| front/rear tire usage | 2 | `include_tire_usage` |

`include_prev_action` is a historical flag name. The tail is the **rate-limited command that affected physics** (`info["smoothed_action"]`), not the raw requested action.

Formula: `11 + 2 * n_lookahead + optional tails` (see `docs/observation-and-action.md`).

Keep curriculum env configs on the **same observation shape**. Env YAML is fail-closed: unknown keys (for example `lap_complet`) raise `ValueError`.

## Reward shaping for RL

New optional weights (default `0` on classic envs):

| Key | Effect |
|-----|--------|
| `lap_complete` | Sparse bonus when a lap completes |
| `tire_usage` | Penalty × mean axle tire usage |
| `action_rate` | Penalty × ‖Δ smoothed action‖ this step |

Classic dense terms (`progress`, gates, path errors, etc.) remain.

## VecNormalize notes

When `normalize.enabled` is true:

- Training observations/rewards are normalized online.
- BC, when used, must fit/apply the **same** observation normalization and save stats next to the BC zip.
- PPO `pretrained` loads those stats instead of creating empty running mean/std.
- The agent bundle includes `vecnormalize.pkl`. Evaluation **must** load it.

## Curriculum & multi-track

Use multiple `env.env_configs` plus optional `curriculum.stages` with probability vectors. `TrackCurriculumEnv` rejects mismatched observation shapes.

## Smoke test

```bash
racesim train -- --config configs/train_ppo_quality_smoke.yaml
```

Short run (~2k steps) to validate the pipeline; not a competitive policy.

## Dependencies

```bash
uv pip install -e ".[rl]"
# or: pip install -e ".[rl]"
```
