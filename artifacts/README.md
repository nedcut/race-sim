# Artifacts

Small pretrained checkpoints for onboarding demos live here.

## `ppo_oval_quick.zip`

Tiny PPO policy trained on the default oval env for a few thousand timesteps.
It is only a smoke / demo weight, not a performance baseline.

**Generate or refresh:**

```bash
# From repo root, with RL extras installed:
pip install -e ".[rl]"
python scripts/train_quick_checkpoint.py

# Or:
racesim-train-ppo --config configs/train_ppo_oval_quick.yaml
cp results/ppo_oval_quick/final_model.zip artifacts/ppo_oval_quick.zip
```

**Evaluate when present:**

```bash
racesim-evaluate-policy --model artifacts/ppo_oval_quick.zip --config configs/env.yaml --episodes 2 --deterministic
```
