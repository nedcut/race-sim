"""Disjoint seed ranges for training vs live eval vs final evaluation."""

from __future__ import annotations

# Training rollouts / PPO seed.
DEFAULT_TRAIN_SEED = 1000
# Live / validation eval during training (LiveEvalCallback, train YAML eval.seed_base).
DEFAULT_LIVE_EVAL_SEED_BASE = 10_000
# Held-out final evaluation (`racesim evaluate-policy --seed-base`).
DEFAULT_FINAL_EVAL_SEED_BASE = 20_000
