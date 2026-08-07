#!/usr/bin/env python3
"""Minimal Stable-Baselines3 PPO training (CPU / auto device).

Requires the optional ``rl`` extra::

    pip install -e ".[rl]"

Run from the repository root::

    python examples/sb3_train_minimal.py
    python examples/sb3_train_minimal.py --timesteps 10000 --device auto

No custom train YAML is required; defaults keep the run small (≤50k steps).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym

import racesim  # noqa: F401 — registers RaceSim-v0
from racesim.utils.device import resolve_torch_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--timesteps", type=int, default=50_000)
    parser.add_argument(
        "--device",
        default="auto",
        help='Torch/SB3 device: "auto", "cpu", "cuda", or "mps".',
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/ppo_minimal/model"),
    )
    return parser.parse_args()


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "stable-baselines3 is required. Install with: pip install -e '.[rl]'"
        ) from exc

    args = parse_args()
    device = resolve_torch_device(args.device)
    env = Monitor(gym.make("RaceSim-v0", config=str(args.config)))

    model = PPO(
        "MlpPolicy",
        env,
        seed=args.seed,
        device=device,
        verbose=1,
        n_steps=512,
        batch_size=64,
        learning_rate=3e-4,
    )
    print(f"Training on device={device} for {args.timesteps} timesteps")
    model.learn(total_timesteps=args.timesteps)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.output)
    print(f"Saved {args.output}.zip")
    env.close()


if __name__ == "__main__":
    main()
