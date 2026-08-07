#!/usr/bin/env python3
"""Minimal custom controller episode loop (no RL dependency).

Run from the repository root::

    python examples/custom_controller.py
    python examples/custom_controller.py --episodes 2 --config configs/env.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np

import racesim  # noqa: F401 — registers RaceSim-v0


def act(observation: np.ndarray, info: dict) -> np.ndarray:
    """Simple stabilizing policy: steer toward the track, mild throttle.

    Observation indices (see docs/observation-and-action.md):
      1 lateral_error_norm, 2 heading_error_norm, 6 speed.
    """
    del info  # available for richer controllers (progress, grip_scale, ...)
    lateral = float(observation[1])
    heading = float(observation[2])
    speed = float(observation[6])

    steering = float(np.clip(-1.4 * lateral - 1.2 * heading, -1.0, 1.0))
    target_speed = 4.5
    speed_error = target_speed - speed
    throttle = float(np.clip(0.2 * speed_error, 0.0, 0.6))
    brake = float(np.clip(-0.15 * speed_error, 0.0, 0.4))
    return np.array([steering, throttle, brake], dtype=np.float32)


def run_episode(
    env: gym.Env,
    *,
    seed: int,
    max_steps: int,
) -> dict[str, float | bool | int]:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    for step in range(1, max_steps + 1):
        action = act(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps = step
        if terminated or truncated:
            break
    return {
        "seed": seed,
        "steps": steps,
        "total_reward": total_reward,
        "lap_complete": bool(info["lap_complete"]),
        "off_track": bool(info["off_track"]),
        "cumulative_lap_fraction": float(info["cumulative_lap_fraction"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = gym.make("RaceSim-v0", config=str(args.config))
    try:
        for episode in range(args.episodes):
            metrics = run_episode(env, seed=args.seed + episode, max_steps=args.max_steps)
            print(
                f"episode={episode} steps={metrics['steps']} "
                f"reward={metrics['total_reward']:.2f} "
                f"lap_frac={metrics['cumulative_lap_fraction']:.3f} "
                f"done={metrics['lap_complete']} off_track={metrics['off_track']}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
