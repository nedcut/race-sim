from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from racesim.controllers.heuristic import HeuristicController
from racesim.env.racing_env import RacingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short open-loop environment rollout.")
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--controller",
        choices=("heuristic", "open_loop"),
        default="heuristic",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = RacingEnv(args.config)
    observation, info = env.reset(seed=args.seed)
    controller = HeuristicController(env.track)

    total_reward = 0.0
    terminated = False
    truncated = False
    last_info = info

    steps_taken = 0
    for _step in range(args.steps):
        if args.controller == "heuristic":
            action = controller.act(observation, last_info)
        else:
            action = np.array([0.05, 0.35, 0.0], dtype=np.float32)

        observation, reward, terminated, truncated, last_info = env.step(action)
        total_reward += reward
        steps_taken += 1
        if terminated or truncated:
            break

    print(
        "Rollout finished: "
        f"steps={steps_taken}, reward={total_reward:.2f}, "
        f"lap_fraction={last_info['lap_fraction']:.3f}, "
        f"cumulative_lap_fraction={last_info['cumulative_lap_fraction']:.3f}, "
        f"lap_complete={last_info['lap_complete']}, "
        f"off_track={last_info['off_track']}, "
        f"terminated={terminated}, truncated={truncated}"
    )


if __name__ == "__main__":
    main()
