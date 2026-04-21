from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from racesim.controllers.heuristic import HeuristicController
from racesim.env.racing_env import RacingEnv
from racesim.eval.metrics import EpisodeMetrics, summarize_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a controller on the racing environment.")
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--controller", choices=("heuristic", "open_loop"), default="heuristic")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/eval_heuristic.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        config_path=args.config,
        controller_name=args.controller,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


def evaluate(
    config_path: str | Path,
    controller_name: str,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict:
    env = RacingEnv(config_path)
    controller = HeuristicController(env.track)
    episode_metrics = []

    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        metrics = run_episode(
            env=env,
            controller=controller,
            controller_name=controller_name,
            episode=episode_index,
            seed=episode_seed,
            max_steps=max_steps,
        )
        episode_metrics.append(metrics)

    summary = summarize_episodes(episode_metrics)
    return {
        "controller": controller_name,
        "config": str(config_path),
        "summary": summary,
        "episodes": [asdict(episode) for episode in episode_metrics],
    }


def run_episode(
    env: RacingEnv,
    controller: HeuristicController,
    controller_name: str,
    episode: int,
    seed: int,
    max_steps: int,
) -> EpisodeMetrics:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    for _step in range(max_steps):
        if controller_name == "heuristic":
            action = controller.act(observation, info)
        elif controller_name == "open_loop":
            action = np.array([0.05, 0.35, 0.0], dtype=np.float32)
        else:
            raise ValueError(f"Unsupported controller: {controller_name}")

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    sim_time = steps * env.frame_skip * env.model.opt.timestep
    return EpisodeMetrics(
        episode=episode,
        seed=seed,
        steps=steps,
        sim_time=float(sim_time),
        total_reward=float(total_reward),
        lap_complete=bool(info["lap_complete"]),
        off_track=bool(info["off_track"]),
        lap_fraction=float(info["lap_fraction"]),
        cumulative_lap_fraction=float(info["cumulative_lap_fraction"]),
    )


if __name__ == "__main__":
    main()
