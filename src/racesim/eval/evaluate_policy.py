from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from racesim.env.racing_env import RacingEnv
from racesim.eval.evaluate import telemetry_row
from racesim.eval.metrics import EpisodeMetrics, summarize_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lap-target", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--record-trajectory", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/eval_policy.json"))
    return parser.parse_args()


def main() -> None:
    from stable_baselines3 import PPO

    args = parse_args()
    model = PPO.load(args.model)
    result = evaluate_policy_model(
        model=model,
        config_path=args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        lap_target=args.lap_target,
        deterministic=args.deterministic,
        record_trajectory=args.record_trajectory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


def evaluate_policy_model(
    model: object,
    config_path: Path,
    episodes: int,
    max_steps: int,
    lap_target: float,
    deterministic: bool,
    record_trajectory: bool,
) -> dict:
    env = RacingEnv(config_path)
    env.max_episode_steps = max_steps
    env.lap_target = lap_target
    results = []
    for episode in range(episodes):
        observation, info = env.reset(seed=episode)
        total_reward = 0.0
        trajectory = []
        steps = 0
        for steps in range(1, max_steps + 1):
            action, _state = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if record_trajectory:
                sim_time = steps * env.frame_skip * env.model.opt.timestep
                trajectory.append(
                    telemetry_row(
                        steps,
                        float(sim_time),
                        np.asarray(action),
                        reward,
                        terminated,
                        truncated,
                        info,
                    )
                )
            if terminated or truncated:
                break

        metrics = EpisodeMetrics(
            episode=episode,
            seed=episode,
            steps=steps,
            sim_time=float(steps * env.frame_skip * env.model.opt.timestep),
            total_reward=float(total_reward),
            lap_complete=bool(info["lap_complete"]),
            off_track=bool(info["off_track"]),
            lap_fraction=float(info["lap_fraction"]),
            cumulative_lap_fraction=float(info["cumulative_lap_fraction"]),
        )
        results.append({"metrics": metrics, "trajectory": trajectory})

    return {
        "controller": "ppo",
        "config": str(config_path),
        "summary": summarize_episodes([result["metrics"] for result in results]),
        "episodes": [
            {
                "metrics": asdict(result["metrics"]),
                "trajectory": result["trajectory"],
            }
            if record_trajectory
            else asdict(result["metrics"])
            for result in results
        ],
    }


if __name__ == "__main__":
    main()
