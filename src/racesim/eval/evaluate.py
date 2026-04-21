from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from racesim.controllers.factory import CONTROLLERS, make_controller
from racesim.env.racing_env import RacingEnv
from racesim.eval.metrics import EpisodeMetrics, summarize_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a controller on the racing environment.")
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--controller", choices=CONTROLLERS, default="centerline")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/eval_heuristic.json"))
    parser.add_argument(
        "--record-trajectory",
        action="store_true",
        help="Include per-step trajectory and control telemetry in the output JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = evaluate(
        config_path=args.config,
        controller_name=args.controller,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        record_trajectory=args.record_trajectory,
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
    record_trajectory: bool = False,
) -> dict:
    env = RacingEnv(config_path)
    controller = make_controller(controller_name, env.track)
    episode_results = []

    for episode_index in range(episodes):
        episode_seed = seed + episode_index
        result = run_episode(
            env=env,
            controller=controller,
            controller_name=controller_name,
            episode=episode_index,
            seed=episode_seed,
            max_steps=max_steps,
            record_trajectory=record_trajectory,
        )
        episode_results.append(result)

    summary = summarize_episodes([result["metrics"] for result in episode_results])
    return {
        "controller": controller_name,
        "config": str(config_path),
        "summary": summary,
        "episodes": [
            {
                "metrics": asdict(result["metrics"]),
                "trajectory": result["trajectory"],
            }
            if record_trajectory
            else asdict(result["metrics"])
            for result in episode_results
        ],
    }


def run_episode(
    env: RacingEnv,
    controller: object,
    controller_name: str,
    episode: int,
    seed: int,
    max_steps: int,
    record_trajectory: bool = False,
) -> dict:
    observation, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    trajectory = []

    for _step in range(max_steps):
        action = controller.act(observation, info)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if record_trajectory:
            sim_time = steps * env.frame_skip * env.model.opt.timestep
            trajectory.append(
                telemetry_row(
                    steps,
                    float(sim_time),
                    action,
                    reward,
                    terminated,
                    truncated,
                    info,
                )
            )
        if terminated or truncated:
            break

    sim_time = steps * env.frame_skip * env.model.opt.timestep
    return {
        "metrics": EpisodeMetrics(
            episode=episode,
            seed=seed,
            steps=steps,
            sim_time=float(sim_time),
            total_reward=float(total_reward),
            lap_complete=bool(info["lap_complete"]),
            off_track=bool(info["off_track"]),
            lap_fraction=float(info["lap_fraction"]),
            cumulative_lap_fraction=float(info["cumulative_lap_fraction"]),
        ),
        "trajectory": trajectory,
    }


def telemetry_row(
    step: int,
    sim_time: float,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
) -> dict:
    smoothed_action = np.asarray(info["smoothed_action"], dtype=float)
    return {
        "step": step,
        "time": sim_time,
        "x": float(info["position"][0]),
        "y": float(info["position"][1]),
        "heading": float(info["heading"]),
        "progress": float(info["progress"]),
        "lap_fraction": float(info["lap_fraction"]),
        "cumulative_lap_fraction": float(info["cumulative_lap_fraction"]),
        "speed": float(info["speed"]),
        "longitudinal_speed": float(info["longitudinal_speed"]),
        "lateral_speed": float(info["lateral_speed"]),
        "yaw_rate": float(info["yaw_rate"]),
        "lateral_error": float(info["lateral_error"]),
        "heading_error": float(info["heading_error"] or 0.0),
        "steering": float(action[0]),
        "throttle": float(action[1]),
        "brake": float(action[2]),
        "smoothed_steering": float(smoothed_action[0]),
        "smoothed_throttle": float(smoothed_action[1]),
        "smoothed_brake": float(smoothed_action[2]),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "off_track": bool(info["off_track"]),
        "lap_complete": bool(info["lap_complete"]),
    }


if __name__ == "__main__":
    main()
