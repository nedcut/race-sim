from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from racesim.controllers.heuristic import HeuristicController
from racesim.env.racing_env import RacingEnv
from racesim.eval.evaluate import run_episode
from racesim.eval.metrics import EpisodeMetrics, summarize_episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep vehicle drivetrain and grip parameters.")
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv-output", type=Path, default=Path("results/vehicle_sweep.csv"))
    parser.add_argument("--json-output", type=Path, default=Path("results/vehicle_sweep.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_sweep(
        config_path=args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )
    write_outputs(result, args.csv_output, args.json_output)
    print(json.dumps(result["cases"], indent=2))
    print(f"Wrote {args.csv_output} and {args.json_output}")


def run_sweep(
    config_path: str | Path,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    cases = []
    for case in sweep_cases():
        env = RacingEnv(config_path)
        env.control = replace(
            env.control,
            drivetrain=case["drivetrain"],
            front_cornering_stiffness=env.control.front_cornering_stiffness * case["grip_scale"],
            rear_cornering_stiffness=env.control.rear_cornering_stiffness * case["grip_scale"],
            max_lateral_force=env.control.max_lateral_force * case["grip_scale"],
        )
        controller = HeuristicController(env.track)
        episode_metrics: list[EpisodeMetrics] = []

        for episode_index in range(episodes):
            result = run_episode(
                env=env,
                controller=controller,
                controller_name="heuristic",
                episode=episode_index,
                seed=seed + episode_index,
                max_steps=max_steps,
            )
            episode_metrics.append(result["metrics"])

        summary = summarize_episodes(episode_metrics)
        cases.append(
            {
                **case,
                "summary": summary,
                "episodes": [asdict(metrics) for metrics in episode_metrics],
            }
        )

    return {
        "config": str(config_path),
        "episodes_per_case": episodes,
        "cases": cases,
    }


def sweep_cases() -> list[dict[str, Any]]:
    return [
        {"drivetrain": drivetrain, "grip_scale": grip_scale}
        for drivetrain in ("rwd", "fwd", "awd")
        for grip_scale in (0.75, 1.0, 1.25)
    ]


def write_outputs(result: dict[str, Any], csv_output: Path, json_output: Path) -> None:
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")

    with csv_output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "drivetrain",
                "grip_scale",
                "completion_rate",
                "off_track_rate",
                "mean_completed_lap_time",
                "best_completed_lap_time",
                "mean_reward",
                "mean_steps",
            ],
        )
        writer.writeheader()
        for case in result["cases"]:
            summary = case["summary"]
            writer.writerow(
                {
                    "drivetrain": case["drivetrain"],
                    "grip_scale": case["grip_scale"],
                    "completion_rate": summary["completion_rate"],
                    "off_track_rate": summary["off_track_rate"],
                    "mean_completed_lap_time": summary["mean_completed_lap_time"],
                    "best_completed_lap_time": summary["best_completed_lap_time"],
                    "mean_reward": summary["mean_reward"],
                    "mean_steps": summary["mean_steps"],
                }
            )


if __name__ == "__main__":
    main()
