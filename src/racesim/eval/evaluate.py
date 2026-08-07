from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from racesim.controllers.factory import CONTROLLERS, make_controller
from racesim.env.racing_env import RacingEnv
from racesim.eval.metrics import EpisodeMetrics, path_error_metrics, summarize_episodes
from racesim.eval.telemetry import telemetry_row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a controller on the racing environment.")
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=None,
        help="Optional multi-seed eval yaml (see configs/eval.yaml).",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--controller", choices=CONTROLLERS, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Episode seeds. When provided, one episode is run per seed "
            "(cycles if fewer than episodes)."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=3000)
    parser.add_argument("--lap-target", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/eval_heuristic.json"))
    parser.add_argument(
        "--record-trajectory",
        action="store_true",
        help="Include per-step trajectory and control telemetry in the output JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.eval_config is not None:
        result = evaluate_from_config(
            eval_config_path=args.eval_config,
            controller_name=args.controller,
            max_steps=args.max_steps,
            lap_target=args.lap_target,
            record_trajectory=args.record_trajectory,
            env_config_override=args.config,
            episodes_override=args.episodes,
            seeds_override=args.seeds,
        )
    else:
        result = evaluate(
            config_path=args.config or Path("configs/env.yaml"),
            controller_name=args.controller or "centerline",
            episodes=args.episodes if args.episodes is not None else 5,
            max_steps=args.max_steps,
            seed=args.seed,
            seeds=args.seeds,
            record_trajectory=args.record_trajectory,
            lap_target=args.lap_target,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if "by_controller" in result:
        print(
            json.dumps(
                {name: row["summary"] for name, row in result["by_controller"].items()},
                indent=2,
            )
        )
    else:
        print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


def load_eval_config(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Eval config must be a mapping: {path}")
    return data


def evaluate_from_config(
    eval_config_path: str | Path,
    controller_name: str | None = None,
    max_steps: int = 3000,
    lap_target: float | None = None,
    record_trajectory: bool = False,
    env_config_override: str | Path | None = None,
    episodes_override: int | None = None,
    seeds_override: list[int] | None = None,
) -> dict[str, Any]:
    """Run multi-seed evaluation from configs/eval.yaml-style settings."""
    cfg = load_eval_config(eval_config_path)
    env_config = env_config_override or cfg.get("env_config", "configs/env.yaml")
    episodes = episodes_override if episodes_override is not None else int(cfg.get("episodes", 5))
    if seeds_override is not None:
        seeds = seeds_override
    else:
        seeds = [int(s) for s in cfg.get("seeds", [0])]
    controllers = (
        [controller_name] if controller_name else list(cfg.get("controllers", ["heuristic"]))
    )
    # Drop controllers that are not registered env-side heuristics (e.g. bare "ppo").
    runnable = [name for name in controllers if name in CONTROLLERS]
    skipped = [name for name in controllers if name not in CONTROLLERS]

    by_controller: dict[str, Any] = {}
    for name in runnable:
        by_controller[name] = evaluate(
            config_path=env_config,
            controller_name=name,
            episodes=episodes,
            max_steps=max_steps,
            seed=seeds[0] if seeds else 0,
            seeds=seeds,
            record_trajectory=record_trajectory,
            lap_target=lap_target,
        )

    return {
        "eval_config": str(eval_config_path),
        "config": str(env_config),
        "episodes": episodes,
        "seeds": seeds,
        "skipped_controllers": skipped,
        "by_controller": by_controller,
        "summary": {name: result["summary"] for name, result in by_controller.items()},
    }


def episode_seeds(episodes: int, seed: int, seeds: list[int] | None) -> list[int]:
    if seeds:
        if len(seeds) >= episodes:
            return list(seeds[:episodes])
        # Cycle the provided seeds until episodes are filled.
        return [seeds[i % len(seeds)] for i in range(episodes)]
    return [seed + episode_index for episode_index in range(episodes)]


def evaluate(
    config_path: str | Path,
    controller_name: str,
    episodes: int,
    max_steps: int,
    seed: int,
    seeds: list[int] | None = None,
    record_trajectory: bool = False,
    lap_target: float | None = None,
) -> dict:
    env = RacingEnv(config_path)
    env.max_episode_steps = max_steps
    if lap_target is not None:
        env.lap_target = lap_target
    controller = make_controller(controller_name, env.track)
    resolved_seeds = episode_seeds(episodes, seed, seeds)
    episode_results = []

    for episode_index, episode_seed in enumerate(resolved_seeds):
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
        "seeds": resolved_seeds,
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
    lateral_errors: list[float] = []
    heading_errors: list[float] = []
    speeds: list[float] = []

    for _step in range(max_steps):
        action = controller.act(observation, info)

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        lateral_errors.append(float(info["lateral_error"]))
        heading_errors.append(float(info["heading_error"] or 0.0))
        speeds.append(float(info["speed"]))
        if record_trajectory:
            sim_time = steps * env.control_timestep()
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

    sim_time = steps * env.control_timestep()
    path = path_error_metrics(lateral_errors, heading_errors, speeds)
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
            **path,
        ),
        "trajectory": trajectory,
    }


if __name__ == "__main__":
    main()
