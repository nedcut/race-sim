from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from racesim.env.racing_env import RacingEnv
from racesim.eval.evaluate import telemetry_row
from racesim.eval.metrics import EpisodeMetrics, summarize_episodes

PredictFn = Callable[[np.ndarray, dict[str, Any]], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a policy (SB3 zip or any predict(obs, info) -> action)."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Stable-Baselines3 checkpoint (.zip). Required unless --predict-demo.",
    )
    parser.add_argument(
        "--predict-demo",
        action="store_true",
        help="Run a built-in open-loop throttle demo policy (no --model).",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--lap-target", type=float, default=1.0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--record-trajectory", action="store_true")
    parser.add_argument("--grip-scale", type=float, default=None)
    parser.add_argument("--randomize-reset", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("results/eval_policy.json"))
    parser.add_argument(
        "--controller-name",
        default=None,
        help="Label stored in the result JSON (defaults to ppo or predict).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predict_demo:
        predict = demo_predict
        controller_name = args.controller_name or "predict_demo"
    elif args.model is not None:
        from stable_baselines3 import PPO

        model = PPO.load(args.model)
        predict = sb3_predict(model, deterministic=args.deterministic)
        controller_name = args.controller_name or "ppo"
    else:
        raise SystemExit("Provide --model PATH or --predict-demo")

    result = evaluate_predict(
        predict=predict,
        config_path=args.config,
        episodes=args.episodes,
        max_steps=args.max_steps,
        lap_target=args.lap_target,
        record_trajectory=args.record_trajectory,
        reset_options=reset_options(args),
        controller_name=controller_name,
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
    reset_options: dict[str, Any] | None = None,
) -> dict:
    """Evaluate an SB3-style model with ``predict(obs, deterministic=...)``."""
    return evaluate_predict(
        predict=sb3_predict(model, deterministic=deterministic),
        config_path=config_path,
        episodes=episodes,
        max_steps=max_steps,
        lap_target=lap_target,
        record_trajectory=record_trajectory,
        reset_options=reset_options,
        controller_name="ppo",
    )


def evaluate_predict(
    predict: PredictFn,
    config_path: Path | str,
    episodes: int,
    max_steps: int,
    lap_target: float = 1.0,
    record_trajectory: bool = False,
    reset_options: dict[str, Any] | None = None,
    controller_name: str = "predict",
    seeds: list[int] | None = None,
) -> dict:
    """Policy-agnostic evaluation: ``predict(observation, info) -> action``."""
    env = RacingEnv(config_path)
    env.max_episode_steps = max_steps
    env.lap_target = lap_target
    results = []
    for episode in range(episodes):
        seed = seeds[episode] if seeds is not None else episode
        observation, info = env.reset(seed=seed, options=reset_options)
        total_reward = 0.0
        trajectory: list[dict[str, Any]] = []
        steps = 0
        for steps in range(1, max_steps + 1):
            action = np.asarray(predict(observation, info), dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
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

        metrics = EpisodeMetrics(
            episode=episode,
            seed=seed,
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
        "controller": controller_name,
        "config": str(config_path),
        "reset_options": reset_options or {},
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


def sb3_predict(model: object, deterministic: bool = True) -> PredictFn:
    def predict(observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        del info
        action, _state = model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.float32)

    return predict


def demo_predict(observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
    """Open-loop partial throttle for smoke-testing evaluate_predict."""
    del observation, info
    return np.array([0.0, 0.25, 0.0], dtype=np.float32)


def reset_options(args: argparse.Namespace) -> dict[str, Any] | None:
    options: dict[str, Any] = {}
    if args.grip_scale is not None:
        options["grip_scale"] = args.grip_scale
    if args.randomize_reset:
        options["randomize"] = True
    return options or None


if __name__ == "__main__":
    main()
