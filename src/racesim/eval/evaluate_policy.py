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
from racesim.eval.metrics import EpisodeMetrics, path_error_metrics, summarize_episodes
from racesim.paths import default_env_config
from racesim.training.agent_bundle import (
    AgentBundleError,
    is_agent_bundle,
    load_agent_bundle,
    resolve_model_checkpoint,
)
from racesim.training.seeds import DEFAULT_FINAL_EVAL_SEED_BASE

PredictFn = Callable[[np.ndarray, dict[str, Any]], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a policy (agent bundle, SB3 zip, or predict demo)."
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Agent bundle directory (preferred). Includes model + VecNormalize + schema.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="SB3 checkpoint (.zip). Errors if a required VecNormalize file is missing.",
    )
    parser.add_argument(
        "--predict-demo",
        action="store_true",
        help="Run a built-in open-loop throttle demo policy (no --model/--bundle).",
    )
    parser.add_argument("--config", type=Path, default=None)
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
    parser.add_argument(
        "--vecnormalize",
        type=Path,
        default=None,
        help="VecNormalize stats (.pkl). Required when the checkpoint was trained with normalize.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=DEFAULT_FINAL_EVAL_SEED_BASE,
        help="Base seed for multi-episode evaluation (seed = base + episode index). "
        "Default 20000 is the final-evaluation range (disjoint from training/live eval).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.predict_demo:
        predict = demo_predict
        controller_name = args.controller_name or "predict_demo"
    elif args.bundle is not None or (args.model is not None and is_agent_bundle(args.model)):
        bundle_path = args.bundle if args.bundle is not None else args.model
        predict = predict_from_bundle(
            bundle_path,
            deterministic=args.deterministic,
            env_config=args.config or default_env_config(),
        )
        controller_name = args.controller_name or "ppo"
    elif args.model is not None:
        from stable_baselines3 import PPO

        try:
            model_path, vec_path = resolve_model_checkpoint(args.model, args.vecnormalize)
        except AgentBundleError as exc:
            raise SystemExit(str(exc)) from exc
        model = PPO.load(model_path)
        predict = sb3_predict(
            model,
            deterministic=args.deterministic,
            vecnormalize_path=vec_path,
            env_config=args.config or default_env_config(),
            require_vecnormalize=vec_path is not None,
        )
        controller_name = args.controller_name or "ppo"
    else:
        raise SystemExit("Provide --bundle PATH, --model PATH, or --predict-demo")

    seeds = [args.seed_base + episode for episode in range(args.episodes)]
    result = evaluate_predict(
        predict=predict,
        config_path=args.config or default_env_config(),
        episodes=args.episodes,
        max_steps=args.max_steps,
        lap_target=args.lap_target,
        record_trajectory=args.record_trajectory,
        reset_options=reset_options(args),
        controller_name=controller_name,
        seeds=seeds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"Wrote {args.output}")


def predict_from_bundle(
    bundle_path: Path | str,
    *,
    deterministic: bool,
    env_config: Path | str,
) -> PredictFn:
    from stable_baselines3 import PPO

    try:
        bundle = load_agent_bundle(bundle_path)
    except AgentBundleError as exc:
        raise SystemExit(str(exc)) from exc
    model = PPO.load(bundle.model_path)
    return sb3_predict(
        model,
        deterministic=deterministic,
        vecnormalize_path=bundle.vecnormalize_path,
        env_config=env_config,
        require_vecnormalize=bundle.vecnormalize_path is not None
        or bool(bundle.manifest.get("normalize_enabled")),
    )


def evaluate_policy_model(
    model: object,
    config_path: Path,
    episodes: int,
    max_steps: int,
    lap_target: float,
    deterministic: bool,
    record_trajectory: bool,
    reset_options: dict[str, Any] | None = None,
    vecnormalize_path: Path | str | None = None,
    seeds: list[int] | None = None,
) -> dict:
    """Evaluate an SB3-style model with ``predict(obs, deterministic=...)``."""
    return evaluate_predict(
        predict=sb3_predict(
            model,
            deterministic=deterministic,
            vecnormalize_path=vecnormalize_path,
            env_config=config_path,
            require_vecnormalize=vecnormalize_path is not None,
        ),
        config_path=config_path,
        episodes=episodes,
        max_steps=max_steps,
        lap_target=lap_target,
        record_trajectory=record_trajectory,
        reset_options=reset_options,
        controller_name="ppo",
        seeds=seeds,
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
        lateral_errors: list[float] = []
        heading_errors: list[float] = []
        speeds: list[float] = []
        for steps in range(1, max_steps + 1):
            action = np.asarray(predict(observation, info), dtype=np.float32)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
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

        path = path_error_metrics(lateral_errors, heading_errors, speeds)
        metrics = EpisodeMetrics(
            episode=episode,
            seed=seed,
            steps=steps,
            sim_time=float(steps * env.control_timestep()),
            total_reward=float(total_reward),
            lap_complete=bool(info["lap_complete"]),
            off_track=bool(info["off_track"]),
            lap_fraction=float(info["lap_fraction"]),
            cumulative_lap_fraction=float(info["cumulative_lap_fraction"]),
            **path,
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


def sb3_predict(
    model: object,
    deterministic: bool = True,
    vecnormalize_path: Path | str | None = None,
    env_config: Path | str | None = None,
    require_vecnormalize: bool = False,
) -> PredictFn:
    if require_vecnormalize and vecnormalize_path is None:
        raise AgentBundleError(
            "This policy was trained with VecNormalize; refusing to evaluate on raw observations."
        )
    normalizer = None
    if vecnormalize_path is not None:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        if env_config is None:
            raise ValueError("env_config is required when loading VecNormalize stats")

        def _make() -> RacingEnv:
            return RacingEnv(env_config)

        vec = DummyVecEnv([_make])
        try:
            normalizer = VecNormalize.load(str(vecnormalize_path), vec)
        except Exception as exc:
            raise AgentBundleError(
                f"Failed to load VecNormalize stats from {vecnormalize_path}: {exc}"
            ) from exc
        normalizer.training = False
        normalizer.norm_reward = False

    def predict(observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        del info
        obs = observation
        if normalizer is not None:
            from racesim.training.live_eval import maybe_normalize_obs

            obs = maybe_normalize_obs(observation, normalizer)
        action, _state = model.predict(obs, deterministic=deterministic)
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
