"""Collect expert (heuristic controller) transitions for BC warm-start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from racesim.controllers.factory import make_controller
from racesim.env.racing_env import RacingEnv
from racesim.paths import default_env_config
from racesim.training.provenance import (
    array_digest,
    dependency_versions,
    git_sha,
    hash_env_config_tree,
    observation_action_schema,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect expert transitions for imitation / BC.")
    parser.add_argument("--config", type=Path, default=None, help="Env YAML")
    parser.add_argument("--controller", default="racing_line", help="Builtin controller name")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--randomize-reset", action="store_true")
    parser.add_argument("--lap-target", type=float, default=1.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/expert/racing_line_oval.npz"),
        help="Output .npz with observations/actions arrays",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config or default_env_config()
    result = collect_expert(
        config_path=config_path,
        controller_name=args.controller,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        randomize_reset=args.randomize_reset,
        lap_target=args.lap_target,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = result["metadata"]
    np.savez_compressed(
        args.output,
        observations=result["observations"],
        actions=result["actions"],
        rewards=result["rewards"],
        dones=result["dones"],
        episode_starts=result["episode_starts"],
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    meta_path = args.output.with_suffix(".json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    print(f"Wrote {args.output} and {meta_path}")


def collect_expert(
    config_path: str | Path,
    controller_name: str,
    episodes: int,
    max_steps: int,
    seed: int = 0,
    randomize_reset: bool = False,
    lap_target: float | None = None,
) -> dict[str, Any]:
    env = RacingEnv(config_path)
    if lap_target is not None:
        env.lap_target = lap_target
    controller = make_controller(controller_name, env.track)
    schema = observation_action_schema(env)

    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    episode_starts: list[bool] = []
    episode_rewards: list[float] = []
    completions: list[float] = []

    for episode in range(episodes):
        observation, info = env.reset(
            seed=seed + episode,
            options={"randomize": randomize_reset},
        )
        total_reward = 0.0
        episode_start = True
        collector_truncated = True
        for _ in range(max_steps):
            action = np.asarray(controller.act(observation, info), dtype=np.float32)
            observations.append(np.asarray(observation, dtype=np.float32))
            actions.append(action)
            episode_starts.append(episode_start)
            episode_start = False

            observation, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            rewards.append(float(reward))
            dones.append(done)
            total_reward += float(reward)
            if terminated or truncated:
                collector_truncated = False
                break
        if collector_truncated and dones:
            # Horizon hit without env terminal: mark the last transition truncated.
            dones[-1] = True
        episode_rewards.append(total_reward)
        completions.append(1.0 if info.get("lap_complete") else 0.0)

    env.close()
    arrays = {
        "observations": np.stack(observations, axis=0) if observations else np.zeros((0, 0)),
        "actions": np.stack(actions, axis=0) if actions else np.zeros((0, 0)),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "episode_starts": np.asarray(episode_starts, dtype=bool),
    }
    metadata = {
        "config": str(config_path),
        "controller": controller_name,
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": seed,
        "n_transitions": int(arrays["observations"].shape[0]),
        "mean_reward": (float(np.asarray(episode_rewards).mean()) if episode_rewards else 0.0),
        "completion_rate": (float(np.asarray(completions).mean()) if completions else 0.0),
        "observation_action_schema": schema,
        "action_bounds": {"low": schema["action_low"], "high": schema["action_high"]},
        "env_config_hashes": hash_env_config_tree(config_path),
        "git_sha": git_sha(),
        "dependencies": dependency_versions(),
        "dataset_digest": array_digest(arrays),
    }
    return {
        **arrays,
        "episode_rewards": np.asarray(episode_rewards, dtype=np.float32),
        "completions": np.asarray(completions, dtype=np.float32),
        "metadata": metadata,
    }


if __name__ == "__main__":
    main()
