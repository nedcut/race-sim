from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from racesim.training.curriculum_env import TrackCurriculumEnv
from racesim.training.live_eval import LiveEvalCallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the racing environment.")
    parser.add_argument("--config", type=Path, default=Path("configs/train_ppo_oval.yaml"))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and build envs only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config)
    output_dir = Path(config.get("output_dir", "results/ppo"))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    env = make_training_env(config)
    observation, info = env.reset(seed=int(config.get("seed", 0)))
    print(
        "Training env ready: "
        f"obs_shape={observation.shape}, action_shape={env.action_space.shape}, "
        f"initial_track={info['track_name']}"
    )
    if args.dry_run:
        print("Dry run complete.")
        return

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv

    seed = int(config.get("seed", 0))
    vec_env = DummyVecEnv([lambda: Monitor(make_training_env(config))])

    ppo_config = dict(config.get("ppo", {}))
    policy = ppo_config.pop("policy", "MlpPolicy")
    model = PPO(
        policy,
        vec_env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(output_dir),
        **ppo_config,
    )

    callbacks = []
    eval_config = config.get("eval", {})
    live_config = config.get("live", {})
    if eval_config:
        callbacks.append(
            LiveEvalCallback(
                eval_env_config=eval_config.get("env_config", "configs/env.yaml"),
                output_dir=Path(live_config.get("directory", output_dir / "live")),
                eval_freq=int(eval_config.get("frequency", 10000)),
                episodes=int(eval_config.get("episodes", 3)),
                max_steps=int(eval_config.get("max_steps", 1500)),
                deterministic=bool(eval_config.get("deterministic", True)),
                enabled=bool(live_config.get("enabled", True)),
            )
        )

    callback = CallbackList(callbacks) if callbacks else None
    model.learn(total_timesteps=int(config["total_timesteps"]), callback=callback)
    final_path = output_dir / "final_model"
    model.save(final_path)
    print(f"Saved {final_path}.zip")


def load_train_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def make_training_env(config: dict[str, Any]) -> TrackCurriculumEnv:
    env_config = config.get("env", {})
    env_configs = env_config.get("env_configs")
    if not env_configs:
        legacy_env_config = config.get("env_config")
        env_configs = [legacy_env_config] if legacy_env_config else ["configs/env.yaml"]
    return TrackCurriculumEnv(
        env_configs=env_configs,
        randomize_reset=bool(env_config.get("randomize_reset", True)),
        lap_target=env_config.get("lap_target"),
        max_episode_steps=env_config.get("max_episode_steps"),
        probabilities=env_config.get("probabilities"),
    )


if __name__ == "__main__":
    main()
