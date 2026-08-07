from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from racesim.training.curriculum_env import TrackCurriculumEnv
from racesim.training.live_eval import LiveEvalCallback
from racesim.utils.device import apply_device_to_ppo_config, resolve_torch_device

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover - dry-run config tests do not need SB3 installed.
    BaseCallback = object  # type: ignore[misc,assignment]


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
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    seed = int(config.get("seed", 0))
    vec_env = make_vec_env(config, Monitor, DummyVecEnv, SubprocVecEnv)

    ppo_config = apply_device_to_ppo_config(dict(config.get("ppo", {})))
    policy = ppo_config.pop("policy", "MlpPolicy")
    device = ppo_config.get("device", resolve_torch_device("auto"))
    model = PPO(
        policy,
        vec_env,
        seed=seed,
        verbose=1,
        tensorboard_log=str(output_dir),
        **ppo_config,
    )
    print(f"PPO device: {model.device} (resolved from config as {device})")

    callbacks = []
    stage_config = config.get("curriculum", {}).get("stages", [])
    if stage_config:
        callbacks.append(CurriculumStageCallback(stage_config))
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
                best_model_path=output_dir / "best_model"
                if bool(eval_config.get("save_best", True))
                else None,
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
        reset_options=env_config.get("reset_options"),
        reset_option_ranges=env_config.get("reset_option_ranges"),
    )


def make_vec_env(
    config: dict[str, Any],
    monitor_cls: type,
    dummy_vec_env_cls: type,
    subproc_vec_env_cls: type,
) -> Any:
    env_config = config.get("env", {})
    n_envs = int(env_config.get("n_envs", 1))
    vec_env_type = str(env_config.get("vec_env", "dummy")).lower()

    def make_env(rank: int) -> Any:
        def _init() -> Any:
            env = make_training_env(config)
            env.reset(seed=int(config.get("seed", 0)) + rank)
            return monitor_cls(env)

        return _init

    env_fns = [make_env(index) for index in range(n_envs)]
    if vec_env_type == "dummy":
        return dummy_vec_env_cls(env_fns)
    if vec_env_type == "subproc":
        start_method = env_config.get("start_method", "forkserver")
        return subproc_vec_env_cls(env_fns, start_method=start_method)
    raise ValueError(f"Unsupported vec_env type: {vec_env_type}")


class CurriculumStageCallback(BaseCallback):
    """Update track sampling probabilities at configured training steps."""

    def __init__(self, stages: list[dict[str, Any]]) -> None:
        super().__init__()
        self.stages = sorted(stages, key=lambda stage: int(stage["at_timesteps"]))
        self.next_stage = 0

    def _on_step(self) -> bool:
        while (
            self.next_stage < len(self.stages)
            and self.num_timesteps >= int(self.stages[self.next_stage]["at_timesteps"])
        ):
            stage = self.stages[self.next_stage]
            probabilities = [float(value) for value in stage["probabilities"]]
            self.training_env.env_method("set_probabilities", probabilities)
            print(
                "Curriculum stage "
                f"{self.next_stage + 1}/{len(self.stages)} at {self.num_timesteps}: "
                f"probabilities={probabilities}"
            )
            self.next_stage += 1
        return True


if __name__ == "__main__":
    main()
