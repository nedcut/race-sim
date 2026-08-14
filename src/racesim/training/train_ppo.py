from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import yaml

from racesim.paths import default_train_config
from racesim.training.agent_bundle import (
    AgentBundleError,
    is_agent_bundle,
    load_agent_bundle,
    normalize_enabled,
    resolve_model_checkpoint,
    save_agent_bundle,
)
from racesim.training.curriculum_env import TrackCurriculumEnv
from racesim.training.live_eval import LiveEvalCallback
from racesim.training.provenance import observation_action_schema
from racesim.training.schedules import prepare_ppo_schedules
from racesim.training.seeds import (
    DEFAULT_LIVE_EVAL_SEED_BASE,
    DEFAULT_TRAIN_SEED,
)
from racesim.utils.device import apply_device_to_ppo_config, resolve_torch_device

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover - dry-run config tests do not need SB3 installed.
    BaseCallback = object  # type: ignore[misc,assignment]

InitMode = Literal["fresh", "pretrained", "resume"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on the racing environment.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and build envs only; do not write artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_train_config(args.config or default_train_config())
    output_dir = Path(config.get("output_dir", "results/ppo"))
    init_mode = resolve_init_mode(config)

    env = make_training_env(config)
    observation, info = env.reset(seed=int(config.get("seed", DEFAULT_TRAIN_SEED)))
    schema = observation_action_schema(env)
    print(
        "Training env ready: "
        f"obs_shape={observation.shape}, action_shape={env.action_space.shape}, "
        f"initial_track={info['track_name']}, init_mode={init_mode}"
    )
    if args.dry_run:
        env.close()
        print("Dry run complete.")
        return
    env.close()

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_config.json").write_text(
        json.dumps(config, indent=2, default=str), encoding="utf-8"
    )

    seed = int(config.get("seed", DEFAULT_TRAIN_SEED))
    set_random_seed(seed)
    vec_env = make_vec_env(config, Monitor, DummyVecEnv, SubprocVecEnv)
    vec_env, vec_stats_path = wrap_or_load_vecnormalize(vec_env, config, init_mode)

    ppo_config = apply_device_to_ppo_config(prepare_ppo_schedules(dict(config.get("ppo", {}))))
    policy = ppo_config.pop("policy", "MlpPolicy")
    device = ppo_config.get("device", resolve_torch_device("auto"))

    reset_num_timesteps = True
    if init_mode == "resume":
        model_path, _stats = resolve_init_checkpoint(config["resume"], require_vecnormalize=False)
        model = PPO.load(str(model_path), env=vec_env, device=device)
        model.set_env(vec_env)
        reset_num_timesteps = False
        print(f"Resumed full PPO checkpoint from {model_path} (reset_num_timesteps=False)")
    elif init_mode == "pretrained":
        model = PPO(
            policy,
            vec_env,
            seed=seed,
            verbose=1,
            tensorboard_log=str(output_dir / "tb"),
            **ppo_config,
        )
        donor_path, _stats = resolve_init_checkpoint(
            config["pretrained"], require_vecnormalize=False
        )
        donor = PPO.load(str(donor_path), device=device)
        model.policy.load_state_dict(donor.policy.state_dict())
        print(
            f"Loaded pretrained policy weights from {donor_path} into a fresh PPO "
            "(optimizer, schedules, and timestep counters were not restored)"
        )
    else:
        model = PPO(
            policy,
            vec_env,
            seed=seed,
            verbose=1,
            tensorboard_log=str(output_dir / "tb"),
            **ppo_config,
        )
    print(f"PPO device: {model.device} (resolved from config as {device})")

    n_envs = int(config.get("env", {}).get("n_envs", 1))
    callbacks: list[Any] = []
    stage_config = config.get("curriculum", {}).get("stages", [])
    if stage_config:
        callbacks.append(CurriculumStageCallback(stage_config))

    checkpoint_freq = int(config.get("checkpoint_freq", 0) or 0)
    if checkpoint_freq > 0:
        save_freq = timestep_to_callback_calls(checkpoint_freq, n_envs)
        checkpoint_kwargs: dict[str, Any] = {
            "save_freq": save_freq,
            "save_path": str(output_dir / "checkpoints"),
            "name_prefix": "ppo",
            "save_replay_buffer": False,
        }
        try:
            callbacks.append(CheckpointCallback(**checkpoint_kwargs, save_vecnormalize=True))
        except TypeError:
            callbacks.append(CheckpointCallback(**checkpoint_kwargs))

    if bool(config.get("save_vecnormalize", True)) and isinstance(vec_env, VecNormalize):
        callbacks.append(
            SaveVecNormalizeCallback(
                save_path=output_dir / "vecnormalize.pkl",
                save_freq=max(int(config.get("vecnormalize_save_freq", 10_000)), 1),
            )
        )

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
                vec_normalize=vec_env if isinstance(vec_env, VecNormalize) else None,
                seed_base=int(eval_config.get("seed_base", DEFAULT_LIVE_EVAL_SEED_BASE)),
                bundle_dir=output_dir / "best_agent"
                if bool(eval_config.get("save_best", True))
                else None,
                train_config=config,
                observation_action_schema=schema,
            )
        )

    callback = CallbackList(callbacks) if callbacks else None
    model.learn(
        total_timesteps=int(config["total_timesteps"]),
        callback=callback,
        progress_bar=bool(config.get("progress_bar", False)),
        reset_num_timesteps=reset_num_timesteps,
    )
    final_path = output_dir / "final_model"
    model.save(final_path)
    print(f"Saved {final_path}.zip")
    vec_path = output_dir / "vecnormalize.pkl"
    if isinstance(vec_env, VecNormalize):
        vec_env.save(str(vec_path))
        print(f"Saved {vec_path}")
    elif vec_stats_path is not None:
        vec_path = Path(vec_stats_path)
    try:
        bundle_dir = save_agent_bundle(
            output_dir / "agent_bundle",
            model_path=Path(str(final_path) + ".zip"),
            vecnormalize_path=vec_path if vec_path.exists() else None,
            train_config=config,
            observation_action_schema=schema,
            timesteps=int(getattr(model, "num_timesteps", config.get("total_timesteps", 0))),
            seed=seed,
            init_mode=init_mode,
        )
        print(f"Saved agent bundle at {bundle_dir}")
    except AgentBundleError as exc:
        vec_env.close()
        raise SystemExit(f"Failed to save agent bundle: {exc}") from exc
    vec_env.close()


def load_train_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_init_mode(config: dict[str, Any]) -> InitMode:
    pretrained = config.get("pretrained")
    resume = config.get("resume")
    if pretrained and resume:
        raise ValueError(
            "Specify only one of `pretrained` (policy weights / BC warm-start) or "
            "`resume` (full PPO + VecNormalize + timestep restore)"
        )
    if resume:
        return "resume"
    if pretrained:
        return "pretrained"
    return "fresh"


def timestep_to_callback_calls(timestep_freq: int, n_envs: int) -> int:
    """Convert a timestep interval to SB3 CheckpointCallback `save_freq` (`n_calls`)."""
    return max(int(timestep_freq) // max(int(n_envs), 1), 1)


def resolve_init_checkpoint(
    path: Path | str,
    *,
    require_vecnormalize: bool,
) -> tuple[Path, Path | None]:
    candidate = Path(path)
    if is_agent_bundle(candidate):
        bundle = load_agent_bundle(candidate)
        if require_vecnormalize and bundle.vecnormalize_path is None:
            raise AgentBundleError(f"Resume bundle is missing VecNormalize stats: {candidate}")
        return bundle.model_path, bundle.vecnormalize_path
    model_path, vec_path = resolve_model_checkpoint(candidate)
    if require_vecnormalize and vec_path is None:
        raise AgentBundleError(
            f"normalize.enabled is true but VecNormalize stats were not found for {candidate}"
        )
    return model_path, vec_path


def wrap_or_load_vecnormalize(
    vec_env: Any, config: dict[str, Any], init_mode: InitMode
) -> tuple[Any, Path | None]:
    from stable_baselines3.common.vec_env import VecNormalize

    if not normalize_enabled(config):
        return vec_env, None

    normalize_cfg = config.get("normalize", {})
    if isinstance(normalize_cfg, bool):
        normalize_cfg = {"enabled": normalize_cfg}

    stats_path: Path | None = None
    if init_mode == "resume":
        _model, stats_path = resolve_init_checkpoint(config["resume"], require_vecnormalize=True)
    elif init_mode == "pretrained":
        _model, stats_path = resolve_init_checkpoint(
            config["pretrained"], require_vecnormalize=True
        )

    gamma = float(config.get("ppo", {}).get("gamma", 0.99))
    if stats_path is not None:
        vec_env = VecNormalize.load(str(stats_path), vec_env)
        vec_env.training = True
        vec_env.norm_reward = bool(normalize_cfg.get("norm_reward", True))
        print(f"Loaded VecNormalize stats from {stats_path}")
        return vec_env, stats_path

    vec_env = VecNormalize(
        vec_env,
        training=True,
        norm_obs=bool(normalize_cfg.get("norm_obs", True)),
        norm_reward=bool(normalize_cfg.get("norm_reward", True)),
        clip_obs=float(normalize_cfg.get("clip_obs", 10.0)),
        clip_reward=float(normalize_cfg.get("clip_reward", 10.0)),
        gamma=gamma,
    )
    print(
        "VecNormalize enabled: "
        f"norm_obs={normalize_cfg.get('norm_obs', True)}, "
        f"norm_reward={normalize_cfg.get('norm_reward', True)}"
    )
    return vec_env, None


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
            env.reset(seed=int(config.get("seed", DEFAULT_TRAIN_SEED)) + rank)
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
        while self.next_stage < len(self.stages) and self.num_timesteps >= int(
            self.stages[self.next_stage]["at_timesteps"]
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


class SaveVecNormalizeCallback(BaseCallback):
    """Periodically snapshot VecNormalize running statistics (interval in timesteps)."""

    def __init__(self, save_path: Path, save_freq: int) -> None:
        super().__init__()
        self.save_path = Path(save_path)
        self.save_freq = max(int(save_freq), 1)
        self.next_save_timestep = self.save_freq

    def _on_step(self) -> bool:
        if self.num_timesteps < self.next_save_timestep:
            return True
        env = self.training_env
        if hasattr(env, "save") and hasattr(env, "normalize_obs"):
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            env.save(str(self.save_path))
        self.next_save_timestep = max(
            self.next_save_timestep + self.save_freq,
            self.num_timesteps + self.save_freq,
        )
        return True


if __name__ == "__main__":
    main()
