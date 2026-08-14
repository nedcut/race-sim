from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:  # pragma: no cover - optional [rl] extra
    BaseCallback = object  # type: ignore[misc, assignment]

from racesim.env.racing_env import RacingEnv
from racesim.eval.evaluate import telemetry_row
from racesim.eval.metrics import EpisodeMetrics, path_error_metrics, summarize_episodes
from racesim.training.agent_bundle import AgentBundleError, save_agent_bundle
from racesim.training.seeds import DEFAULT_LIVE_EVAL_SEED_BASE


class LiveEvalCallback(BaseCallback):
    """Stable-Baselines3 callback that writes latest eval rollout telemetry."""

    def __init__(
        self,
        eval_env_config: str,
        output_dir: Path,
        eval_freq: int,
        episodes: int,
        max_steps: int,
        deterministic: bool = True,
        enabled: bool = True,
        best_model_path: Path | None = None,
        vec_normalize: Any | None = None,
        seed_base: int = DEFAULT_LIVE_EVAL_SEED_BASE,
        bundle_dir: Path | None = None,
        train_config: dict[str, Any] | None = None,
        observation_action_schema: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.eval_env_config = eval_env_config
        self.output_dir = output_dir
        self.eval_freq = max(int(eval_freq), 1)
        self.episodes = episodes
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.enabled = enabled
        self.best_model_path = best_model_path
        self.vec_normalize = vec_normalize
        self.seed_base = int(seed_base)
        self.bundle_dir = Path(bundle_dir) if bundle_dir is not None else None
        self.train_config = train_config or {}
        self.observation_action_schema = observation_action_schema or {}
        self.best_score: tuple[float, float, float] | None = None
        self.next_eval_timestep = self.eval_freq

    def _on_step(self) -> bool:
        if not self.enabled:
            return True
        if self.num_timesteps >= self.next_eval_timestep:
            summary = run_live_eval(
                model=self.model,
                eval_env_config=self.eval_env_config,
                output_dir=self.output_dir,
                episodes=self.episodes,
                max_steps=self.max_steps,
                deterministic=self.deterministic,
                timestep=self.num_timesteps,
                vec_normalize=self.vec_normalize,
                seed_base=self.seed_base,
            )
            self._maybe_save_best(summary)
            self.next_eval_timestep = max(
                self.next_eval_timestep + self.eval_freq,
                self.num_timesteps + self.eval_freq,
            )
        return True

    def _maybe_save_best(self, summary: dict[str, float | int | None]) -> None:
        if self.best_model_path is None:
            return
        score = best_score(summary)
        if self.best_score is not None and score <= self.best_score:
            return
        self.best_score = score
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(self.best_model_path)
        vec_path = None
        if self.vec_normalize is not None and hasattr(self.vec_normalize, "save"):
            vec_path = Path(str(self.best_model_path) + "_vecnormalize.pkl")
            self.vec_normalize.save(str(vec_path))
        if self.bundle_dir is not None and self.observation_action_schema:
            try:
                save_agent_bundle(
                    self.bundle_dir,
                    model_path=Path(str(self.best_model_path) + ".zip"),
                    vecnormalize_path=vec_path,
                    train_config=self.train_config,
                    observation_action_schema=self.observation_action_schema,
                    timesteps=int(self.num_timesteps),
                    seed=self.train_config.get("seed"),
                    init_mode="best_eval",
                )
            except AgentBundleError as exc:
                print(f"Warning: could not save best-eval agent bundle: {exc}")
        print(
            "Saved best eval model "
            f"at {self.num_timesteps}: completion={summary['completion_rate']:.3f}, "
            f"lap_time={summary['mean_completed_lap_time']}, "
            f"reward={summary['mean_reward']:.1f}"
        )


def run_live_eval(
    model: object,
    eval_env_config: str,
    output_dir: Path,
    episodes: int,
    max_steps: int,
    deterministic: bool,
    timestep: int,
    vec_normalize: Any | None = None,
    seed_base: int = DEFAULT_LIVE_EVAL_SEED_BASE,
) -> dict[str, float | int | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = RacingEnv(eval_env_config)
    env.max_episode_steps = max_steps
    episode_results = []

    for episode in range(episodes):
        seed = int(seed_base) + episode
        observation, info = env.reset(seed=seed)
        total_reward = 0.0
        trajectory = []
        steps = 0
        lateral_errors: list[float] = []
        heading_errors: list[float] = []
        speeds: list[float] = []
        for steps in range(1, max_steps + 1):
            model_obs = maybe_normalize_obs(observation, vec_normalize)
            action, _state = model.predict(model_obs, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            lateral_errors.append(float(info["lateral_error"]))
            heading_errors.append(float(info["heading_error"] or 0.0))
            speeds.append(float(info["speed"]))
            sim_time = steps * env.control_timestep()
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
        episode_results.append({"metrics": metrics, "trajectory": trajectory})

    env.close()
    summary = summarize_episodes([result["metrics"] for result in episode_results])
    payload = {
        "timestep": timestep,
        "config": eval_env_config,
        "summary": summary,
        "episodes": [
            {
                "metrics": asdict(result["metrics"]),
                "trajectory": result["trajectory"],
            }
            for result in episode_results
        ],
    }
    latest_path = output_dir / "latest_rollout.json"
    latest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (output_dir / "metrics.jsonl").open("a", encoding="utf-8") as file:
        file.write(json.dumps({"timestep": timestep, **summary}) + "\n")
    return summary


def maybe_normalize_obs(observation: np.ndarray, vec_normalize: Any | None) -> np.ndarray:
    if vec_normalize is None:
        return observation
    was_training = getattr(vec_normalize, "training", False)
    if hasattr(vec_normalize, "training"):
        vec_normalize.training = False
    try:
        return vec_normalize.normalize_obs(np.asarray(observation, dtype=np.float32))
    finally:
        if hasattr(vec_normalize, "training"):
            vec_normalize.training = was_training


def best_score(summary: dict[str, float | int | None]) -> tuple[float, float, float]:
    """Rank eval summaries by completion, then lap time, then reward."""

    completion = float(summary["completion_rate"])
    lap_time = summary["mean_completed_lap_time"]
    reward = float(summary["mean_reward"])
    lap_time_score = -float(lap_time) if lap_time is not None else float("-inf")
    return (completion, lap_time_score, reward)
