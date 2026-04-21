from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from racesim.env.racing_env import RacingEnv
from racesim.eval.evaluate import telemetry_row
from racesim.eval.metrics import EpisodeMetrics, summarize_episodes


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
    ) -> None:
        super().__init__()
        self.eval_env_config = eval_env_config
        self.output_dir = output_dir
        self.eval_freq = eval_freq
        self.episodes = episodes
        self.max_steps = max_steps
        self.deterministic = deterministic
        self.enabled = enabled

    def _on_step(self) -> bool:
        if not self.enabled:
            return True
        if self.n_calls == 1 or self.n_calls % self.eval_freq == 0:
            run_live_eval(
                model=self.model,
                eval_env_config=self.eval_env_config,
                output_dir=self.output_dir,
                episodes=self.episodes,
                max_steps=self.max_steps,
                deterministic=self.deterministic,
                timestep=self.num_timesteps,
            )
        return True


def run_live_eval(
    model: object,
    eval_env_config: str,
    output_dir: Path,
    episodes: int,
    max_steps: int,
    deterministic: bool,
    timestep: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = RacingEnv(eval_env_config)
    env.max_episode_steps = max_steps
    episode_results = []

    for episode in range(episodes):
        observation, info = env.reset(seed=episode)
        total_reward = 0.0
        trajectory = []
        steps = 0
        for steps in range(1, max_steps + 1):
            action, _state = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            sim_time = steps * env.frame_skip * env.model.opt.timestep
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

        metrics = EpisodeMetrics(
            episode=episode,
            seed=episode,
            steps=steps,
            sim_time=float(steps * env.frame_skip * env.model.opt.timestep),
            total_reward=float(total_reward),
            lap_complete=bool(info["lap_complete"]),
            off_track=bool(info["off_track"]),
            lap_fraction=float(info["lap_fraction"]),
            cumulative_lap_fraction=float(info["cumulative_lap_fraction"]),
        )
        episode_results.append({"metrics": metrics, "trajectory": trajectory})

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
