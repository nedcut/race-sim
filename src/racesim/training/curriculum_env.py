from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from racesim.env.racing_env import RacingEnv


class TrackCurriculumEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium wrapper that samples one of several RacingEnv configs on reset."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_configs: list[str | Path],
        randomize_reset: bool = True,
        lap_target: float | None = None,
        max_episode_steps: int | None = None,
        probabilities: list[float] | None = None,
    ) -> None:
        super().__init__()
        if not env_configs:
            raise ValueError("TrackCurriculumEnv requires at least one env config.")

        self.envs = [RacingEnv(path) for path in env_configs]
        self.env_configs = [str(path) for path in env_configs]
        self.randomize_reset = randomize_reset
        self.current_index = 0
        self.current_env = self.envs[self.current_index]

        if lap_target is not None:
            for env in self.envs:
                env.lap_target = lap_target
        if max_episode_steps is not None:
            for env in self.envs:
                env.max_episode_steps = max_episode_steps

        self.probabilities = normalize_probabilities(probabilities, len(self.envs))
        self.action_space = self.current_env.action_space
        self.observation_space = self.current_env.observation_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_index = int(self.np_random.choice(len(self.envs), p=self.probabilities))
        self.current_env = self.envs[self.current_index]

        reset_options = dict(options or {})
        reset_options.setdefault("randomize", self.randomize_reset)
        observation, info = self.current_env.reset(seed=seed, options=reset_options)
        return observation, self._annotate_info(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.current_env.step(action)
        return observation, reward, terminated, truncated, self._annotate_info(info)

    def set_probabilities(self, probabilities: list[float]) -> None:
        self.probabilities = normalize_probabilities(probabilities, len(self.envs))

    def _annotate_info(self, info: dict[str, Any]) -> dict[str, Any]:
        annotated = dict(info)
        annotated["track_index"] = self.current_index
        annotated["env_config"] = self.env_configs[self.current_index]
        annotated["track_name"] = self.current_env.track.name
        return annotated


def normalize_probabilities(probabilities: list[float] | None, size: int) -> np.ndarray | None:
    if probabilities is None:
        return None
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (size,):
        raise ValueError("Track probabilities must match env config count.")
    total = float(values.sum())
    if total <= 0:
        raise ValueError("Track probabilities must sum to a positive value.")
    return values / total
