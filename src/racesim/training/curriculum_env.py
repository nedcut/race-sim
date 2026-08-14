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
        reset_options: dict[str, Any] | None = None,
        reset_option_ranges: dict[str, tuple[float, float] | list[float]] | None = None,
    ) -> None:
        super().__init__()
        if not env_configs:
            raise ValueError("TrackCurriculumEnv requires at least one env config.")

        self.envs = [RacingEnv(path) for path in env_configs]
        self.env_configs = [str(path) for path in env_configs]
        self.randomize_reset = randomize_reset
        self.reset_options = dict(reset_options or {})
        self.reset_option_ranges = dict(reset_option_ranges or {})
        self.current_index = 0
        self.current_env = self.envs[self.current_index]

        obs_shapes = {env.observation_space.shape for env in self.envs}
        if len(obs_shapes) != 1:
            raise ValueError(
                "TrackCurriculumEnv requires matching observation shapes across configs; "
                f"got {sorted(obs_shapes)}"
            )
        action_shapes = {env.action_space.shape for env in self.envs}
        if len(action_shapes) != 1:
            raise ValueError(
                "TrackCurriculumEnv requires matching action shapes across configs; "
                f"got {sorted(action_shapes)}"
            )

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
        if seed is not None:
            for index, env in enumerate(self.envs):
                env.reset(seed=int(seed) + index + 1)
        self.current_index = int(self.np_random.choice(len(self.envs), p=self.probabilities))
        self.current_env = self.envs[self.current_index]

        reset_options = dict(options or {})
        reset_options.setdefault("randomize", self.randomize_reset)
        reset_options.update(self.reset_options)
        reset_options.update(self._sample_reset_options())
        observation, info = self.current_env.reset(seed=seed, options=reset_options)
        return observation, self._annotate_info(info)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.current_env.step(action)
        return observation, reward, terminated, truncated, self._annotate_info(info)

    def set_probabilities(self, probabilities: list[float]) -> None:
        self.probabilities = normalize_probabilities(probabilities, len(self.envs))

    def close(self) -> None:
        for env in self.envs:
            env.close()
        super().close()

    def _annotate_info(self, info: dict[str, Any]) -> dict[str, Any]:
        annotated = dict(info)
        annotated["track_index"] = self.current_index
        annotated["env_config"] = self.env_configs[self.current_index]
        annotated["track_name"] = self.current_env.track.name
        return annotated

    def _sample_reset_options(self) -> dict[str, float]:
        sampled: dict[str, float] = {}
        for key, value_range in self.reset_option_ranges.items():
            if len(value_range) != 2:
                raise ValueError(f"Reset option range for {key!r} must have two values.")
            low, high = float(value_range[0]), float(value_range[1])
            if high < low:
                raise ValueError(f"Reset option range for {key!r} must be ordered low to high.")
            sampled[key] = float(self.np_random.uniform(low, high))
        return sampled


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
