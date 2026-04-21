from __future__ import annotations

import numpy as np

from racesim.env.racing_env import RacingEnv


def test_env_reset_returns_valid_observation() -> None:
    env = RacingEnv("configs/env.yaml")

    observation, info = env.reset(seed=0)

    assert observation.shape == env.observation_space.shape
    assert env.observation_space.contains(observation)
    assert info["lap_fraction"] == 0.0
    assert info["cumulative_lap_fraction"] == 0.0
    assert not info["off_track"]
    assert not info["lap_complete"]


def test_env_step_advances_progress_with_throttle() -> None:
    env = RacingEnv("configs/env.yaml")
    env.reset(seed=0)

    observation, reward, terminated, truncated, info = env.step(
        np.array([0.0, 0.5, 0.0], dtype=np.float32)
    )

    assert observation.shape == env.observation_space.shape
    assert reward > 0.0
    assert not terminated
    assert not truncated
    assert info["progress"] > 0.0


def test_env_eventually_truncates_when_step_limit_is_small() -> None:
    env = RacingEnv("configs/env.yaml")
    env.max_episode_steps = 1
    env.reset(seed=0)

    _observation, _reward, _terminated, truncated, _info = env.step(
        np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )

    assert truncated
