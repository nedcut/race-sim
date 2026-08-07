from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

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


def test_env_can_target_more_than_one_lap() -> None:
    env = RacingEnv("configs/env.yaml")
    env.lap_target = 2.0

    assert not env._lap_complete()
    env.cumulative_forward_progress = 1.5 * env.track.length
    assert not env._lap_complete()
    env.cumulative_forward_progress = 2.0 * env.track.length
    assert env._lap_complete()


def test_env_truncates_when_no_progress_window_fails() -> None:
    env = RacingEnv("configs/env.yaml")
    env.termination_config = replace(
        env.termination_config,
        no_progress_window_steps=1,
        no_progress_min_delta=1.0,
    )
    env.reward_config = replace(env.reward_config, no_progress=3.0)
    env.reset(seed=0)

    _observation, _reward, _terminated, truncated, info = env.step(
        np.array([0.0, 0.0, 0.0], dtype=np.float32)
    )

    assert truncated
    assert info["no_progress_timeout"]
    assert info["reward_terms"]["no_progress"] == -3.0


def test_progress_gate_reward_fires_when_crossing_cumulative_gate() -> None:
    env = RacingEnv("configs/env.yaml")
    env.reward_config = replace(
        env.reward_config,
        progress_gate=2.5,
        progress_gate_spacing=0.0001,
        no_progress=0.0,
    )
    env.reset(seed=0)

    _observation, _reward, _terminated, _truncated, info = env.step(
        np.array([0.0, 1.0, 0.0], dtype=np.float32)
    )

    assert info["reward_terms"]["progress_gate"] >= 2.5


def test_progress_delta_is_capped_by_physical_motion() -> None:
    env = RacingEnv("configs/env_stop_go.yaml")

    progress_delta = env._validated_progress_delta(raw_delta=100.0, physical_delta=1.0)

    assert progress_delta == env.termination_config.max_progress_delta_factor + (
        env.termination_config.max_progress_delta_slack
    )


def test_rgb_array_render_returns_uint8_image() -> None:
    env = RacingEnv("configs/env.yaml", render_mode="rgb_array")
    env.reset(seed=0)
    try:
        frame = env.render()
    except Exception as exc:  # pragma: no cover - headless CI without OpenGL
        env.close()
        pytest.skip(f"MuJoCo offscreen render unavailable: {exc}")
    assert frame is not None
    assert frame.dtype == np.uint8
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    env.close()
