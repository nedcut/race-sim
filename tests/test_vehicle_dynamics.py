from __future__ import annotations

import numpy as np

from racesim.env.racing_env import RacingEnv


def test_positive_steering_turns_car_left() -> None:
    env = RacingEnv("configs/env.yaml")
    _observation, start_info = env.reset(seed=0)

    for _ in range(12):
        _observation, _reward, _terminated, _truncated, info = env.step(
            np.array([1.0, 0.25, 0.0], dtype=np.float32)
        )

    assert info["heading"] > start_info["heading"]
    assert info["yaw_rate"] > 0.0


def test_action_smoothing_limits_single_step_change() -> None:
    env = RacingEnv("configs/env.yaml")
    env.reset(seed=0)

    env.step(np.array([1.0, 1.0, 1.0], dtype=np.float32))

    np.testing.assert_allclose(
        env.smoothed_action,
        np.array([0.4, 0.4, 0.64]),
        atol=1e-12,
    )


def test_configured_drivetrain_is_rwd() -> None:
    env = RacingEnv("configs/env.yaml")

    assert env._drive_split() == (0.0, 1.0)
