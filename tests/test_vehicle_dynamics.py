from __future__ import annotations

from dataclasses import replace

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


def test_load_transfer_moves_normal_load_rearward_under_acceleration() -> None:
    env = RacingEnv("configs/env.yaml")

    static_front, static_rear = env._axle_normal_loads(0.0, speed=0.0)
    accelerating_front, accelerating_rear = env._axle_normal_loads(5.0, speed=0.0)
    braking_front, braking_rear = env._axle_normal_loads(-5.0, speed=0.0)

    assert accelerating_front < static_front
    assert accelerating_rear > static_rear
    assert braking_front > static_front
    assert braking_rear < static_rear


def test_aero_downforce_increases_available_tire_force() -> None:
    env = RacingEnv("configs/env.yaml")
    env.tire_model = replace(env.tire_model, aero_downforce_coefficient=12.0)

    slow_limits = env._axle_tire_limits(env._axle_normal_loads(0.0, speed=0.0))
    fast_limits = env._axle_tire_limits(env._axle_normal_loads(0.0, speed=10.0))

    assert fast_limits[0] > slow_limits[0]
    assert fast_limits[1] > slow_limits[1]


def test_combined_tire_force_saturates_to_limit() -> None:
    env = RacingEnv("configs/env.yaml")

    longitudinal, lateral, usage = env._limit_combined_tire_force(900.0, 1200.0, 1000.0)

    assert usage == 1.0
    assert np.hypot(longitudinal, lateral) <= 1000.0 + 1e-9


def test_tire_curve_is_monotonic_until_saturation() -> None:
    env = RacingEnv("configs/env.yaml")
    force_limit = 2000.0

    low = abs(env._tire_forces(0.0, 0.02, 7000.0, force_limit).lateral)
    medium = abs(env._tire_forces(0.0, 0.08, 7000.0, force_limit).lateral)
    high = abs(env._tire_forces(0.0, 0.30, 7000.0, force_limit).lateral)

    assert low < medium < high <= force_limit


def test_braking_consumes_lateral_tire_capacity() -> None:
    env = RacingEnv("configs/env.yaml")
    force_limit = 2000.0

    corner_only = env._tire_forces(0.0, 0.20, 7000.0, force_limit)
    braking_corner = env._tire_forces(-1800.0, 0.20, 7000.0, force_limit)

    assert abs(braking_corner.lateral) < abs(corner_only.lateral)
    assert braking_corner.usage >= corner_only.usage


def test_step_reports_separate_bounded_tire_usage() -> None:
    env = RacingEnv("configs/env.yaml")
    env.reset(seed=0, options={"initial_speed": 8.0})

    _observation, _reward, _terminated, _truncated, info = env.step(
        np.array([1.0, 1.0, 0.0], dtype=np.float32)
    )

    assert set(info["tire_usage"]) == {"front", "rear"}
    assert set(info["slip_angles"]) == {"front", "rear"}
    assert 0.0 <= info["tire_usage"]["front"] <= 1.0
    assert 0.0 <= info["tire_usage"]["rear"] <= 1.0
    assert info["normal_loads"]["front"] > 0.0
    assert info["normal_loads"]["rear"] > 0.0
    assert "yaw_torque" in info
    assert "understeer_score" in info


def test_off_track_info_matches_termination_margin() -> None:
    env = RacingEnv("configs/env.yaml")
    env.off_track_margin = 2.0
    env.reset(seed=0)
    # Far beyond half-width but still inside the termination margin buffer.
    far_lateral = env.track.half_width + 1.0
    assert not env.track.is_off_track(
        env.track.centerline[0] + env.track.normals[0] * far_lateral,
        margin=env.off_track_margin,
    )


def test_info_off_track_uses_margin() -> None:
    env = RacingEnv("configs/env.yaml")
    env.max_episode_steps = 20
    env.lap_target = float("inf")
    env.off_track_margin = 1e6
    env.reset(seed=0, options={"initial_speed": 8.0})

    for _ in range(15):
        _observation, _reward, terminated, _truncated, info = env.step(
            np.array([1.0, 0.5, 0.0], dtype=np.float32)
        )

    assert info["off_track"] is False
    assert terminated is False


def test_sustained_steering_generates_yaw_without_artificial_speed_collapse() -> None:
    env = RacingEnv("configs/env.yaml")
    env.max_episode_steps = 80
    env.lap_target = float("inf")
    env.off_track_margin = 1e6
    env.reset(seed=0, options={"initial_speed": 8.0})

    for _ in range(60):
        _observation, _reward, _terminated, _truncated, info = env.step(
            np.array([1.0, 0.22, 0.0], dtype=np.float32)
        )

    assert info["speed"] > 7.0
    assert info["yaw_rate"] > 0.3
