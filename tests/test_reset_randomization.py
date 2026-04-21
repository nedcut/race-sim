from __future__ import annotations

import numpy as np

from racesim.controllers.factory import make_controller
from racesim.env.racing_env import RacingEnv


def test_reset_randomization_is_deterministic_with_seed() -> None:
    env = RacingEnv("configs/env.yaml")

    _obs_a, info_a = env.reset(seed=123, options={"randomize": True})
    _obs_b, info_b = env.reset(seed=123, options={"randomize": True})

    np.testing.assert_allclose(info_a["position"], info_b["position"])
    assert info_a["heading"] == info_b["heading"]


def test_reset_options_can_place_car_off_center_with_heading_error() -> None:
    env = RacingEnv("configs/env.yaml")

    _observation, info = env.reset(
        seed=0,
        options={
            "progress": 40.0,
            "lateral_offset": 1.25,
            "heading_error": 0.2,
            "initial_speed": 3.0,
        },
    )

    assert info["progress"] > 35.0
    assert abs(info["lateral_error"]) > 1.0
    assert abs(info["heading_error"]) > 0.15


def test_centerline_controller_recovers_from_small_pose_perturbation() -> None:
    env = RacingEnv("configs/env.yaml")
    controller = make_controller("centerline", env.track)
    observation, info = env.reset(
        seed=0,
        options={
            "progress": 25.0,
            "lateral_offset": 1.0,
            "heading_error": 0.15,
            "initial_speed": 2.5,
        },
    )
    for _ in range(80):
        action = controller.act(observation, info)
        observation, _reward, terminated, _truncated, info = env.step(action)
        if terminated:
            break

    assert not info["off_track"]
    assert abs(info["lateral_error"]) < env.track.half_width
