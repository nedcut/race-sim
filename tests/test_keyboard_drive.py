from __future__ import annotations

import argparse
from unittest.mock import Mock

import mujoco
import numpy as np
import pytest

from racesim.env.racing_env import RacingEnv
from racesim.scripts.keyboard_drive import (
    PolicyDriver,
    apply_viewer_camera,
    make_autopilot,
    maybe_reexec_with_mjpython,
)


def test_no_reexec_flag_returns_without_relaunching() -> None:
    args = argparse.Namespace(no_reexec=True)

    assert maybe_reexec_with_mjpython(args) is None


def test_make_autopilot_uses_builtin_controller_without_policy_model() -> None:
    env = RacingEnv("configs/env.yaml")
    args = argparse.Namespace(policy_model=None, autopilot="racing_line")

    controller, name = make_autopilot(env, args)

    assert name == "racing_line"
    assert hasattr(controller, "act")


def test_policy_driver_uses_model_prediction() -> None:
    model = Mock()
    model.predict.return_value = ([0.1, 0.2, 0.0], None)
    driver = PolicyDriver(model, deterministic=True)

    action = driver.act(np.zeros(3, dtype=np.float32), {})

    np.testing.assert_allclose(action, np.array([0.1, 0.2, 0.0], dtype=np.float32))
    model.predict.assert_called_once()


def test_apply_viewer_camera_selects_fixed_named_camera() -> None:
    env = RacingEnv("configs/env.yaml")
    viewer = Mock()
    viewer.cam = Mock()

    apply_viewer_camera(viewer, env, "fixed")

    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert viewer.cam.fixedcamid >= 0


def test_apply_viewer_camera_can_use_dynamic_chase_camera() -> None:
    env = RacingEnv("configs/env.yaml")
    env.reset(seed=0)
    viewer = Mock()
    viewer.cam = Mock()
    viewer.cam.lookat = [0.0, 0.0, 0.0]

    apply_viewer_camera(viewer, env, "chase")

    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert viewer.cam.distance == 8.0
    assert viewer.cam.azimuth == pytest.approx(-90.0)
    assert viewer.cam.elevation == -18.0


def test_apply_viewer_camera_can_reset_free_camera() -> None:
    env = RacingEnv("configs/env.yaml")
    viewer = Mock()
    viewer.cam = Mock()
    viewer.cam.lookat = [0.0, 0.0, 0.0]

    apply_viewer_camera(viewer, env, "free")

    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert viewer.cam.distance == 95.0
