from __future__ import annotations

import argparse
from unittest.mock import Mock

import mujoco
import pytest

from racesim.env.racing_env import RacingEnv
from racesim.scripts.keyboard_drive import apply_viewer_camera, maybe_reexec_with_mjpython


def test_no_reexec_flag_returns_without_relaunching() -> None:
    args = argparse.Namespace(no_reexec=True)

    assert maybe_reexec_with_mjpython(args) is None


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
