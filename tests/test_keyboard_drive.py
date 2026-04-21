from __future__ import annotations

import argparse
from unittest.mock import Mock

import mujoco

from racesim.env.racing_env import RacingEnv
from racesim.scripts.keyboard_drive import apply_viewer_camera, maybe_reexec_with_mjpython


def test_no_reexec_flag_returns_without_relaunching() -> None:
    args = argparse.Namespace(no_reexec=True)

    assert maybe_reexec_with_mjpython(args) is None


def test_apply_viewer_camera_selects_fixed_named_camera() -> None:
    env = RacingEnv("configs/env.yaml")
    viewer = Mock()
    viewer.cam = Mock()

    apply_viewer_camera(viewer, env, "follow")

    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED
    assert viewer.cam.fixedcamid >= 0


def test_apply_viewer_camera_can_reset_free_camera() -> None:
    env = RacingEnv("configs/env.yaml")
    viewer = Mock()
    viewer.cam = Mock()
    viewer.cam.lookat = [0.0, 0.0, 0.0]

    apply_viewer_camera(viewer, env, "free")

    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert viewer.cam.distance == 95.0
