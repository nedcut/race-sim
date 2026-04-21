from __future__ import annotations

from pathlib import Path

import mujoco


def test_world_contains_visual_track_and_cameras() -> None:
    model_path = Path("assets/mjcf/world.xml")
    model = mujoco.MjModel.from_xml_path(str(model_path))

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "track_surface") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "follow") >= 0
    assert model.ngeom > 100
