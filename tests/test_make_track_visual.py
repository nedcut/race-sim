from __future__ import annotations

import mujoco

from racesim.env.track import ClosedTrack
from racesim.scripts.make_track_visual import track_visual_xml


def test_generated_track_visual_xml_loads_in_mujoco(tmp_path) -> None:
    track = ClosedTrack.from_config("configs/tracks/technical.yaml")
    path = tmp_path / "track.xml"
    path.write_text(track_visual_xml(track, "test_track"), encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(path))

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "track_surface") >= 0
