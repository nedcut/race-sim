from __future__ import annotations

from pathlib import Path

import mujoco
import yaml

from racesim.env.racing_env import RacingEnv
from racesim.env.track import ClosedTrack


def catalog() -> dict:
    return yaml.safe_load(Path("configs/track_catalog.yaml").read_text(encoding="utf-8"))[
        "tracks"
    ]


def test_track_catalog_tracks_have_sane_geometry() -> None:
    for name, entry in catalog().items():
        track = ClosedTrack.from_config(entry["track"])

        assert track.name == name
        assert track.length > 80.0
        assert track.width > 0.0


def test_track_catalog_envs_and_worlds_load() -> None:
    for entry in catalog().values():
        env = RacingEnv(entry["env"])

        assert env.model.ngeom > 10
        assert mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "track_surface") >= 0
