from __future__ import annotations

from pathlib import Path

from racesim.eval.smoke_tracks import smoke_tracks


def test_smoke_tracks_can_run_one_short_step() -> None:
    rows = smoke_tracks(
        catalog_path=Path("configs/track_catalog.yaml"),
        controller_name="open_loop",
        steps=1,
    )

    assert len(rows) >= 10
    assert rows[0]["steps"] == 1
