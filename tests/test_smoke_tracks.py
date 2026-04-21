from __future__ import annotations

from pathlib import Path

from racesim.eval.smoke_tracks import smoke_tracks


def test_smoke_tracks_can_run_one_short_step() -> None:
    rows = smoke_tracks(
        catalog_path=Path("configs/track_catalog.yaml"),
        controller_name="open_loop",
        steps=1,
        lap_target=1.0,
    )

    assert len(rows) >= 10
    assert rows[0]["steps"] == 1


def test_smoke_tracks_step_budget_overrides_env_truncation() -> None:
    rows = smoke_tracks(
        catalog_path=Path("configs/track_catalog.yaml"),
        controller_name="open_loop",
        steps=2,
        lap_target=5.0,
    )

    assert rows[0]["steps"] == 2
