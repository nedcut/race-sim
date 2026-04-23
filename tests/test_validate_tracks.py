from __future__ import annotations

from pathlib import Path

from racesim.scripts.validate_tracks import catalog_track_paths, validate_track_paths


def test_catalog_track_paths_reads_all_tracks() -> None:
    paths = catalog_track_paths(Path("configs/track_catalog.yaml"))

    assert len(paths) >= 14
    assert Path("configs/tracks/oval.yaml") in paths
    assert Path("configs/tracks/grand_prix.yaml") in paths
    assert Path("configs/tracks/street_circuit.yaml") in paths
    assert Path("configs/tracks/kartplex.yaml") in paths
    assert Path("configs/tracks/endurance.yaml") in paths


def test_validate_track_paths_reports_clean_catalog_tracks() -> None:
    paths = catalog_track_paths(Path("configs/track_catalog.yaml"))
    results = validate_track_paths(paths)

    assert all(not issues for _path, issues in results)
