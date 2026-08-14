from __future__ import annotations

from racesim.paths import default_track_catalog
from racesim.scripts.validate_tracks import catalog_track_paths, validate_track_paths


def test_catalog_track_paths_reads_all_tracks() -> None:
    paths = catalog_track_paths(default_track_catalog())
    names = {path.name for path in paths}

    assert len(paths) >= 14
    assert "oval.yaml" in names
    assert "grand_prix.yaml" in names
    assert "street_circuit.yaml" in names
    assert "kartplex.yaml" in names
    assert "endurance.yaml" in names
    assert all(path.is_file() for path in paths)


def test_validate_track_paths_reports_clean_catalog_tracks() -> None:
    paths = catalog_track_paths(default_track_catalog())
    results = validate_track_paths(paths)

    assert all(not issues for _path, issues in results)
