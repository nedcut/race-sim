from __future__ import annotations

from pathlib import Path

import numpy as np

from racesim.scripts.render_rollout import car_outline, load_catalog_env_configs


def test_car_outline_has_expected_front_point() -> None:
    outline = car_outline(np.array([1.0, 2.0]), heading=0.0, length=2.0, width=1.0)

    np.testing.assert_allclose(outline[0], np.array([2.0, 2.0]))
    assert outline.shape == (5, 2)


def test_load_catalog_env_configs_reads_track_env_paths(tmp_path: Path) -> None:
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        """
tracks:
  oval:
    env: configs/env.yaml
    track: configs/tracks/oval.yaml
  metadata_only:
    track: configs/tracks/ignored.yaml
""",
        encoding="utf-8",
    )

    assert load_catalog_env_configs(catalog) == {"oval": Path("configs/env.yaml")}
