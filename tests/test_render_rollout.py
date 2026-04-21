from __future__ import annotations

import numpy as np

from racesim.scripts.render_rollout import car_outline


def test_car_outline_has_expected_front_point() -> None:
    outline = car_outline(np.array([1.0, 2.0]), heading=0.0, length=2.0, width=1.0)

    np.testing.assert_allclose(outline[0], np.array([2.0, 2.0]))
    assert outline.shape == (5, 2)
