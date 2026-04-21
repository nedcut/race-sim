from __future__ import annotations

import pytest

from racesim.eval.plots import get_trajectory, progress_array
from racesim.eval.sweep_vehicle import sweep_cases


def test_get_trajectory_requires_recorded_trajectory() -> None:
    result = {"episodes": [{"episode": 0}]}

    with pytest.raises(ValueError):
        get_trajectory(result, 0)


def test_progress_array_reads_cumulative_lap_fraction() -> None:
    progress = progress_array(
        [
            {"cumulative_lap_fraction": 0.1},
            {"cumulative_lap_fraction": 0.2},
        ]
    )

    assert progress.tolist() == [0.1, 0.2]


def test_sweep_cases_cover_drivetrains_and_grip_scales() -> None:
    cases = sweep_cases()

    assert len(cases) == 9
    assert {"drivetrain": "rwd", "grip_scale": 1.0} in cases
    assert {"drivetrain": "awd", "grip_scale": 1.25} in cases
