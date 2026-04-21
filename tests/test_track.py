from __future__ import annotations

import math

import numpy as np
import pytest

from racesim.env.track import ClosedTrack, catmull_rom_closed_centerline


def square_track() -> ClosedTrack:
    return ClosedTrack(
        centerline=np.array(
            [
                [0.0, 0.0],
                [10.0, 0.0],
                [10.0, 10.0],
                [0.0, 10.0],
            ]
        ),
        width=4.0,
        name="square",
    )


def test_projection_on_first_straight_has_expected_progress_and_lateral_error() -> None:
    track = square_track()

    projection = track.project(np.array([3.0, 1.5]), heading=0.0)

    assert projection.segment_index == 0
    assert projection.progress == 3.0
    assert projection.lateral_error == 1.5
    assert projection.heading_error == 0.0


def test_heading_error_wraps_to_small_signed_angle() -> None:
    track = square_track()

    projection = track.project(np.array([5.0, -0.2]), heading=2.0 * math.pi - 0.1)

    assert projection.heading_error == pytest.approx(-0.1)


def test_progress_delta_handles_lap_wrap() -> None:
    track = square_track()

    delta = track.progress_delta(previous_progress=39.0, current_progress=1.0)

    assert delta == 2.0


def test_progress_delta_is_negative_when_moving_backward() -> None:
    track = square_track()

    delta = track.progress_delta(previous_progress=5.0, current_progress=3.0)

    assert delta == -2.0


def test_off_track_detection_uses_half_width() -> None:
    track = square_track()

    assert not track.is_off_track(np.array([5.0, 1.9]))
    assert track.is_off_track(np.array([5.0, 2.1]))


def test_sample_at_wraps_progress() -> None:
    track = square_track()

    point, tangent, normal = track.sample_at(41.0)

    np.testing.assert_allclose(point, np.array([1.0, 0.0]))
    np.testing.assert_allclose(tangent, np.array([1.0, 0.0]))
    np.testing.assert_allclose(normal, np.array([0.0, 1.0]))


def test_catmull_rom_closed_centerline_samples_smooth_closed_track() -> None:
    centerline = catmull_rom_closed_centerline(
        np.array(
            [
                [10.0, 0.0],
                [0.0, 8.0],
                [-10.0, 0.0],
                [0.0, -8.0],
            ]
        ),
        samples_per_segment=8,
    )
    track = ClosedTrack(centerline, width=4.0)

    assert centerline.shape == (32, 2)
    assert track.length > 40.0
