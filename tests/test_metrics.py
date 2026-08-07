from __future__ import annotations

from racesim.eval.metrics import EpisodeMetrics, summarize_episodes


def test_summarize_episodes_reports_completion_and_best_lap_time() -> None:
    episodes = [
        EpisodeMetrics(
            0,
            0,
            100,
            8.0,
            10.0,
            True,
            False,
            0.0,
            1.0,
            mean_abs_lateral_error=0.4,
            max_abs_lateral_error=0.8,
            mean_abs_heading_error=0.1,
            mean_speed=5.0,
            rms_lateral_error=0.5,
        ),
        EpisodeMetrics(
            1,
            1,
            120,
            9.6,
            5.0,
            False,
            True,
            0.5,
            0.5,
            mean_abs_lateral_error=0.6,
            max_abs_lateral_error=1.2,
            mean_abs_heading_error=0.3,
            mean_speed=4.0,
            rms_lateral_error=0.7,
        ),
    ]

    summary = summarize_episodes(episodes)

    assert summary["completion_rate"] == 0.5
    assert summary["off_track_rate"] == 0.5
    assert summary["best_completed_lap_time"] == 8.0
    assert summary["mean_abs_lateral_error"] == 0.5
    assert summary["mean_max_abs_lateral_error"] == 1.0
    assert summary["mean_speed"] == 4.5


def test_path_error_metrics_rms() -> None:
    from racesim.eval.metrics import path_error_metrics

    metrics = path_error_metrics([3.0, -4.0], [0.0, 0.0], [1.0, 3.0])
    assert metrics["mean_abs_lateral_error"] == 3.5
    assert metrics["max_abs_lateral_error"] == 4.0
    assert metrics["mean_speed"] == 2.0
    assert abs(metrics["rms_lateral_error"] - 3.5355339059) < 1e-6
