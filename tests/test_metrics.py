from __future__ import annotations

from racesim.eval.metrics import EpisodeMetrics, summarize_episodes


def test_summarize_episodes_reports_completion_and_best_lap_time() -> None:
    episodes = [
        EpisodeMetrics(0, 0, 100, 8.0, 10.0, True, False, 0.0, 1.0),
        EpisodeMetrics(1, 1, 120, 9.6, 5.0, False, True, 0.5, 0.5),
    ]

    summary = summarize_episodes(episodes)

    assert summary["completion_rate"] == 0.5
    assert summary["off_track_rate"] == 0.5
    assert summary["best_completed_lap_time"] == 8.0
