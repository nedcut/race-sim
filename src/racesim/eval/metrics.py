from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class EpisodeMetrics:
    episode: int
    seed: int
    steps: int
    sim_time: float
    total_reward: float
    lap_complete: bool
    off_track: bool
    lap_fraction: float
    cumulative_lap_fraction: float
    mean_abs_lateral_error: float = 0.0
    max_abs_lateral_error: float = 0.0
    mean_abs_heading_error: float = 0.0
    mean_speed: float = 0.0
    rms_lateral_error: float = 0.0


def path_error_metrics(
    lateral_errors: list[float],
    heading_errors: list[float],
    speeds: list[float],
) -> dict[str, float]:
    """Aggregate track-keeping / speed stats for a finished episode."""
    if not lateral_errors:
        return {
            "mean_abs_lateral_error": 0.0,
            "max_abs_lateral_error": 0.0,
            "mean_abs_heading_error": 0.0,
            "mean_speed": 0.0,
            "rms_lateral_error": 0.0,
        }
    abs_lat = [abs(value) for value in lateral_errors]
    abs_head = [abs(value) for value in heading_errors] if heading_errors else [0.0]
    return {
        "mean_abs_lateral_error": float(mean(abs_lat)),
        "max_abs_lateral_error": float(max(abs_lat)),
        "mean_abs_heading_error": float(mean(abs_head)),
        "mean_speed": float(mean(speeds)) if speeds else 0.0,
        "rms_lateral_error": float(sqrt(mean(value * value for value in abs_lat))),
    }


def summarize_episodes(episodes: list[EpisodeMetrics]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("Cannot summarize an empty episode list.")

    completed = [episode for episode in episodes if episode.lap_complete]
    return {
        "episodes": len(episodes),
        "completion_rate": len(completed) / len(episodes),
        "off_track_rate": sum(episode.off_track for episode in episodes) / len(episodes),
        "mean_reward": mean(episode.total_reward for episode in episodes),
        "mean_steps": mean(episode.steps for episode in episodes),
        "mean_sim_time": mean(episode.sim_time for episode in episodes),
        "mean_completed_lap_time": mean(episode.sim_time for episode in completed)
        if completed
        else None,
        "best_completed_lap_time": min((episode.sim_time for episode in completed), default=None),
        "mean_abs_lateral_error": mean(episode.mean_abs_lateral_error for episode in episodes),
        "mean_max_abs_lateral_error": mean(episode.max_abs_lateral_error for episode in episodes),
        "mean_abs_heading_error": mean(episode.mean_abs_heading_error for episode in episodes),
        "mean_speed": mean(episode.mean_speed for episode in episodes),
        "mean_rms_lateral_error": mean(episode.rms_lateral_error for episode in episodes),
    }
