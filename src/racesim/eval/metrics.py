from __future__ import annotations

from dataclasses import dataclass
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
    }
