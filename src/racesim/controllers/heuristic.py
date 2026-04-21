from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from racesim.env.track import ClosedTrack
from racesim.utils.geometry import wrap_angle


@dataclass
class HeuristicController:
    """PID-ish baseline controller for the simplified racing environment."""

    track: ClosedTrack
    max_speed: float = 8.0
    min_speed: float = 2.5
    curvature_gain: float = 100.0
    lateral_gain: float = 1.20
    heading_gain: float = 2.80
    speed_gain: float = 0.24

    def act(self, observation: np.ndarray, info: dict) -> np.ndarray:
        lateral_error_norm = float(observation[1])
        heading_error = float(observation[2] * np.pi)
        speed = float(observation[6])
        progress = float(info["progress"])

        target_speed = self.target_speed(progress)
        steering = -self.lateral_gain * lateral_error_norm - self.heading_gain * heading_error
        speed_error = target_speed - speed

        throttle = np.clip(self.speed_gain * speed_error, 0.0, 1.0)
        brake = np.clip(-self.speed_gain * speed_error, 0.0, 1.0)

        return np.array([np.clip(steering, -1.0, 1.0), throttle, brake], dtype=np.float32)

    def target_speed(self, progress: float) -> float:
        curvature = self._lookahead_curvature(progress)
        return float(
            np.clip(
                self.max_speed / (1.0 + self.curvature_gain * curvature),
                self.min_speed,
                self.max_speed,
            )
        )

    def _lookahead_curvature(self, progress: float) -> float:
        _point, tangent, _normal = self.track.sample_at(progress)
        heading = np.arctan2(tangent[1], tangent[0])
        curvature_estimates = []

        for lookahead in (8.0, 16.0, 28.0):
            _future_point, future_tangent, _future_normal = self.track.sample_at(
                progress + lookahead
            )
            future_heading = np.arctan2(future_tangent[1], future_tangent[0])
            curvature_estimates.append(abs(wrap_angle(future_heading - heading)) / lookahead)

        return float(max(curvature_estimates))
