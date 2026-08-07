"""Shared telemetry flattening for rollout eval and open-loop benchmarks."""

from __future__ import annotations

from typing import Any

import numpy as np


def flatten_info_telemetry(info: dict[str, Any]) -> dict[str, float | bool]:
    """Extract a flat telemetry dict from a RacingEnv info payload."""
    position = np.asarray(info["position"], dtype=float)
    slip_angles = info.get("slip_angles", {})
    tire_forces = info.get("tire_forces", {})
    tire_usage = info.get("tire_usage", {})
    normal_loads = info.get("normal_loads", {})
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "heading": float(info["heading"]),
        "progress": float(info["progress"]),
        "lap_fraction": float(info["lap_fraction"]),
        "cumulative_lap_fraction": float(info["cumulative_lap_fraction"]),
        "progress_delta": float(info.get("progress_delta", 0.0)),
        "raw_progress_delta": float(info.get("raw_progress_delta", 0.0)),
        "progress_delta_clipped": bool(info.get("progress_delta_clipped", False)),
        "speed": float(info["speed"]),
        "longitudinal_speed": float(info["longitudinal_speed"]),
        "lateral_speed": float(info["lateral_speed"]),
        "yaw_rate": float(info["yaw_rate"]),
        "lateral_error": float(info["lateral_error"]),
        "heading_error": float(info["heading_error"] or 0.0),
        "grip_scale": float(info["grip_scale"]),
        "target_speed": float(info.get("target_speed", 0.0)),
        "steering_angle": float(info.get("steering_angle", 0.0)),
        "front_slip_angle": float(slip_angles.get("front", 0.0)),
        "rear_slip_angle": float(slip_angles.get("rear", 0.0)),
        "front_longitudinal_force": float(tire_forces.get("front_longitudinal", 0.0)),
        "rear_longitudinal_force": float(tire_forces.get("rear_longitudinal", 0.0)),
        "front_lateral_force": float(tire_forces.get("front_lateral", 0.0)),
        "rear_lateral_force": float(tire_forces.get("rear_lateral", 0.0)),
        "front_tire_usage": float(tire_usage.get("front", 0.0)),
        "rear_tire_usage": float(tire_usage.get("rear", 0.0)),
        "front_normal_load": float(normal_loads.get("front", 0.0)),
        "rear_normal_load": float(normal_loads.get("rear", 0.0)),
        "yaw_torque": float(info.get("yaw_torque", 0.0)),
        "understeer_score": float(info.get("understeer_score", 0.0)),
        "off_track": bool(info["off_track"]),
        "lap_complete": bool(info["lap_complete"]),
    }


def telemetry_row(
    step: int,
    sim_time: float,
    action: np.ndarray,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> dict[str, Any]:
    """Build a full rollout trajectory row from step outputs."""
    smoothed_action = np.asarray(info["smoothed_action"], dtype=float)
    row = flatten_info_telemetry(info)
    row.update(
        {
            "step": step,
            "time": sim_time,
            "steering": float(action[0]),
            "throttle": float(action[1]),
            "brake": float(action[2]),
            "smoothed_steering": float(smoothed_action[0]),
            "smoothed_throttle": float(smoothed_action[1]),
            "smoothed_brake": float(smoothed_action[2]),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
        }
    )
    return row
