from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from racesim.env.track import ClosedTrack
from racesim.utils.geometry import wrap_angle


@dataclass(frozen=True)
class ControlConfig:
    max_drive_force: float = 280.0
    max_brake_force: float = 360.0
    max_yaw_torque: float = 85.0
    lateral_damping: float = 90.0
    linear_drag: float = 0.9


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 1.0
    lateral_error: float = 0.05
    heading_error: float = 0.02
    off_track: float = 25.0


class RacingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Minimal MuJoCo racing environment.

    This first environment intentionally uses direct chassis force and yaw
    torque. It gives us a stable control loop and reward surface before we make
    the vehicle model more physical.
    """

    metadata = {"render_modes": []}

    def __init__(self, config_path: str | Path = "configs/env.yaml") -> None:
        super().__init__()
        self.config_path = Path(config_path)
        self.config = self._load_config(self.config_path)
        root = (
            self.config_path.parent.parent
            if self.config_path.parent.name == "configs"
            else Path(".")
        )

        track_path = self._resolve_path(self.config["track"], root)
        model_path = self._resolve_path(self.config["model"]["xml"], root)

        self.track = ClosedTrack.from_config(track_path)
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.car_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "car")
        self.root_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "root")
        self.root_qpos_adr = int(self.model.jnt_qposadr[self.root_joint_id])
        self.root_dof_adr = int(self.model.jnt_dofadr[self.root_joint_id])

        sim_config = self.config.get("simulation", {})
        self.frame_skip = int(sim_config.get("frame_skip", 1))
        self.max_episode_steps = int(sim_config.get("max_episode_steps", 3000))
        self.initial_speed = float(sim_config.get("initial_speed", 0.0))

        self.control = ControlConfig(**self.config.get("control", {}))
        self.reward_config = RewardConfig(**self.config.get("reward", {}))
        self.off_track_margin = float(
            self.config.get("termination", {}).get("off_track_margin", 0.0)
        )

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(8,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.previous_progress = 0.0
        self.cumulative_forward_progress = 0.0

    @staticmethod
    def _load_config(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as file:
            return yaml.safe_load(file)

    @staticmethod
    def _resolve_path(path: str | Path, root: Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (root / candidate).resolve()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        options = options or {}
        start_progress = float(options.get("progress", 0.0))

        point, tangent, _normal = self.track.sample_at(start_progress)
        heading = float(np.arctan2(tangent[1], tangent[0]))
        quat = yaw_to_quat(heading)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = [point[0], point[1], 0.32]
        self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = quat
        self.data.qvel[self.root_dof_adr : self.root_dof_adr + 3] = self.initial_speed * np.array(
            [tangent[0], tangent[1], 0.0]
        )
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        projection = self.track.project(point, heading=heading)
        self.previous_progress = projection.progress
        self.cumulative_forward_progress = 0.0
        self.step_count = 0

        observation = self._observation(projection)
        return observation, self._info(projection, reward_terms={})

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=float),
            self.action_space.low,
            self.action_space.high,
        )

        previous_progress = self.previous_progress
        for _ in range(self.frame_skip):
            self._apply_action(action)
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[:] = 0.0

        self.step_count += 1
        pose = self._pose()
        projection = self.track.project(pose[:2], heading=pose[2])
        progress_delta = self.track.progress_delta(previous_progress, projection.progress)
        self.previous_progress = projection.progress
        self.cumulative_forward_progress += max(progress_delta, 0.0)

        off_track = self.track.is_off_track(pose[:2], margin=self.off_track_margin)
        lap_complete = self.cumulative_forward_progress >= self.track.length
        reward_terms = {
            "progress": self.reward_config.progress * progress_delta,
            "lateral_error": -self.reward_config.lateral_error * abs(projection.lateral_error),
            "heading_error": -self.reward_config.heading_error
            * abs(projection.heading_error or 0.0),
            "off_track": -self.reward_config.off_track if off_track else 0.0,
        }
        reward = float(sum(reward_terms.values()))

        terminated = off_track or lap_complete
        truncated = self.step_count >= self.max_episode_steps
        observation = self._observation(projection)
        return observation, reward, terminated, truncated, self._info(projection, reward_terms)

    def _apply_action(self, action: np.ndarray) -> None:
        steering, throttle, brake = action
        x, y, yaw = self._pose()
        del x, y

        velocity = self._linear_velocity()
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0])

        forward_speed = float(np.dot(velocity, forward))
        lateral_speed = float(np.dot(velocity, lateral))
        drive_force = throttle * self.control.max_drive_force
        brake_force = brake * self.control.max_brake_force * np.sign(forward_speed)
        drag_force = self.control.linear_drag * forward_speed * abs(forward_speed)
        lateral_grip_force = self.control.lateral_damping * lateral_speed

        force = (
            (drive_force - brake_force - drag_force) * forward
            - lateral_grip_force * lateral
        )
        yaw_torque = steering * self.control.max_yaw_torque * max(abs(forward_speed), 0.5)

        self.data.xfrc_applied[self.car_body_id, 0:3] = force
        self.data.xfrc_applied[self.car_body_id, 5] = yaw_torque

    def _observation(self, projection: Any) -> np.ndarray:
        velocity = self._linear_velocity()
        pose = self._pose()
        yaw = pose[2]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        longitudinal_speed = float(np.dot(velocity, forward))
        lateral_speed = float(np.dot(velocity, lateral))
        yaw_rate = self._yaw_rate()
        speed = float(np.linalg.norm(velocity[:2]))

        return np.array(
            [
                projection.progress / self.track.length,
                projection.lateral_error / self.track.half_width,
                (projection.heading_error or 0.0) / np.pi,
                longitudinal_speed,
                lateral_speed,
                yaw_rate,
                speed,
                float(abs(projection.lateral_error) > self.track.half_width),
            ],
            dtype=np.float32,
        )

    def _info(self, projection: Any, reward_terms: dict[str, float]) -> dict[str, Any]:
        return {
            "progress": projection.progress,
            "lap_fraction": projection.progress / self.track.length,
            "cumulative_lap_fraction": self.cumulative_forward_progress / self.track.length,
            "lateral_error": projection.lateral_error,
            "heading_error": projection.heading_error,
            "off_track": abs(projection.lateral_error) > self.track.half_width,
            "lap_complete": self.cumulative_forward_progress >= self.track.length,
            "reward_terms": reward_terms,
        }

    def _pose(self) -> np.ndarray:
        qpos = self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 7]
        return np.array([qpos[0], qpos[1], quat_to_yaw(qpos[3:7])], dtype=float)

    def _linear_velocity(self) -> np.ndarray:
        return self.data.qvel[self.root_dof_adr : self.root_dof_adr + 3].copy()

    def _yaw_rate(self) -> float:
        return float(self.data.qvel[self.root_dof_adr + 5])


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(wrap_angle(np.arctan2(siny_cosp, cosy_cosp)))
