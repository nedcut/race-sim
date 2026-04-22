from __future__ import annotations

from collections import deque
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
    drivetrain: str = "rwd"
    max_steer_angle: float = 0.55
    steer_rate: float = 3.5
    throttle_rate: float = 4.0
    brake_rate: float = 6.0
    max_drive_force: float = 420.0
    max_brake_force: float = 520.0
    front_cornering_stiffness: float = 1450.0
    rear_cornering_stiffness: float = 1750.0
    max_lateral_force: float = 950.0
    yaw_damping: float = 35.0
    linear_drag: float = 1.4
    rolling_resistance: float = 12.0
    wheelbase: float = 1.35
    center_of_mass_to_front: float = 0.68
    center_of_mass_to_rear: float = 0.67


@dataclass(frozen=True)
class RewardConfig:
    progress: float = 1.0
    time: float = 0.0
    lateral_error: float = 0.05
    heading_error: float = 0.02
    boundary_margin: float = 0.0
    boundary_margin_start: float = 1.0
    progress_gate: float = 0.0
    progress_gate_spacing: float = 0.1
    speed_excess: float = 0.0
    target_speed_max: float = 9.5
    target_speed_min: float = 3.0
    target_speed_curvature_gain: float = 5.0
    no_progress: float = 0.0
    off_track: float = 25.0


@dataclass(frozen=True)
class TerminationConfig:
    off_track_margin: float = 0.0
    no_progress_window_steps: int = 0
    no_progress_min_delta: float = 0.0
    max_progress_delta_factor: float = 2.0
    max_progress_delta_slack: float = 0.5


@dataclass(frozen=True)
class ObservationConfig:
    lookahead_distances: tuple[float, ...] = (6.0, 12.0, 24.0, 40.0)
    curvature_scale: float = 20.0


@dataclass(frozen=True)
class ResetRandomizationConfig:
    enabled: bool = False
    progress: bool = False
    lateral_offset: float = 0.0
    heading_error: float = 0.0
    speed_min: float | None = None
    speed_max: float | None = None
    grip_min: float | None = None
    grip_max: float | None = None


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
        self.lap_target = float(sim_config.get("lap_target", 1.0))

        self.control = ControlConfig(**self.config.get("control", {}))
        self.reward_config = RewardConfig(**self.config.get("reward", {}))
        self.termination_config = TerminationConfig(**self.config.get("termination", {}))
        self.observation_config = ObservationConfig(**self.config.get("observation", {}))
        self.reset_randomization = ResetRandomizationConfig(
            **self.config.get("reset_randomization", {})
        )
        self.off_track_margin = self.termination_config.off_track_margin

        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )
        observation_size = 11 + 2 * len(self.observation_config.lookahead_distances)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size,),
            dtype=np.float32,
        )

        self.step_count = 0
        self.previous_progress = 0.0
        self.cumulative_forward_progress = 0.0
        self.smoothed_action = np.zeros(3, dtype=float)
        self.grip_scale = 1.0
        self.next_progress_gate = 0.0
        self.progress_history: deque[tuple[int, float]] = deque()

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
        randomize = bool(options.get("randomize", self.reset_randomization.enabled))
        start_progress = self._reset_progress(options, randomize)

        point, tangent, normal = self.track.sample_at(start_progress)
        heading = float(np.arctan2(tangent[1], tangent[0]))
        lateral_offset = self._reset_lateral_offset(options, randomize)
        heading += self._reset_heading_error(options, randomize)
        initial_speed = self._reset_initial_speed(options, randomize)
        self.grip_scale = self._reset_grip_scale(options, randomize)
        point = point + lateral_offset * normal
        quat = yaw_to_quat(heading)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 3] = [point[0], point[1], 0.32]
        self.data.qpos[self.root_qpos_adr + 3 : self.root_qpos_adr + 7] = quat
        self.data.qvel[self.root_dof_adr : self.root_dof_adr + 3] = initial_speed * np.array(
            [tangent[0], tangent[1], 0.0]
        )
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        projection = self.track.project(point, heading=heading)
        self.previous_progress = projection.progress
        self.cumulative_forward_progress = 0.0
        self.smoothed_action[:] = 0.0
        self.step_count = 0
        self.next_progress_gate = self._progress_gate_distance()
        self.progress_history.clear()
        self.progress_history.append((0, self.cumulative_forward_progress))

        observation = self._observation(projection)
        return observation, self._info(projection, reward_terms={})

    def _reset_progress(self, options: dict[str, Any], randomize: bool) -> float:
        if "progress" in options:
            return float(options["progress"])
        if randomize and self.reset_randomization.progress:
            return float(self.np_random.uniform(0.0, self.track.length))
        return 0.0

    def _reset_lateral_offset(self, options: dict[str, Any], randomize: bool) -> float:
        if "lateral_offset" in options:
            return float(options["lateral_offset"])
        if randomize and self.reset_randomization.lateral_offset > 0:
            limit = min(self.reset_randomization.lateral_offset, self.track.half_width * 0.85)
            return float(self.np_random.uniform(-limit, limit))
        return 0.0

    def _reset_heading_error(self, options: dict[str, Any], randomize: bool) -> float:
        if "heading_error" in options:
            return float(options["heading_error"])
        if randomize and self.reset_randomization.heading_error > 0:
            limit = self.reset_randomization.heading_error
            return float(self.np_random.uniform(-limit, limit))
        return 0.0

    def _reset_initial_speed(self, options: dict[str, Any], randomize: bool) -> float:
        if "initial_speed" in options:
            return float(options["initial_speed"])
        if (
            randomize
            and self.reset_randomization.speed_min is not None
            and self.reset_randomization.speed_max is not None
        ):
            return float(
                self.np_random.uniform(
                    self.reset_randomization.speed_min,
                    self.reset_randomization.speed_max,
                )
            )
        return self.initial_speed

    def _reset_grip_scale(self, options: dict[str, Any], randomize: bool) -> float:
        if "grip_scale" in options:
            return float(options["grip_scale"])
        if (
            randomize
            and self.reset_randomization.grip_min is not None
            and self.reset_randomization.grip_max is not None
        ):
            return float(
                self.np_random.uniform(
                    self.reset_randomization.grip_min,
                    self.reset_randomization.grip_max,
                )
            )
        return 1.0

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.clip(
            np.asarray(action, dtype=float),
            self.action_space.low,
            self.action_space.high,
        )

        previous_progress = self.previous_progress
        previous_position = self._pose()[:2]
        for _ in range(self.frame_skip):
            self._update_smoothed_action(action)
            self._apply_action()
            mujoco.mj_step(self.model, self.data)
            self.data.xfrc_applied[:] = 0.0

        self.step_count += 1
        pose = self._pose()
        projection = self.track.project(pose[:2], heading=pose[2])
        raw_progress_delta = self.track.progress_delta(previous_progress, projection.progress)
        physical_delta = float(np.linalg.norm(pose[:2] - previous_position))
        progress_delta = self._validated_progress_delta(raw_progress_delta, physical_delta)
        progress_delta_clipped = not np.isclose(progress_delta, raw_progress_delta)
        self.previous_progress = projection.progress
        self.cumulative_forward_progress += max(progress_delta, 0.0)
        gates_crossed = self._consume_progress_gates()

        off_track = self.track.is_off_track(pose[:2], margin=self.off_track_margin)
        lap_complete = self._lap_complete()
        no_progress_timeout = self._no_progress_timeout()
        boundary_margin = self.track.half_width - abs(projection.lateral_error)
        boundary_shortfall = max(self.reward_config.boundary_margin_start - boundary_margin, 0.0)
        speed = float(np.linalg.norm(self._linear_velocity()[:2]))
        target_speed = self._target_speed(projection.progress)
        speed_excess = max(speed - target_speed, 0.0)
        reward_terms = {
            "progress": self.reward_config.progress * progress_delta,
            "time": -self.reward_config.time,
            "lateral_error": -self.reward_config.lateral_error * abs(projection.lateral_error),
            "heading_error": -self.reward_config.heading_error
            * abs(projection.heading_error or 0.0),
            "boundary_margin": -self.reward_config.boundary_margin * boundary_shortfall**2,
            "progress_gate": self.reward_config.progress_gate * gates_crossed,
            "speed_excess": -self.reward_config.speed_excess * speed_excess**2,
            "no_progress": -self.reward_config.no_progress if no_progress_timeout else 0.0,
            "off_track": -self.reward_config.off_track if off_track else 0.0,
        }
        reward = float(sum(reward_terms.values()))

        terminated = off_track or lap_complete
        truncated = self.step_count >= self.max_episode_steps or no_progress_timeout
        observation = self._observation(projection)
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._info(
                projection,
                reward_terms,
                progress_delta=progress_delta,
                raw_progress_delta=raw_progress_delta,
                progress_delta_clipped=progress_delta_clipped,
            ),
        )

    def _update_smoothed_action(self, action: np.ndarray) -> None:
        dt = self.model.opt.timestep
        rates = np.array(
            [
                self.control.steer_rate,
                self.control.throttle_rate,
                self.control.brake_rate,
            ],
            dtype=float,
        )
        delta = np.clip(action - self.smoothed_action, -rates * dt, rates * dt)
        self.smoothed_action += delta

    def _apply_action(self) -> None:
        steering, throttle, brake = self.smoothed_action
        x, y, yaw = self._pose()
        del x, y

        velocity = self._linear_velocity()
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0])

        forward_speed = float(np.dot(velocity, forward))
        lateral_speed = float(np.dot(velocity, lateral))
        yaw_rate = self._yaw_rate()
        steer_angle = steering * self.control.max_steer_angle
        safe_speed = np.copysign(max(abs(forward_speed), 0.75), forward_speed or 1.0)

        lf = self.control.center_of_mass_to_front
        lr = self.control.center_of_mass_to_rear
        front_slip = np.arctan2(lateral_speed + lf * yaw_rate, abs(safe_speed)) - steer_angle
        rear_slip = np.arctan2(lateral_speed - lr * yaw_rate, abs(safe_speed))

        front_lateral_limit = self.control.max_lateral_force * self.grip_scale
        rear_lateral_limit = self.control.max_lateral_force * self.grip_scale
        front_lateral_force = np.clip(
            -self.control.front_cornering_stiffness * self.grip_scale * front_slip,
            -front_lateral_limit,
            front_lateral_limit,
        )
        rear_lateral_force = np.clip(
            -self.control.rear_cornering_stiffness * self.grip_scale * rear_slip,
            -rear_lateral_limit,
            rear_lateral_limit,
        )

        drive_force = throttle * self.control.max_drive_force
        brake_force = brake * self.control.max_brake_force * np.sign(forward_speed)
        drag_force = self.control.linear_drag * forward_speed * abs(forward_speed)
        rolling_force = self.control.rolling_resistance * np.sign(forward_speed)
        longitudinal_force = drive_force - brake_force - drag_force - rolling_force

        front_drive_fraction, rear_drive_fraction = self._drive_split()
        front_forward_force = longitudinal_force * front_drive_fraction
        rear_forward_force = longitudinal_force * rear_drive_fraction

        front_direction = normalize_2d(
            np.cos(steer_angle) * forward + np.sin(steer_angle) * lateral
        )
        front_lateral_direction = normalize_2d(
            -np.sin(steer_angle) * forward + np.cos(steer_angle) * lateral
        )

        front_force = (
            front_forward_force * front_direction
            + front_lateral_force * front_lateral_direction
        )
        rear_force = rear_forward_force * forward + rear_lateral_force * lateral
        force = front_force + rear_force
        yaw_torque = (
            lf * cross_z(forward, front_force)
            - lr * cross_z(forward, rear_force)
            - self.control.yaw_damping * yaw_rate
        )

        self.data.xfrc_applied[self.car_body_id, 0:3] = force
        self.data.xfrc_applied[self.car_body_id, 5] = yaw_torque

    def _drive_split(self) -> tuple[float, float]:
        drivetrain = self.control.drivetrain.lower()
        if drivetrain == "fwd":
            return 1.0, 0.0
        if drivetrain == "awd":
            return 0.5, 0.5
        if drivetrain == "rwd":
            return 0.0, 1.0
        raise ValueError(f"Unsupported drivetrain: {self.control.drivetrain}")

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

        features = [
            projection.progress / self.track.length,
            projection.lateral_error / self.track.half_width,
            (projection.heading_error or 0.0) / np.pi,
            longitudinal_speed,
            lateral_speed,
            yaw_rate,
            speed,
            float(abs(projection.lateral_error) > self.track.half_width),
        ]
        features.extend(self._boundary_margin_features(projection.lateral_error))
        features.append(
            self._target_speed(projection.progress) / self.reward_config.target_speed_max
        )
        features.extend(self._lookahead_features(projection.progress, yaw))
        return np.asarray(features, dtype=np.float32)

    def _boundary_margin_features(self, lateral_error: float) -> list[float]:
        left_margin = self.track.half_width - lateral_error
        right_margin = self.track.half_width + lateral_error
        return [left_margin / self.track.width, right_margin / self.track.width]

    def _lookahead_features(self, progress: float, yaw: float) -> list[float]:
        _point, tangent, _normal = self.track.sample_at(progress)
        current_heading = float(np.arctan2(tangent[1], tangent[0]))
        features: list[float] = []
        for distance in self.observation_config.lookahead_distances:
            _future_point, future_tangent, _future_normal = self.track.sample_at(
                progress + distance
            )
            future_heading = float(np.arctan2(future_tangent[1], future_tangent[0]))
            heading_to_future = wrap_angle(future_heading - yaw) / np.pi
            signed_curvature = (
                wrap_angle(future_heading - current_heading)
                / max(distance, 1e-6)
                * self.observation_config.curvature_scale
            )
            features.extend([heading_to_future, signed_curvature])
        return features

    def _target_speed(self, progress: float) -> float:
        curvatures = self._upcoming_curvatures(progress)
        max_curvature = max((abs(curvature) for curvature in curvatures), default=0.0)
        target_speed = self.reward_config.target_speed_max / (
            1.0 + self.reward_config.target_speed_curvature_gain * max_curvature
        )
        return float(
            np.clip(
                target_speed,
                self.reward_config.target_speed_min,
                self.reward_config.target_speed_max,
            )
        )

    def _upcoming_curvatures(self, progress: float) -> list[float]:
        _point, tangent, _normal = self.track.sample_at(progress)
        current_heading = float(np.arctan2(tangent[1], tangent[0]))
        curvatures: list[float] = []
        for distance in self.observation_config.lookahead_distances:
            _future_point, future_tangent, _future_normal = self.track.sample_at(
                progress + distance
            )
            future_heading = float(np.arctan2(future_tangent[1], future_tangent[0]))
            curvatures.append(wrap_angle(future_heading - current_heading) / max(distance, 1e-6))
        return curvatures

    def _progress_gate_distance(self) -> float:
        spacing = self.reward_config.progress_gate_spacing
        if spacing <= 0.0:
            return float("inf")
        if spacing <= 1.0:
            return spacing * self.track.length
        return spacing

    def _consume_progress_gates(self) -> int:
        if self.reward_config.progress_gate <= 0.0 or not np.isfinite(self.next_progress_gate):
            return 0
        spacing = self._progress_gate_distance()
        gates_crossed = 0
        while self.cumulative_forward_progress >= self.next_progress_gate:
            gates_crossed += 1
            self.next_progress_gate += spacing
        return gates_crossed

    def _no_progress_timeout(self) -> bool:
        window = self.termination_config.no_progress_window_steps
        min_delta = self.termination_config.no_progress_min_delta
        if window <= 0 or min_delta <= 0.0:
            return False

        self.progress_history.append((self.step_count, self.cumulative_forward_progress))
        cutoff = self.step_count - window
        while len(self.progress_history) > 1 and self.progress_history[1][0] <= cutoff:
            self.progress_history.popleft()

        oldest_step, oldest_progress = self.progress_history[0]
        if self.step_count - oldest_step < window:
            return False
        return self.cumulative_forward_progress - oldest_progress < min_delta

    def _info(
        self,
        projection: Any,
        reward_terms: dict[str, float],
        progress_delta: float = 0.0,
        raw_progress_delta: float = 0.0,
        progress_delta_clipped: bool = False,
    ) -> dict[str, Any]:
        pose = self._pose()
        velocity = self._linear_velocity()
        yaw = pose[2]
        forward = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        longitudinal_speed = float(np.dot(velocity, forward))
        lateral_speed = float(np.dot(velocity, lateral))

        return {
            "progress": projection.progress,
            "lap_fraction": projection.progress / self.track.length,
            "cumulative_lap_fraction": self.cumulative_forward_progress / self.track.length,
            "progress_delta": progress_delta,
            "raw_progress_delta": raw_progress_delta,
            "progress_delta_clipped": progress_delta_clipped,
            "position": pose[:2].copy(),
            "heading": float(pose[2]),
            "speed": float(np.linalg.norm(velocity[:2])),
            "longitudinal_speed": longitudinal_speed,
            "lateral_speed": lateral_speed,
            "yaw_rate": self._yaw_rate(),
            "smoothed_action": self.smoothed_action.copy(),
            "lateral_error": projection.lateral_error,
            "heading_error": projection.heading_error,
            "target_speed": self._target_speed(projection.progress),
            "grip_scale": self.grip_scale,
            "off_track": abs(projection.lateral_error) > self.track.half_width,
            "lap_complete": self._lap_complete(),
            "no_progress_timeout": self._no_progress_timeout_info(),
            "reward_terms": reward_terms,
        }

    def _no_progress_timeout_info(self) -> bool:
        window = self.termination_config.no_progress_window_steps
        min_delta = self.termination_config.no_progress_min_delta
        if window <= 0 or min_delta <= 0.0 or not self.progress_history:
            return False
        oldest_step, oldest_progress = self.progress_history[0]
        return (
            self.step_count - oldest_step >= window
            and self.cumulative_forward_progress - oldest_progress < min_delta
        )

    def _validated_progress_delta(self, raw_delta: float, physical_delta: float) -> float:
        max_delta = (
            self.termination_config.max_progress_delta_factor * physical_delta
            + self.termination_config.max_progress_delta_slack
        )
        if max_delta <= 0.0:
            return raw_delta
        return float(np.clip(raw_delta, -max_delta, max_delta))

    def _pose(self) -> np.ndarray:
        qpos = self.data.qpos[self.root_qpos_adr : self.root_qpos_adr + 7]
        return np.array([qpos[0], qpos[1], quat_to_yaw(qpos[3:7])], dtype=float)

    def _linear_velocity(self) -> np.ndarray:
        return self.data.qvel[self.root_dof_adr : self.root_dof_adr + 3].copy()

    def _yaw_rate(self) -> float:
        return float(self.data.qvel[self.root_dof_adr + 5])

    def _lap_complete(self) -> bool:
        return self.cumulative_forward_progress >= self.lap_target * self.track.length


def yaw_to_quat(yaw: float) -> np.ndarray:
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float)


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(wrap_angle(np.arctan2(siny_cosp, cosy_cosp)))


def normalize_2d(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector[:2])
    if norm <= 1e-12:
        return vector
    return vector / norm


def cross_z(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])
