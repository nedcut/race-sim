from __future__ import annotations

from collections import deque
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np
import yaml
from gymnasium import spaces

from racesim.env.track import ClosedTrack
from racesim.paths import default_env_config, project_root, resolve_resource
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
class TireModelConfig:
    center_of_mass_height: float = 0.28
    tire_friction_coefficient: float | None = None
    combined_slip_exponent: float = 2.0
    brake_front_bias: float = 0.6
    aero_downforce_coefficient: float = 0.0
    aero_drag_coefficient: float = 0.0
    aero_front_balance: float = 0.5
    peak_slip_angle: float = 0.12
    lateral_saturation_softness: float = 1.15
    longitudinal_saturation_softness: float = 1.0
    low_speed_slip_floor: float = 1.2


@dataclass(frozen=True)
class ChassisConfig:
    mass_kg: float | None = None
    ixx: float | None = None
    iyy: float | None = None
    izz: float | None = None


@dataclass(frozen=True)
class TireForces:
    longitudinal: float
    lateral: float
    usage: float
    raw_lateral: float
    slip_angle: float


@dataclass(frozen=True)
class TireTelemetry:
    front_slip_angle: float = 0.0
    rear_slip_angle: float = 0.0
    front_longitudinal_force: float = 0.0
    rear_longitudinal_force: float = 0.0
    front_lateral_force: float = 0.0
    rear_lateral_force: float = 0.0
    front_raw_lateral_force: float = 0.0
    rear_raw_lateral_force: float = 0.0
    yaw_torque: float = 0.0
    steering_angle: float = 0.0
    understeer_score: float = 0.0


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

    The car is still a free body, but the applied forces come from a compact
    bicycle-style proxy with axle loads and combined tire limits.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 12}

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        resolved = default_env_config() if config_path is None else resolve_resource(config_path)
        self.config_path = resolved
        self.config = self._load_config(self.config_path)
        if self.config_path.parent.name == "configs":
            root = self.config_path.parent.parent
        else:
            root = project_root()

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
        if "dt" in sim_config:
            self.model.opt.timestep = float(sim_config["dt"])

        render_config = self.config.get("render", {})
        self.render_mode = render_mode if render_mode is not None else render_config.get("mode")
        self._render_width = int(render_config.get("width", 640))
        self._render_height = int(render_config.get("height", 480))
        self._renderer: mujoco.Renderer | None = None
        self.metadata = {
            "render_modes": ["rgb_array", "human"],
            "render_fps": max(1, int(round(1.0 / max(self.control_timestep(), 1e-6)))),
        }

        self.control = self._load_control_config(root)
        self.tire_model = self._load_tire_model_config(root)
        self.chassis = self._load_chassis_config(root)
        self._apply_chassis_overrides()
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
        self.last_tire_usage = {"front": 0.0, "rear": 0.0}
        self.last_normal_loads = {"front": 0.0, "rear": 0.0}
        self.last_tire_telemetry = TireTelemetry()
        self.next_progress_gate = 0.0
        self.progress_history: deque[tuple[int, float]] = deque()
        self._cached_target_speed: tuple[float, float] | None = None

    def control_timestep(self) -> float:
        """Wall-clock seconds between successive env.step() transitions."""
        return float(self.frame_skip * self.model.opt.timestep)

    def render(self) -> np.ndarray | None:
        if self.render_mode is None or self.render_mode == "human":
            return None
        if self.render_mode != "rgb_array":
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self._render_height, width=self._render_width
            )

        camera = "topdown"
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
        if camera_id < 0:
            self._renderer.update_scene(self.data)
        else:
            self._renderer.update_scene(self.data, camera=camera)
        pixels = self._renderer.render()
        return np.asarray(pixels, dtype=np.uint8)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        super().close()

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

    def _load_control_config(self, root: Path) -> ControlConfig:
        control_config = {}
        vehicle_path = self.config.get("vehicle")
        if vehicle_path is not None:
            vehicle_config = self._load_config(self._resolve_path(vehicle_path, root))
            control_config.update(vehicle_config.get("control", {}))
            control_config.update(vehicle_config.get("tire_model", {}))
        control_config.update(self.config.get("tire_model", {}))
        control_config.update(self.config.get("control", {}))
        control_fields = {field.name for field in fields(ControlConfig)}
        return ControlConfig(
            **{key: value for key, value in control_config.items() if key in control_fields}
        )

    def _load_tire_model_config(self, root: Path) -> TireModelConfig:
        tire_model_config = {}
        vehicle_path = self.config.get("vehicle")
        if vehicle_path is not None:
            vehicle_config = self._load_config(self._resolve_path(vehicle_path, root))
            tire_model_config.update(vehicle_config.get("tire_model", {}))
        tire_model_config.update(self.config.get("tire_model", {}))
        tire_model_fields = {field.name for field in fields(TireModelConfig)}
        tire_model_config.update(
            {
                key: value
                for key, value in self.config.get("control", {}).items()
                if key in tire_model_fields
            }
        )
        return TireModelConfig(
            **{key: value for key, value in tire_model_config.items() if key in tire_model_fields}
        )

    def _load_chassis_config(self, root: Path) -> ChassisConfig:
        chassis_config: dict[str, Any] = {}
        vehicle_path = self.config.get("vehicle")
        if vehicle_path is not None:
            vehicle_config = self._load_config(self._resolve_path(vehicle_path, root))
            chassis_config.update(vehicle_config.get("chassis", {}))
        chassis_config.update(self.config.get("chassis", {}))
        chassis_fields = {field.name for field in fields(ChassisConfig)}
        return ChassisConfig(
            **{key: value for key, value in chassis_config.items() if key in chassis_fields}
        )

    def _apply_chassis_overrides(self) -> None:
        if self.chassis.mass_kg is not None:
            self.model.body_mass[self.car_body_id] = float(self.chassis.mass_kg)
        inertia = self.model.body_inertia[self.car_body_id].copy()
        if self.chassis.ixx is not None:
            inertia[0] = float(self.chassis.ixx)
        if self.chassis.iyy is not None:
            inertia[1] = float(self.chassis.iyy)
        if self.chassis.izz is not None:
            inertia[2] = float(self.chassis.izz)
        self.model.body_inertia[self.car_body_id] = inertia
        if any(
            value is not None
            for value in (
                self.chassis.mass_kg,
                self.chassis.ixx,
                self.chassis.iyy,
                self.chassis.izz,
            )
        ):
            mujoco.mj_setConst(self.model, self.data)

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
        self.last_tire_usage = {"front": 0.0, "rear": 0.0}
        self.last_normal_loads = {"front": 0.0, "rear": 0.0}
        self.last_tire_telemetry = TireTelemetry()
        self.step_count = 0
        self.next_progress_gate = self._progress_gate_distance()
        self.progress_history.clear()
        self.progress_history.append((0, self.cumulative_forward_progress))
        self._cached_target_speed = None

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
        self._cached_target_speed = (projection.progress, target_speed)
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
        low_speed_floor = max(self.tire_model.low_speed_slip_floor, 0.1)
        safe_speed = np.copysign(max(abs(forward_speed), low_speed_floor), forward_speed or 1.0)

        lf = self.control.center_of_mass_to_front
        lr = self.control.center_of_mass_to_rear
        front_slip = np.arctan2(lateral_speed + lf * yaw_rate, abs(safe_speed)) - steer_angle
        rear_slip = np.arctan2(lateral_speed - lr * yaw_rate, abs(safe_speed))

        drive_force = throttle * self.control.max_drive_force
        brake_direction = np.sign(forward_speed) if abs(forward_speed) > 0.1 else 0.0
        brake_force = brake * self.control.max_brake_force * brake_direction
        drag_force = (
            (self.control.linear_drag + self.tire_model.aero_drag_coefficient)
            * forward_speed
            * abs(forward_speed)
        )
        rolling_direction = np.sign(forward_speed) if abs(forward_speed) > 0.1 else 0.0
        rolling_force = self.control.rolling_resistance * rolling_direction

        front_drive_fraction, rear_drive_fraction = self._drive_split()
        front_brake_fraction, rear_brake_fraction = self._brake_split()
        front_forward_force = (
            drive_force * front_drive_fraction - brake_force * front_brake_fraction
        )
        rear_forward_force = drive_force * rear_drive_fraction - brake_force * rear_brake_fraction

        mass = self._vehicle_mass()
        longitudinal_acceleration = (drive_force - brake_force - drag_force - rolling_force) / max(
            mass, 1e-9
        )
        normal_loads = self._axle_normal_loads(longitudinal_acceleration, abs(forward_speed))
        front_lateral_limit, rear_lateral_limit = self._axle_tire_limits(normal_loads)
        front_tire = self._tire_forces(
            longitudinal_force=front_forward_force,
            slip_angle=front_slip,
            cornering_stiffness=self.control.front_cornering_stiffness,
            force_limit=front_lateral_limit,
        )
        rear_tire = self._tire_forces(
            longitudinal_force=rear_forward_force,
            slip_angle=rear_slip,
            cornering_stiffness=self.control.rear_cornering_stiffness,
            force_limit=rear_lateral_limit,
        )
        self.last_tire_usage = {"front": front_tire.usage, "rear": rear_tire.usage}
        self.last_normal_loads = {"front": normal_loads[0], "rear": normal_loads[1]}

        front_direction = normalize_2d(
            np.cos(steer_angle) * forward + np.sin(steer_angle) * lateral
        )

        front_force = front_tire.longitudinal * front_direction + front_tire.lateral * lateral
        rear_force = rear_tire.longitudinal * forward + rear_tire.lateral * lateral
        force = front_force + rear_force - (drag_force + rolling_force) * forward
        yaw_torque = (
            lf * cross_z(forward, front_force)
            - lr * cross_z(forward, rear_force)
            - self.control.yaw_damping * yaw_rate
        )
        expected_yaw_rate = forward_speed * np.tan(steer_angle) / self._effective_wheelbase()
        self.last_tire_telemetry = TireTelemetry(
            front_slip_angle=front_tire.slip_angle,
            rear_slip_angle=rear_tire.slip_angle,
            front_longitudinal_force=front_tire.longitudinal,
            rear_longitudinal_force=rear_tire.longitudinal,
            front_lateral_force=front_tire.lateral,
            rear_lateral_force=rear_tire.lateral,
            front_raw_lateral_force=front_tire.raw_lateral,
            rear_raw_lateral_force=rear_tire.raw_lateral,
            yaw_torque=float(yaw_torque),
            steering_angle=float(steer_angle),
            understeer_score=float(expected_yaw_rate - yaw_rate),
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

    def _brake_split(self) -> tuple[float, float]:
        front_fraction = float(np.clip(self.tire_model.brake_front_bias, 0.0, 1.0))
        return front_fraction, 1.0 - front_fraction

    def _vehicle_mass(self) -> float:
        subtree_mass = float(self.model.body_subtreemass[self.car_body_id])
        if subtree_mass > 0.0:
            return subtree_mass
        return float(max(self.model.body_mass[self.car_body_id], 1e-9))

    def _axle_normal_loads(
        self,
        longitudinal_acceleration: float,
        speed: float,
    ) -> tuple[float, float]:
        mass = self._vehicle_mass()
        gravity = abs(float(self.model.opt.gravity[2]))
        wheelbase = self._effective_wheelbase()
        lf = self.control.center_of_mass_to_front
        lr = self.control.center_of_mass_to_rear

        static_front = mass * gravity * lr / wheelbase
        static_rear = mass * gravity * lf / wheelbase
        load_transfer = mass * longitudinal_acceleration * self.tire_model.center_of_mass_height
        load_transfer /= wheelbase

        downforce = max(self.tire_model.aero_downforce_coefficient, 0.0) * speed**2
        front_balance = float(np.clip(self.tire_model.aero_front_balance, 0.0, 1.0))

        front_load = static_front - load_transfer + downforce * front_balance
        rear_load = static_rear + load_transfer + downforce * (1.0 - front_balance)
        minimum_load = 0.02 * mass * gravity
        return max(front_load, minimum_load), max(rear_load, minimum_load)

    def _axle_tire_limits(self, normal_loads: tuple[float, float]) -> tuple[float, float]:
        mu = self._tire_friction_coefficient()
        front_load, rear_load = normal_loads
        return mu * self.grip_scale * front_load, mu * self.grip_scale * rear_load

    def _tire_friction_coefficient(self) -> float:
        configured_mu = self.tire_model.tire_friction_coefficient
        if configured_mu is not None and configured_mu > 0.0:
            return configured_mu

        mass = self._vehicle_mass()
        gravity = abs(float(self.model.opt.gravity[2]))
        total_normal_load = max(mass * gravity, 1e-9)
        return 2.0 * self.control.max_lateral_force / total_normal_load

    def _tire_forces(
        self,
        longitudinal_force: float,
        slip_angle: float,
        cornering_stiffness: float,
        force_limit: float,
    ) -> TireForces:
        if force_limit <= 1e-9:
            return TireForces(0.0, 0.0, 0.0, 0.0, float(slip_angle))

        peak_slip = max(self.tire_model.peak_slip_angle, 1e-6)
        softness = max(self.tire_model.lateral_saturation_softness, 1e-6)
        linear_lateral = -cornering_stiffness * self.grip_scale * slip_angle
        shaped_slip = slip_angle / peak_slip
        peak_lateral = force_limit * np.tanh(abs(shaped_slip) / softness)
        lateral_force = -np.sign(slip_angle) * peak_lateral
        if abs(slip_angle) < peak_slip:
            blend = abs(slip_angle) / peak_slip
            lateral_force = (1.0 - blend) * linear_lateral + blend * lateral_force

        longitudinal_force = self._soft_longitudinal_force(longitudinal_force, force_limit)
        longitudinal, lateral, usage = self._limit_combined_tire_force(
            longitudinal_force,
            lateral_force,
            force_limit,
        )
        return TireForces(
            longitudinal=float(longitudinal),
            lateral=float(lateral),
            usage=float(usage),
            raw_lateral=float(linear_lateral),
            slip_angle=float(slip_angle),
        )

    def _soft_longitudinal_force(self, force: float, force_limit: float) -> float:
        softness = max(self.tire_model.longitudinal_saturation_softness, 1e-6)
        return float(force_limit * np.tanh(force / max(force_limit * softness, 1e-9)))

    def _limit_combined_tire_force(
        self,
        longitudinal_force: float,
        lateral_force: float,
        force_limit: float,
    ) -> tuple[float, float, float]:
        if force_limit <= 1e-9:
            return 0.0, 0.0, 0.0

        exponent = max(self.tire_model.combined_slip_exponent, 1.0)
        usage = (abs(longitudinal_force) / force_limit) ** exponent + (
            abs(lateral_force) / force_limit
        ) ** exponent
        if usage <= 1.0:
            return longitudinal_force, lateral_force, usage ** (1.0 / exponent)

        scale = usage ** (-1.0 / exponent)
        return longitudinal_force * scale, lateral_force * scale, 1.0

    def _effective_wheelbase(self) -> float:
        configured = self.control.center_of_mass_to_front + self.control.center_of_mass_to_rear
        if configured > 1e-9:
            return configured
        return max(self.control.wheelbase, 1e-9)

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
            self._lookup_target_speed(projection.progress) / self.reward_config.target_speed_max
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

    def _lookup_target_speed(self, progress: float) -> float:
        if self._cached_target_speed is not None and self._cached_target_speed[0] == progress:
            return self._cached_target_speed[1]
        return self._target_speed(progress)

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
        tire = self.last_tire_telemetry

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
            "target_speed": self._lookup_target_speed(projection.progress),
            "grip_scale": self.grip_scale,
            "tire_usage": self.last_tire_usage.copy(),
            "normal_loads": self.last_normal_loads.copy(),
            "slip_angles": {
                "front": tire.front_slip_angle,
                "rear": tire.rear_slip_angle,
            },
            "tire_forces": {
                "front_longitudinal": tire.front_longitudinal_force,
                "rear_longitudinal": tire.rear_longitudinal_force,
                "front_lateral": tire.front_lateral_force,
                "rear_lateral": tire.rear_lateral_force,
                "front_raw_lateral": tire.front_raw_lateral_force,
                "rear_raw_lateral": tire.rear_raw_lateral_force,
            },
            "yaw_torque": tire.yaw_torque,
            "steering_angle": tire.steering_angle,
            "understeer_score": tire.understeer_score,
            "off_track": self.track.is_off_track(pose[:2], margin=self.off_track_margin),
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
