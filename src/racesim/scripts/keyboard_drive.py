from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from racesim.controllers.factory import CONTROLLERS, make_controller
from racesim.env.racing_env import RacingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the simplified MuJoCo car manually.")
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--autopilot",
        choices=tuple(name for name in CONTROLLERS if name != "open_loop"),
        default="racing_line",
        help="Built-in controller toggled by H when --policy-model is not set.",
    )
    parser.add_argument(
        "--policy-model",
        type=Path,
        default=None,
        help="Stable-Baselines3 PPO model zip toggled by H instead of a built-in controller.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions for --policy-model.",
    )
    parser.add_argument(
        "--camera",
        choices=("chase", "topdown", "free", "fixed"),
        default="chase",
        help="Initial MuJoCo viewer camera.",
    )
    parser.add_argument(
        "--no-reexec",
        action="store_true",
        help="Do not relaunch through mjpython on macOS.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    maybe_reexec_with_mjpython(args)

    import mujoco.viewer

    env = RacingEnv(args.config)
    controller, controller_name = make_autopilot(env, args)
    observation, info = env.reset(seed=args.seed)

    command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    state = {"autopilot": False, "reset": False, "quit": False, "camera": args.camera}

    def key_callback(key: int) -> None:
        char = chr(key).lower() if 0 <= key < 256 else ""
        if char == "i":
            command[1] = min(command[1] + 0.1, 1.0)
            command[2] = 0.0
        elif char == "k":
            command[1] = max(command[1] - 0.1, 0.0)
        elif char == "j":
            command[2] = min(command[2] + 0.1, 1.0)
            command[1] = 0.0
        elif char == "u":
            command[2] = max(command[2] - 0.1, 0.0)
        elif char == "f":
            command[0] = min(command[0] + 0.1, 1.0)
        elif char == "g":
            command[0] = max(command[0] - 0.1, -1.0)
        elif char == "t":
            command[0] = 0.0
        elif char == " ":
            command[:] = 0.0
        elif char == "h":
            state["autopilot"] = not state["autopilot"]
        elif char == "r":
            state["reset"] = True
        elif char == "q":
            state["quit"] = True
        elif char == "1":
            state["camera"] = "chase"
        elif char == "2":
            state["camera"] = "topdown"
        elif char == "3":
            state["camera"] = "free"
        elif char == "4":
            state["camera"] = "fixed"

    print(
        "Keyboard drive controls: I/K throttle up/down, J/U brake up/down, "
        f"F/G steer, T center, Space zero, H {controller_name} toggle, R reset, Q quit. "
        "Camera: 1 chase, 2 topdown, 3 free, 4 fixed."
    )

    try:
        viewer_context = mujoco.viewer.launch_passive(
            env.model,
            env.data,
            key_callback=key_callback,
            show_left_ui=False,
            show_right_ui=False,
        )
    except RuntimeError as exc:
        if "mjpython" in str(exc):
            raise RuntimeError(
                "MuJoCo's passive viewer requires mjpython on macOS. "
                "Run: mjpython -m racesim.scripts.keyboard_drive"
            ) from exc
        raise

    with viewer_context as viewer:
        apply_viewer_camera(viewer, env, state["camera"])
        next_print = time.monotonic()
        while viewer.is_running() and not state["quit"]:
            apply_viewer_camera(viewer, env, state["camera"])
            if state["reset"]:
                observation, info = env.reset(seed=args.seed)
                command[:] = 0.0
                state["reset"] = False

            action = controller.act(observation, info) if state["autopilot"] else command
            observation, _reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                observation, info = env.reset(seed=args.seed)
                command[:] = 0.0

            now = time.monotonic()
            if now >= next_print:
                mode = controller_name if state["autopilot"] else "manual"
                print(format_hud_status(info, action, mode, state["camera"]))
                next_print = now + 1.0

            update_viewer_hud(viewer, env, hud_fields(info, action))
            viewer.sync()
            time.sleep(env.control_timestep())

        viewer.close()


def make_autopilot(env: RacingEnv, args: argparse.Namespace) -> tuple[object, str]:
    if args.policy_model is None:
        return make_controller(args.autopilot, env.track), args.autopilot

    from stable_baselines3 import PPO

    model = PPO.load(args.policy_model)
    return (
        PolicyDriver(model, deterministic=args.deterministic),
        f"ppo:{args.policy_model.stem}",
    )


class PolicyDriver:
    def __init__(self, model: object, deterministic: bool) -> None:
        self.model = model
        self.deterministic = deterministic

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        del info
        action, _state = self.model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action, dtype=np.float32)


def hud_fields(info: dict[str, Any], action: np.ndarray) -> dict[str, float | str]:
    action = np.asarray(action, dtype=float)
    tire = info.get("tire_usage", {})
    peak_tire = max(float(tire.get("front", 0.0)), float(tire.get("rear", 0.0)))
    return {
        "speed": float(info["speed"]),
        "lateral_error": float(info["lateral_error"]),
        "heading_error": float(info["heading_error"] or 0.0),
        "lap": float(info["cumulative_lap_fraction"]),
        "target_speed": float(info.get("target_speed", 0.0)),
        "peak_tire_usage": peak_tire,
        "steering": float(action[0]),
        "throttle": float(action[1]),
        "brake": float(action[2]),
    }


def format_hud_status(
    info: dict[str, Any],
    action: np.ndarray,
    mode: str,
    camera: str,
) -> str:
    fields = hud_fields(info, action)
    return (
        f"{mode} camera={camera} "
        f"spd={fields['speed']:.2f}m/s tgt={fields['target_speed']:.1f} "
        f"lat={fields['lateral_error']:.2f}m head={fields['heading_error']:.2f} "
        f"lap={fields['lap']:.2f} tire={fields['peak_tire_usage']:.2f} "
        f"act=[{fields['steering']:.2f},{fields['throttle']:.2f},{fields['brake']:.2f}]"
    )


def update_viewer_hud(viewer: object, env: RacingEnv, fields: dict[str, float | str]) -> None:
    """Draw a short lateral-error indicator toward the track centerline."""
    del fields
    if not hasattr(viewer, "user_scn"):
        return
    import mujoco

    pose = env._pose()
    projection = env.track.project(pose[:2], heading=pose[2])
    point, _tangent, _normal = env.track.sample_at(projection.progress)
    car = np.array([pose[0], pose[1], 0.35], dtype=float)
    center = np.array([point[0], point[1], 0.35], dtype=float)
    scn = viewer.user_scn
    scn.ngeom = 0
    if scn.ngeom >= scn.maxgeom:
        return
    mujoco.mjv_initGeom(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_LINE,
        size=np.zeros(3),
        pos=np.zeros(3),
        mat=np.eye(3).flatten(),
        rgba=np.array([0.2, 0.9, 0.3, 0.8], dtype=np.float32),
    )
    mujoco.mjv_connector(
        scn.geoms[scn.ngeom],
        type=mujoco.mjtGeom.mjGEOM_LINE,
        width=0.02,
        from_=car,
        to=center,
    )
    scn.ngeom += 1


def apply_viewer_camera(viewer: object, env: RacingEnv, camera_name: str) -> None:
    import mujoco

    if camera_name == "chase":
        pose = env._pose()
        yaw_degrees = np.degrees(pose[2])
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [pose[0], pose[1], 0.8]
        viewer.cam.distance = 8.0
        viewer.cam.azimuth = yaw_degrees
        viewer.cam.elevation = -18.0
        return

    if camera_name == "free":
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        viewer.cam.distance = 95.0
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -89.0
        return

    if camera_name == "fixed":
        camera_name = "follow"

    camera_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if camera_id < 0:
        raise ValueError(f"Unknown MuJoCo camera: {camera_name}")

    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    viewer.cam.fixedcamid = camera_id


def maybe_reexec_with_mjpython(args: argparse.Namespace) -> None:
    if args.no_reexec:
        return
    if platform.system() != "Darwin":
        return
    if Path(sys.executable).name == "mjpython":
        return
    if os.environ.get("RACESIM_MJPYTHON_REEXEC") == "1":
        return

    mjpython = shutil.which("mjpython")
    if mjpython is None:
        print(
            "MuJoCo's viewer needs mjpython on macOS, but mjpython was not found. "
            "Install MuJoCo's Python package or run the render command instead:\n"
            "  racesim-render-rollout --controller heuristic --steps 1800",
            file=sys.stderr,
        )
        return

    env = os.environ.copy()
    env["RACESIM_MJPYTHON_REEXEC"] = "1"
    command = [mjpython, "-m", "racesim.scripts.keyboard_drive", *sys.argv[1:]]
    print(f"Relaunching MuJoCo viewer with mjpython: {' '.join(command)}")
    completed = subprocess.run(command, env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
