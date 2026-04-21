from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from racesim.controllers.heuristic import HeuristicController
from racesim.env.racing_env import RacingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the simplified MuJoCo car manually.")
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--seed", type=int, default=0)
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
    controller = HeuristicController(env.track)
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
            command[0] = max(command[0] - 0.1, -1.0)
        elif char == "g":
            command[0] = min(command[0] + 0.1, 1.0)
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
        "F/G steer, T center, Space zero, H heuristic toggle, R reset, Q quit. "
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
                mode = "heuristic" if state["autopilot"] else "manual"
                print(
                    f"{mode} action={np.asarray(action).round(2)} "
                    f"speed={info['speed']:.2f}m/s lat={info['lateral_error']:.2f}m "
                    f"lap={info['cumulative_lap_fraction']:.2f} camera={state['camera']}"
                )
                next_print = now + 1.0

            viewer.sync()
            time.sleep(env.frame_skip * env.model.opt.timestep)

        viewer.close()


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
