from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from racesim.controllers.heuristic import HeuristicController
from racesim.env.racing_env import RacingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Drive the simplified MuJoCo car manually.")
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    import mujoco.viewer

    args = parse_args()
    env = RacingEnv(args.config)
    controller = HeuristicController(env.track)
    observation, info = env.reset(seed=args.seed)

    command = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    state = {"autopilot": False, "reset": False, "quit": False}

    def key_callback(key: int) -> None:
        char = chr(key).lower() if 0 <= key < 256 else ""
        if char == "w":
            command[1] = min(command[1] + 0.1, 1.0)
            command[2] = 0.0
        elif char == "x":
            command[1] = max(command[1] - 0.1, 0.0)
        elif char == "s":
            command[2] = min(command[2] + 0.1, 1.0)
            command[1] = 0.0
        elif char == "e":
            command[2] = max(command[2] - 0.1, 0.0)
        elif char == "a":
            command[0] = max(command[0] - 0.1, -1.0)
        elif char == "d":
            command[0] = min(command[0] + 0.1, 1.0)
        elif char == "c":
            command[0] = 0.0
        elif char == " ":
            command[:] = 0.0
        elif char == "h":
            state["autopilot"] = not state["autopilot"]
        elif char == "r":
            state["reset"] = True
        elif char == "q":
            state["quit"] = True

    print(
        "Keyboard drive controls: W/X throttle up/down, S/E brake up/down, "
        "A/D steer, C center, Space zero, H heuristic toggle, R reset, Q quit."
    )

    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=key_callback) as viewer:
        next_print = time.monotonic()
        while viewer.is_running() and not state["quit"]:
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
                    f"lap={info['cumulative_lap_fraction']:.2f}"
                )
                next_print = now + 1.0

            viewer.sync()
            time.sleep(env.frame_skip * env.model.opt.timestep)

        viewer.close()


if __name__ == "__main__":
    main()
