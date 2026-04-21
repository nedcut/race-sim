from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from racesim.controllers.factory import CONTROLLERS, make_controller
from racesim.env.racing_env import RacingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a top-down rollout GIF and trace plot.")
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--controller", choices=CONTROLLERS, default="centerline")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--gif", type=Path, default=Path("results/heuristic_rollout.gif"))
    parser.add_argument("--trace", type=Path, default=Path("results/heuristic_trace.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = RacingEnv(args.config)
    controller = make_controller(args.controller, env.track)
    states, final_info = collect_rollout(env, controller, args.controller, args.steps, args.every)

    args.gif.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    render_gif(env, states, args.gif, fps=args.fps)
    render_trace(env, states, args.trace)

    print(
        f"Wrote {args.gif} and {args.trace}. "
        f"lap_complete={final_info['lap_complete']} off_track={final_info['off_track']} "
        f"cumulative_lap_fraction={final_info['cumulative_lap_fraction']:.3f}"
    )


def collect_rollout(
    env: RacingEnv,
    controller: object,
    controller_name: str,
    steps: int,
    every: int,
) -> tuple[list[dict], dict]:
    observation, info = env.reset(seed=0)
    states: list[dict] = []

    for step in range(steps):
        action = controller.act(observation, info)
        observation, reward, terminated, truncated, info = env.step(action)
        if step % every == 0 or terminated or truncated:
            states.append(
                {
                    "position": np.asarray(info["position"], dtype=float),
                    "heading": float(info["heading"]),
                    "speed": float(info["speed"]),
                    "lateral_error": float(info["lateral_error"]),
                    "reward": float(reward),
                    "action": np.asarray(action, dtype=float),
                    "lap_fraction": float(info["cumulative_lap_fraction"]),
                }
            )

        if terminated or truncated:
            break

    return states, info


def render_trace(env: RacingEnv, states: list[dict], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    draw_track(ax, env)

    positions = np.asarray([state["position"] for state in states])
    speeds = np.asarray([state["speed"] for state in states])
    scatter = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        c=speeds,
        cmap="viridis",
        s=12,
        label="car trajectory",
    )
    fig.colorbar(scatter, ax=ax, label="speed [m/s]")
    ax.set_title("Heuristic rollout trace")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_gif(env: RacingEnv, states: list[dict], output: Path, fps: int) -> None:
    frames = []
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    positions = np.asarray([state["position"] for state in states])
    xmin, ymin = env.track.centerline.min(axis=0) - env.track.width
    xmax, ymax = env.track.centerline.max(axis=0) + env.track.width

    for index, state in enumerate(states):
        ax.clear()
        draw_track(ax, env)
        ax.plot(positions[: index + 1, 0], positions[: index + 1, 1], color="#2563eb")
        car = car_outline(state["position"], state["heading"])
        ax.fill(car[:, 0], car[:, 1], color="#0f172a")
        ax.plot(car[[0, 1], 0], car[[0, 1], 1], color="#f97316", linewidth=2)
        ax.text(
            0.02,
            0.98,
            (
                f"lap {state['lap_fraction']:.2f}\n"
                f"speed {state['speed']:.1f} m/s\n"
                f"lat {state['lateral_error']:.2f} m"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.85},
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())

    imageio.mimsave(output, frames, fps=fps)
    plt.close(fig)


def draw_track(ax: plt.Axes, env: RacingEnv) -> None:
    left, right = env.track.boundaries()
    center = env.track.centerline
    closed_left = np.vstack([left, left[0]])
    closed_right = np.vstack([right, right[0]])
    closed_center = np.vstack([center, center[0]])

    ax.fill(closed_left[:, 0], closed_left[:, 1], color="#d6d1c4", alpha=0.8)
    ax.fill(closed_right[:, 0], closed_right[:, 1], color="white")
    ax.plot(closed_left[:, 0], closed_left[:, 1], color="#991b1b", linewidth=1)
    ax.plot(closed_right[:, 0], closed_right[:, 1], color="#991b1b", linewidth=1)
    ax.plot(closed_center[:, 0], closed_center[:, 1], color="#64748b", linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.15)


def car_outline(
    position: np.ndarray,
    heading: float,
    length: float = 2.0,
    width: float = 1.0,
) -> np.ndarray:
    half_length = length / 2.0
    half_width = width / 2.0
    local = np.array(
        [
            [half_length, 0.0],
            [half_length * 0.35, half_width],
            [-half_length, half_width],
            [-half_length, -half_width],
            [half_length * 0.35, -half_width],
        ],
        dtype=float,
    )
    rotation = np.array(
        [
            [np.cos(heading), -np.sin(heading)],
            [np.sin(heading), np.cos(heading)],
        ]
    )
    return local @ rotation.T + position


if __name__ == "__main__":
    main()
