from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import yaml

from racesim.controllers.factory import CONTROLLERS, make_controller
from racesim.env.racing_env import RacingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render top-down rollout animations and trace plots."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
    parser.add_argument("--controller", choices=CONTROLLERS, default="centerline")
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Stable-Baselines3 PPO model zip to render instead of a built-in controller.",
    )
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--steps", type=int, default=1800)
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--gif", type=Path, default=Path("results/heuristic_rollout.gif"))
    parser.add_argument("--trace", type=Path, default=Path("results/heuristic_trace.png"))
    parser.add_argument(
        "--all-tracks",
        action="store_true",
        help="Render one rollout for every track in the catalog.",
    )
    parser.add_argument("--catalog", type=Path, default=Path("configs/track_catalog.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/policy_rollouts"))
    parser.add_argument(
        "--format",
        choices=("gif", "mp4"),
        default="gif",
        help="Animation format used with --all-tracks.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = load_policy_model(args.model) if args.model is not None else None

    if args.all_tracks:
        render_catalog(args, model)
        return

    env = RacingEnv(args.config)
    driver, driver_name = make_driver(env, args, model)
    states, final_info = collect_rollout(env, driver, driver_name, args.steps, args.every)

    animation_path = args.gif
    animation_path.parent.mkdir(parents=True, exist_ok=True)
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    render_animation(env, states, animation_path, fps=args.fps, title=driver_name)
    render_trace(env, states, args.trace, title=driver_name)

    print(
        f"Wrote {animation_path} and {args.trace}. "
        f"lap_complete={final_info['lap_complete']} off_track={final_info['off_track']} "
        f"cumulative_lap_fraction={final_info['cumulative_lap_fraction']:.3f}"
    )


def render_catalog(args: argparse.Namespace, model: object | None) -> None:
    track_configs = load_catalog_env_configs(args.catalog)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for track_name, config_path in track_configs.items():
        env = RacingEnv(config_path)
        driver, driver_name = make_driver(env, args, model)
        states, final_info = collect_rollout(env, driver, driver_name, args.steps, args.every)

        animation_path = args.output_dir / f"{track_name}_rollout.{args.format}"
        trace_path = args.output_dir / f"{track_name}_trace.png"
        title = f"{track_name} - {driver_name}"
        render_animation(env, states, animation_path, fps=args.fps, title=title)
        render_trace(env, states, trace_path, title=title)

        print(
            f"Wrote {animation_path} and {trace_path}. "
            f"lap_complete={final_info['lap_complete']} off_track={final_info['off_track']} "
            f"cumulative_lap_fraction={final_info['cumulative_lap_fraction']:.3f}"
        )


def load_catalog_env_configs(catalog_path: Path) -> dict[str, Path]:
    with catalog_path.open("r", encoding="utf-8") as file:
        catalog = yaml.safe_load(file)
    return {
        str(name): Path(entry["env"])
        for name, entry in catalog.get("tracks", {}).items()
        if "env" in entry
    }


def load_policy_model(model_path: Path) -> object:
    from stable_baselines3 import PPO

    return PPO.load(model_path)


def make_driver(
    env: RacingEnv,
    args: argparse.Namespace,
    model: object | None,
) -> tuple[object, str]:
    if model is not None:
        return PolicyDriver(model, deterministic=args.deterministic), f"ppo:{args.model.stem}"
    return make_controller(args.controller, env.track), args.controller


class PolicyDriver:
    def __init__(self, model: object, deterministic: bool) -> None:
        self.model = model
        self.deterministic = deterministic

    def act(self, observation: np.ndarray, info: dict[str, Any]) -> np.ndarray:
        del info
        action, _state = self.model.predict(observation, deterministic=self.deterministic)
        return np.asarray(action, dtype=float)


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


def render_trace(env: RacingEnv, states: list[dict], output: Path, title: str) -> None:
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
    ax.set_title(f"{title} rollout trace")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def render_animation(
    env: RacingEnv,
    states: list[dict],
    output: Path,
    fps: int,
    title: str,
) -> None:
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
        ax.set_title(title)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() in {".mp4", ".m4v", ".mov"}:
        write_video(output, frames, fps=fps)
    else:
        imageio.mimsave(output, frames, fps=fps)
    plt.close(fig)


def render_gif(env: RacingEnv, states: list[dict], output: Path, fps: int) -> None:
    render_animation(env, states, output, fps=fps, title="rollout")


def write_video(output: Path, frames: list[np.ndarray], fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("MP4 rendering requires ffmpeg on PATH.")
    if not frames:
        raise ValueError("Cannot write a video without frames.")

    height, width = frames[0].shape[:2]
    command = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    assert process.stderr is not None
    for frame in frames:
        process.stdin.write(np.ascontiguousarray(frame).tobytes())
    process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace")
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {return_code}:\n{stderr}")


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
