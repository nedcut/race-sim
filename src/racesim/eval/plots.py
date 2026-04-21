from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from racesim.env.racing_env import RacingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot trajectory telemetry from an evaluation JSON."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/eval_heuristic_trajectory.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--episode", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    trajectory = get_trajectory(result, args.episode)
    env = RacingEnv(result.get("config", "configs/env.yaml"))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = plot_rollout(env, trajectory, args.output_dir)
    for output in outputs:
        print(f"Wrote {output}")


def get_trajectory(result: dict[str, Any], episode_index: int) -> list[dict[str, Any]]:
    episode = result["episodes"][episode_index]
    if not isinstance(episode, dict) or "trajectory" not in episode:
        raise ValueError(
            "Evaluation JSON does not include trajectory data. Re-run with --record-trajectory."
        )
    trajectory = episode["trajectory"]
    if not trajectory:
        raise ValueError("Selected episode trajectory is empty.")
    return trajectory


def plot_rollout(env: RacingEnv, trajectory: list[dict[str, Any]], output_dir: Path) -> list[Path]:
    outputs = [
        plot_trajectory(env, trajectory, output_dir / "trajectory.png"),
        plot_speed(trajectory, output_dir / "speed_vs_progress.png"),
        plot_controls(trajectory, output_dir / "controls_vs_progress.png"),
        plot_lateral_error(env, trajectory, output_dir / "lateral_error_vs_progress.png"),
    ]
    return outputs


def plot_trajectory(env: RacingEnv, trajectory: list[dict[str, Any]], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    draw_track(ax, env)

    x = np.array([row["x"] for row in trajectory])
    y = np.array([row["y"] for row in trajectory])
    speed = np.array([row["speed"] for row in trajectory])
    scatter = ax.scatter(x, y, c=speed, cmap="viridis", s=10)
    ax.plot(x, y, color="#1d4ed8", linewidth=0.8, alpha=0.6)
    fig.colorbar(scatter, ax=ax, label="speed [m/s]")
    ax.set_title("Trajectory")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_speed(trajectory: list[dict[str, Any]], output: Path) -> Path:
    progress = progress_array(trajectory)
    speed = np.array([row["speed"] for row in trajectory])
    longitudinal = np.array([row["longitudinal_speed"] for row in trajectory])

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    ax.plot(progress, speed, label="speed", color="#2563eb")
    ax.plot(progress, longitudinal, label="longitudinal", color="#16a34a", alpha=0.8)
    ax.set_xlabel("cumulative lap fraction")
    ax.set_ylabel("m/s")
    ax.set_title("Speed vs progress")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_controls(trajectory: list[dict[str, Any]], output: Path) -> Path:
    progress = progress_array(trajectory)
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for key, color in [
        ("smoothed_steering", "#7c3aed"),
        ("smoothed_throttle", "#16a34a"),
        ("smoothed_brake", "#dc2626"),
    ]:
        ax.plot(
            progress,
            [row[key] for row in trajectory],
            label=key.replace("smoothed_", ""),
            color=color,
        )
    ax.set_xlabel("cumulative lap fraction")
    ax.set_ylabel("action")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Controls vs progress")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_lateral_error(env: RacingEnv, trajectory: list[dict[str, Any]], output: Path) -> Path:
    progress = progress_array(trajectory)
    lateral_error = np.array([row["lateral_error"] for row in trajectory])
    heading_error = np.array([row["heading_error"] for row in trajectory])

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, constrained_layout=True)
    axes[0].plot(progress, lateral_error, color="#ea580c")
    axes[0].axhline(env.track.half_width, color="#991b1b", linestyle="--", linewidth=1)
    axes[0].axhline(-env.track.half_width, color="#991b1b", linestyle="--", linewidth=1)
    axes[0].set_ylabel("lateral error [m]")
    axes[0].set_title("Track-relative error")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(progress, heading_error, color="#0891b2")
    axes[1].set_xlabel("cumulative lap fraction")
    axes[1].set_ylabel("heading error [rad]")
    axes[1].grid(True, alpha=0.2)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def draw_track(ax: plt.Axes, env: RacingEnv) -> None:
    left, right = env.track.boundaries()
    center = env.track.centerline
    closed_left = np.vstack([left, left[0]])
    closed_right = np.vstack([right, right[0]])
    closed_center = np.vstack([center, center[0]])

    ax.fill(closed_left[:, 0], closed_left[:, 1], color="#d6d1c4", alpha=0.85)
    ax.fill(closed_right[:, 0], closed_right[:, 1], color="white")
    ax.plot(closed_left[:, 0], closed_left[:, 1], color="#991b1b", linewidth=1)
    ax.plot(closed_right[:, 0], closed_right[:, 1], color="#991b1b", linewidth=1)
    ax.plot(closed_center[:, 0], closed_center[:, 1], color="#64748b", linestyle="--")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.15)


def progress_array(trajectory: list[dict[str, Any]]) -> np.ndarray:
    return np.array([row["cumulative_lap_fraction"] for row in trajectory])


if __name__ == "__main__":
    main()
