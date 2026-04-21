from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from racesim.env.racing_env import RacingEnv
from racesim.eval.plots import draw_track, get_trajectory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare recorded controller rollouts.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Evaluation JSON files created with --record-trajectory.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/comparison"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs = [load_run(path) for path in args.input]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = RacingEnv(runs[0]["config"])
    outputs = [
        plot_trajectory_overlay(env, runs, args.output_dir / "trajectory_overlay.png"),
        plot_speed_overlay(runs, args.output_dir / "speed_overlay.png"),
        plot_summary_table(runs, args.output_dir / "summary_table.png"),
    ]
    for output in outputs:
        print(f"Wrote {output}")


def load_run(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    trajectory = get_trajectory(result, 0)
    metrics = result["episodes"][0]["metrics"]
    label = result.get("controller", path.stem)
    return {
        "path": path,
        "label": label,
        "config": result.get("config", "configs/env.yaml"),
        "trajectory": trajectory,
        "metrics": metrics,
    }


def plot_trajectory_overlay(env: RacingEnv, runs: list[dict[str, Any]], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    draw_track(ax, env)
    for run in runs:
        trajectory = run["trajectory"]
        x = np.array([row["x"] for row in trajectory])
        y = np.array([row["y"] for row in trajectory])
        ax.plot(x, y, linewidth=1.6, label=run["label"])
    ax.set_title("Controller trajectory comparison")
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_speed_overlay(runs: list[dict[str, Any]], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for run in runs:
        trajectory = run["trajectory"]
        progress = np.array([row["cumulative_lap_fraction"] for row in trajectory])
        speed = np.array([row["speed"] for row in trajectory])
        ax.plot(progress, speed, label=run["label"])
    ax.set_xlabel("cumulative lap fraction")
    ax.set_ylabel("speed [m/s]")
    ax.set_title("Speed comparison")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def plot_summary_table(runs: list[dict[str, Any]], output: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 2.4), constrained_layout=True)
    ax.axis("off")
    rows = [
        [
            run["label"],
            f"{run['metrics']['sim_time']:.2f}",
            f"{run['metrics']['total_reward']:.1f}",
            str(run["metrics"]["lap_complete"]),
            str(run["metrics"]["off_track"]),
        ]
        for run in runs
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["controller", "lap time [s]", "reward", "complete", "off track"],
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.4)
    ax.set_title("Controller summary")
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


if __name__ == "__main__":
    main()
