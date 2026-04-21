from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from racesim.env.racing_env import RacingEnv
from racesim.eval.plots import draw_track


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch live PPO eval rollouts.")
    parser.add_argument(
        "--watch",
        type=Path,
        default=Path("results/ppo_oval/live/latest_rollout.json"),
    )
    parser.add_argument("--history", type=int, default=6)
    parser.add_argument("--interval-ms", type=int, default=120)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard = LiveDashboard(args.watch, history=args.history)
    dashboard.run(interval_ms=args.interval_ms)


class LiveDashboard:
    def __init__(self, watch_path: Path, history: int = 6) -> None:
        self.watch_path = watch_path
        self.history_limit = history
        self.last_mtime = 0.0
        self.current: dict[str, Any] | None = None
        self.history: list[list[dict[str, Any]]] = []
        self.frame = 0

        self.fig, self.ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        self.env: RacingEnv | None = None

    def run(self, interval_ms: int) -> None:
        self.animation = FuncAnimation(
            self.fig,
            self.update,
            interval=interval_ms,
            cache_frame_data=False,
        )
        plt.show()

    def update(self, _frame_index: int) -> None:
        self.maybe_reload()
        self.ax.clear()
        if self.current is None:
            self.ax.text(
                0.5,
                0.5,
                f"Waiting for {self.watch_path}",
                transform=self.ax.transAxes,
                ha="center",
                va="center",
            )
            self.ax.axis("off")
            return

        trajectory = self.current["episodes"][0]["trajectory"]
        if not trajectory:
            return
        if self.env is not None:
            draw_track(self.ax, self.env)

        for index, old_trajectory in enumerate(self.history[-self.history_limit :]):
            alpha = 0.08 + 0.10 * (index + 1) / max(len(self.history), 1)
            x = [row["x"] for row in old_trajectory]
            y = [row["y"] for row in old_trajectory]
            self.ax.plot(x, y, color="#64748b", alpha=alpha, linewidth=1.0)

        point_count = len(trajectory)
        self.frame = (self.frame + 1) % max(point_count, 1)
        visible = trajectory[: self.frame + 1]
        x = np.array([row["x"] for row in visible])
        y = np.array([row["y"] for row in visible])
        speed = visible[-1]["speed"]

        self.ax.plot(x, y, color="#2563eb", linewidth=1.8)
        self.ax.scatter([x[-1]], [y[-1]], color="#f97316", s=60, zorder=5)
        summary = self.current["summary"]
        timestep = self.current.get("timestep", 0)
        self.ax.set_title(
            f"PPO live eval | step {timestep} | "
            f"completion {summary['completion_rate']:.2f} | speed {speed:.1f} m/s"
        )

    def maybe_reload(self) -> None:
        if not self.watch_path.exists():
            return
        mtime = self.watch_path.stat().st_mtime
        if mtime <= self.last_mtime:
            return
        self.last_mtime = mtime
        payload = json.loads(self.watch_path.read_text(encoding="utf-8"))
        if self.current is not None:
            old_trajectory = self.current["episodes"][0].get("trajectory", [])
            if old_trajectory:
                self.history.append(old_trajectory)
                self.history = self.history[-self.history_limit :]
        self.current = payload
        self.frame = 0
        self.env = RacingEnv(payload.get("config", "configs/env.yaml"))


if __name__ == "__main__":
    main()
