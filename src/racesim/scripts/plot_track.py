from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from racesim.env.track import ClosedTrack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a configured track.")
    parser.add_argument("--config", type=Path, default=Path("configs/tracks/oval.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/track.png"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    track = ClosedTrack.from_config(args.config)
    left, right = track.boundaries()

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    closed_center = np.vstack([track.centerline, track.centerline[0]])
    closed_left = np.vstack([left, left[0]])
    closed_right = np.vstack([right, right[0]])

    ax.fill(
        closed_left[:, 0],
        closed_left[:, 1],
        color="#d6d1c4",
        alpha=0.7,
        label="track surface",
    )
    ax.fill(closed_right[:, 0], closed_right[:, 1], color="white", alpha=1.0)
    ax.plot(closed_center[:, 0], closed_center[:, 1], color="#1f2937", linewidth=1.5)
    ax.plot(closed_left[:, 0], closed_left[:, 1], color="#991b1b", linewidth=1.0)
    ax.plot(closed_right[:, 0], closed_right[:, 1], color="#991b1b", linewidth=1.0)

    for progress in np.linspace(0, track.length, 12, endpoint=False):
        point, tangent, _normal = track.sample_at(progress)
        ax.arrow(
            point[0],
            point[1],
            tangent[0] * 1.6,
            tangent[1] * 1.6,
            head_width=0.6,
            head_length=0.8,
            color="#2563eb",
            length_includes_head=True,
        )

    ax.set_title(f"{track.name} ({track.length:.1f} m)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
