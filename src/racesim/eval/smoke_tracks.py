from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

from racesim.controllers.factory import CONTROLLERS, make_controller
from racesim.env.racing_env import RacingEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick controller smoke tests across tracks.")
    parser.add_argument("--catalog", type=Path, default=Path("configs/track_catalog.yaml"))
    parser.add_argument("--controller", choices=CONTROLLERS, default="racing_line")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--output", type=Path, default=Path("results/track_smoke.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = smoke_tracks(args.catalog, args.controller, args.steps)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["track", "status", "steps", "lap_fraction", "speed"],
        )
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['track']:12s} {row['status']:8s} steps={row['steps']:4d} "
            f"lap={row['lap_fraction']:.2f} speed={row['speed']:.1f}"
        )
    print(f"Wrote {args.output}")


def smoke_tracks(catalog_path: Path, controller_name: str, steps: int) -> list[dict]:
    tracks = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["tracks"]
    rows = []
    for name, entry in tracks.items():
        env = RacingEnv(entry["env"])
        controller = make_controller(controller_name, env.track)
        observation, info = env.reset(seed=0)
        step_count = 0
        for step in range(1, steps + 1):
            step_count = step
            action = controller.act(observation, info)
            observation, _reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        status = "complete" if info["lap_complete"] else "off" if info["off_track"] else "timeout"
        rows.append(
            {
                "track": name,
                "status": status,
                "steps": step_count,
                "lap_fraction": float(info["cumulative_lap_fraction"]),
                "speed": float(info["speed"]),
            }
        )
    return rows


if __name__ == "__main__":
    main()
