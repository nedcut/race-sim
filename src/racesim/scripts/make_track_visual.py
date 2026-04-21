from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from racesim.env.track import ClosedTrack


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MuJoCo visual geoms for a track config.")
    parser.add_argument("--track", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", default="track_visuals")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    track = ClosedTrack.from_config(args.track)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(track_visual_xml(track, args.model_name), encoding="utf-8")
    print(f"Wrote {args.output}")


def track_visual_xml(track: ClosedTrack, model_name: str) -> str:
    left, right = track.boundaries()
    center = track.centerline
    next_left = np.roll(left, -1, axis=0)
    next_right = np.roll(right, -1, axis=0)
    next_center = np.roll(center, -1, axis=0)

    vertices = []
    for point in left:
        vertices.extend([point[0], point[1], 0.022])
    for point in right:
        vertices.extend([point[0], point[1], 0.022])

    faces = []
    n = len(center)
    for index in range(n):
        next_index = (index + 1) % n
        faces.extend([index, n + next_index, next_index])
        faces.extend([index, n + index, n + next_index])

    lines = [
        f'<mujoco model="{model_name}">',
        "  <asset>",
        f'    <mesh name="track_surface_mesh" vertex="{format_values(vertices)}" '
        f'face="{format_values(faces)}"/>',
        '    <material name="track_asphalt" rgba="0.48 0.49 0.46 1" emission="0.08"/>',
        '    <material name="track_boundary" rgba="0.95 0.08 0.06 1" emission="0.05"/>',
        '    <material name="track_centerline" rgba="1.0 0.95 0.55 1" emission="0.15"/>',
        "  </asset>",
        "  <worldbody>",
        '    <geom name="track_surface" type="mesh" mesh="track_surface_mesh" '
        'material="track_asphalt" contype="0" conaffinity="0"/>',
    ]

    for side_name, points, next_points in [
        ("left", left, next_left),
        ("right", right, next_right),
    ]:
        for index, (start, end) in enumerate(zip(points, next_points, strict=True)):
            lines.append(
                f'    <geom name="track_{side_name}_boundary_{index:03d}" type="capsule" '
                f'fromto="{start[0]:.5f} {start[1]:.5f} 0.09 '
                f'{end[0]:.5f} {end[1]:.5f} 0.09" '
                'size="0.14" material="track_boundary" contype="0" conaffinity="0"/>'
            )

    for index, (start, end) in enumerate(zip(center, next_center, strict=True)):
        if index % 4:
            continue
        midpoint = 0.5 * (start + end)
        vector = end - start
        length = min(float(np.linalg.norm(vector)) * 0.45, 1.0)
        yaw = math.degrees(math.atan2(vector[1], vector[0]))
        lines.append(
            f'    <geom name="track_center_mark_{index:03d}" type="box" '
            f'pos="{midpoint[0]:.5f} {midpoint[1]:.5f} 0.045" '
            f'euler="0 0 {yaw:.5f}" size="{length / 2:.5f} 0.04500 0.018" '
            'material="track_centerline" contype="0" conaffinity="0"/>'
        )

    lines.extend(["  </worldbody>", "</mujoco>"])
    return "\n".join(lines) + "\n"


def format_values(values: list[float] | list[int]) -> str:
    return " ".join(f"{value:.5f}" if isinstance(value, float) else str(value) for value in values)


if __name__ == "__main__":
    main()
