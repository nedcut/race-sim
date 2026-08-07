from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from racesim.utils.geometry import normalize_vectors, rotate_left, wrap_angle


@dataclass(frozen=True)
class TrackProjection:
    """Track-relative coordinates for a world-space point."""

    point: np.ndarray
    nearest: np.ndarray
    segment_index: int
    segment_fraction: float
    progress: float
    lateral_error: float
    tangent: np.ndarray
    normal: np.ndarray
    heading_error: float | None = None


@dataclass(frozen=True)
class TrackGeometryIssue:
    """A geometric problem detected in generated track curves."""

    curve: str
    segment_index: int
    other_segment_index: int
    message: str


class ClosedTrack:
    """Sampled closed centerline with progress and lateral-error queries.

    The centerline is represented by a closed polyline. Progress is measured in
    meters along that polyline, wrapping at ``length``.
    """

    def __init__(self, centerline: np.ndarray, width: float, name: str = "track") -> None:
        points = np.asarray(centerline, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("centerline must have shape (N, 2).")
        if len(points) < 3:
            raise ValueError("centerline must contain at least 3 points.")
        if width <= 0:
            raise ValueError("track width must be positive.")

        if np.allclose(points[0], points[-1]):
            points = points[:-1]

        self.name = name
        self.centerline = points
        self.width = float(width)
        self.half_width = self.width / 2.0

        next_points = np.roll(points, shift=-1, axis=0)
        self.segment_vectors = next_points - points
        self.segment_lengths = np.linalg.norm(self.segment_vectors, axis=1)
        if np.any(self.segment_lengths <= 1e-9):
            raise ValueError("centerline contains duplicate or near-duplicate adjacent points.")

        self.tangents = normalize_vectors(self.segment_vectors)
        self.normals = rotate_left(self.tangents)
        self.arc_lengths = np.concatenate(([0.0], np.cumsum(self.segment_lengths[:-1])))
        self.length = float(np.sum(self.segment_lengths))

    @classmethod
    def from_config(cls, path: str | Path) -> ClosedTrack:
        config_path = Path(path)
        with config_path.open("r", encoding="utf-8") as file:
            data = yaml.safe_load(file)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClosedTrack:
        kind = data.get("kind", "polyline")
        width = float(data["width"])
        name = data.get("name", kind)

        if kind == "polyline":
            return cls(np.asarray(data["centerline"], dtype=float), width=width, name=name)
        if kind == "catmull_rom":
            centerline = catmull_rom_closed_centerline(
                control_points=np.asarray(data["control_points"], dtype=float),
                samples_per_segment=int(data.get("samples_per_segment", 24)),
                alpha=float(data.get("alpha", 0.5)),
            )
            return cls(centerline, width=width, name=name)
        if kind == "rounded_rectangle":
            centerline = rounded_rectangle_centerline(
                length=float(data["length"]),
                width=float(data["height"]),
                radius=float(data["radius"]),
                samples_per_corner=int(data.get("samples_per_corner", 24)),
                samples_per_straight=int(data.get("samples_per_straight", 24)),
            )
            return cls(centerline, width=width, name=name)

        raise ValueError(f"Unsupported track kind: {kind!r}")

    def project(self, point: np.ndarray, heading: float | None = None) -> TrackProjection:
        point = np.asarray(point, dtype=float)
        if point.shape != (2,):
            raise ValueError("point must have shape (2,).")

        starts = self.centerline
        rel = point - starts
        numerators = np.einsum("ij,ij->i", rel, self.segment_vectors)
        denominators = self.segment_lengths**2
        fractions = np.clip(numerators / denominators, 0.0, 1.0)
        nearest_points = starts + fractions[:, None] * self.segment_vectors
        distances_sq = np.sum((point - nearest_points) ** 2, axis=1)

        index = int(np.argmin(distances_sq))
        nearest = nearest_points[index]
        tangent = self.tangents[index]
        normal = self.normals[index]
        lateral_error = float(np.dot(point - nearest, normal))
        progress = float(
            (self.arc_lengths[index] + fractions[index] * self.segment_lengths[index]) % self.length
        )

        heading_error = None
        if heading is not None:
            track_heading = float(np.arctan2(tangent[1], tangent[0]))
            heading_error = float(wrap_angle(heading - track_heading))

        return TrackProjection(
            point=point,
            nearest=nearest,
            segment_index=index,
            segment_fraction=float(fractions[index]),
            progress=progress,
            lateral_error=lateral_error,
            tangent=tangent,
            normal=normal,
            heading_error=heading_error,
        )

    def is_off_track(self, point: np.ndarray, margin: float = 0.0) -> bool:
        projection = self.project(point)
        return abs(projection.lateral_error) > self.half_width + margin

    def progress_delta(self, previous_progress: float, current_progress: float) -> float:
        """Return signed forward progress while handling lap wrap.

        Values are mapped to the shortest signed displacement on the closed
        track. This is useful for step rewards and catches backward motion.
        """
        delta = (current_progress - previous_progress + 0.5 * self.length) % self.length
        return float(delta - 0.5 * self.length)

    def sample_at(self, progress: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        progress = progress % self.length
        index = int(np.searchsorted(self.arc_lengths, progress, side="right") - 1)
        index = min(max(index, 0), len(self.segment_lengths) - 1)
        segment_progress = progress - self.arc_lengths[index]
        fraction = segment_progress / self.segment_lengths[index]
        point = self.centerline[index] + fraction * self.segment_vectors[index]
        return point, self.tangents[index], self.normals[index]

    def boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        point_normals = self._vertex_normals()
        left = self.centerline + self.half_width * point_normals
        right = self.centerline - self.half_width * point_normals
        return left, right

    def validate_geometry(self, *, include_borders: bool = True) -> list[TrackGeometryIssue]:
        """Return self-intersection issues in centerline and generated borders."""
        curves = [("centerline", self.centerline)]
        if include_borders:
            left, right = self.boundaries()
            curves.extend([("left_border", left), ("right_border", right)])

        issues: list[TrackGeometryIssue] = []
        for curve_name, points in curves:
            for index, other_index in polyline_self_intersections(points):
                issues.append(
                    TrackGeometryIssue(
                        curve=curve_name,
                        segment_index=index,
                        other_segment_index=other_index,
                        message=(f"{curve_name} segments {index} and {other_index} intersect"),
                    )
                )
        return issues

    def _vertex_normals(self) -> np.ndarray:
        previous_tangents = np.roll(self.tangents, shift=1, axis=0)
        averaged = rotate_left(normalize_vectors(previous_tangents + self.tangents))
        return averaged


def rounded_rectangle_centerline(
    length: float,
    width: float,
    radius: float,
    samples_per_corner: int = 24,
    samples_per_straight: int = 24,
) -> np.ndarray:
    """Generate a clockwise rounded-rectangle centerline around the origin."""
    if length <= 0 or width <= 0:
        raise ValueError("length and width must be positive.")
    if radius <= 0 or radius >= min(length, width) / 2:
        raise ValueError("radius must be positive and fit inside the rectangle.")

    x = length / 2 - radius
    y = width / 2 - radius
    centers = np.array([[x, y], [x, -y], [-x, -y], [-x, y]], dtype=float)
    angle_ranges = [
        (0.0, -np.pi / 2),
        (-np.pi / 2, -np.pi),
        (-np.pi, -3 * np.pi / 2),
        (-3 * np.pi / 2, -2 * np.pi),
    ]

    points: list[np.ndarray] = []

    def add_line(start: np.ndarray, end: np.ndarray) -> None:
        for fraction in np.linspace(0.0, 1.0, samples_per_straight, endpoint=False):
            points.append(start + fraction * (end - start))

    corner_start_points = [
        centers[0] + radius * np.array([np.cos(0.0), np.sin(0.0)]),
        centers[1] + radius * np.array([np.cos(-np.pi / 2), np.sin(-np.pi / 2)]),
        centers[2] + radius * np.array([np.cos(-np.pi), np.sin(-np.pi)]),
        centers[3] + radius * np.array([np.cos(-3 * np.pi / 2), np.sin(-3 * np.pi / 2)]),
    ]

    for corner_index, (start_angle, end_angle) in enumerate(angle_ranges):
        next_corner = (corner_index + 1) % 4
        straight_start = corner_start_points[corner_index]
        straight_end = centers[next_corner] + radius * np.array(
            [np.cos(start_angle), np.sin(start_angle)]
        )
        add_line(straight_start, straight_end)

        for angle in np.linspace(start_angle, end_angle, samples_per_corner, endpoint=False):
            points.append(centers[next_corner] + radius * np.array([np.cos(angle), np.sin(angle)]))

    return np.asarray(points, dtype=float)


def catmull_rom_closed_centerline(
    control_points: np.ndarray,
    samples_per_segment: int = 24,
    alpha: float = 0.5,
) -> np.ndarray:
    """Generate a closed centripetal Catmull-Rom spline through control points."""
    points = np.asarray(control_points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("control_points must have shape (N, 2).")
    if len(points) < 4:
        raise ValueError("Catmull-Rom tracks need at least 4 control points.")
    if samples_per_segment < 2:
        raise ValueError("samples_per_segment must be at least 2.")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if np.allclose(points[0], points[-1]):
        points = points[:-1]

    samples: list[np.ndarray] = []
    for index in range(len(points)):
        p0 = points[(index - 1) % len(points)]
        p1 = points[index]
        p2 = points[(index + 1) % len(points)]
        p3 = points[(index + 2) % len(points)]
        samples.extend(
            centripetal_catmull_rom_segment(
                p0,
                p1,
                p2,
                p3,
                samples=samples_per_segment,
                alpha=alpha,
            )
        )
    return np.asarray(samples, dtype=float)


def centripetal_catmull_rom_segment(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    samples: int,
    alpha: float,
) -> list[np.ndarray]:
    def tj(ti: float, pa: np.ndarray, pb: np.ndarray) -> float:
        distance = float(np.linalg.norm(pb - pa))
        return ti + max(distance, 1e-9) ** alpha

    t0 = 0.0
    t1 = tj(t0, p0, p1)
    t2 = tj(t1, p1, p2)
    t3 = tj(t2, p2, p3)

    segment_points = []
    for t in np.linspace(t1, t2, samples, endpoint=False):
        a1 = interpolate(p0, p1, t0, t1, t)
        a2 = interpolate(p1, p2, t1, t2, t)
        a3 = interpolate(p2, p3, t2, t3, t)
        b1 = interpolate(a1, a2, t0, t2, t)
        b2 = interpolate(a2, a3, t1, t3, t)
        c = interpolate(b1, b2, t1, t2, t)
        segment_points.append(c)
    return segment_points


def interpolate(pa: np.ndarray, pb: np.ndarray, ta: float, tb: float, t: float) -> np.ndarray:
    if abs(tb - ta) <= 1e-12:
        return pa.copy()
    return ((tb - t) / (tb - ta)) * pa + ((t - ta) / (tb - ta)) * pb


def polyline_self_intersections(points: np.ndarray) -> list[tuple[int, int]]:
    """Return non-adjacent segment pairs that intersect in a closed polyline."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("points must have shape (N, 2).")
    if len(points) < 3:
        raise ValueError("points must contain at least 3 points.")

    intersections: list[tuple[int, int]] = []
    segment_count = len(points)
    for index in range(segment_count):
        start = points[index]
        end = points[(index + 1) % segment_count]
        for other_index in range(index + 1, segment_count):
            if _segments_are_adjacent(index, other_index, segment_count):
                continue
            other_start = points[other_index]
            other_end = points[(other_index + 1) % segment_count]
            if segments_intersect(start, end, other_start, other_end):
                intersections.append((index, other_index))
    return intersections


def _segments_are_adjacent(index: int, other_index: int, segment_count: int) -> bool:
    return (
        index == other_index
        or (index + 1) % segment_count == other_index
        or (other_index + 1) % segment_count == index
    )


def segments_intersect(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    first_start = np.asarray(first_start, dtype=float)
    first_end = np.asarray(first_end, dtype=float)
    second_start = np.asarray(second_start, dtype=float)
    second_end = np.asarray(second_end, dtype=float)

    first_orientation = _orientation(first_start, first_end, second_start)
    second_orientation = _orientation(first_start, first_end, second_end)
    third_orientation = _orientation(second_start, second_end, first_start)
    fourth_orientation = _orientation(second_start, second_end, first_end)

    if (
        first_orientation * second_orientation < -1e-9
        and third_orientation * fourth_orientation < -1e-9
    ):
        return True
    if abs(first_orientation) <= 1e-9 and _point_on_segment(first_start, first_end, second_start):
        return True
    if abs(second_orientation) <= 1e-9 and _point_on_segment(first_start, first_end, second_end):
        return True
    if abs(third_orientation) <= 1e-9 and _point_on_segment(second_start, second_end, first_start):
        return True
    return bool(
        abs(fourth_orientation) <= 1e-9 and _point_on_segment(second_start, second_end, first_end)
    )


def _orientation(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> float:
    vector = end - start
    relative = point - start
    return float(vector[0] * relative[1] - vector[1] * relative[0])


def _point_on_segment(start: np.ndarray, end: np.ndarray, point: np.ndarray) -> bool:
    return bool(
        min(start[0], end[0]) - 1e-9 <= point[0] <= max(start[0], end[0]) + 1e-9
        and min(start[1], end[1]) - 1e-9 <= point[1] <= max(start[1], end[1]) + 1e-9
    )
