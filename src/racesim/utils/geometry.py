from __future__ import annotations

import math

import numpy as np


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-pi, pi)."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("Cannot normalize zero-length vector.")
    return vectors / norms


def rotate_left(vectors: np.ndarray) -> np.ndarray:
    rotated = np.empty_like(vectors, dtype=float)
    rotated[..., 0] = -vectors[..., 1]
    rotated[..., 1] = vectors[..., 0]
    return rotated
