"""Shared hashing, schema, and dependency metadata for training artifacts."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from racesim.env.racing_env import RacingEnv, observation_feature_count
from racesim.paths import resolve_resource
from racesim.version import __version__


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def array_digest(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.tobytes())
        digest.update(str(array.shape).encode("utf-8"))
        digest.update(str(array.dtype).encode("utf-8"))
    return digest.hexdigest()


def git_sha() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    sha = completed.stdout.strip()
    return sha or None


def dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {
        "python": sys.version.split()[0],
        "numpy": getattr(np, "__version__", None),
        "racesim": __version__,
        "torch": None,
        "stable_baselines3": None,
    }
    try:
        import torch

        versions["torch"] = getattr(torch, "__version__", None)
    except ImportError:
        pass
    try:
        import stable_baselines3

        versions["stable_baselines3"] = getattr(stable_baselines3, "__version__", None)
    except ImportError:
        pass
    return versions


def hash_env_config_tree(config_path: str | Path) -> dict[str, str]:
    """SHA-256 of an env YAML plus referenced vehicle/track files when present."""
    path = resolve_resource(config_path)
    hashes = {str(path): sha256_file(path)}
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    for key in ("vehicle", "track"):
        ref = payload.get(key)
        if not ref:
            continue
        try:
            resolved = resolve_resource(ref)
        except TypeError:
            continue
        if resolved.exists() and resolved.is_file():
            hashes[str(resolved)] = sha256_file(resolved)
    return hashes


def observation_action_schema(env: RacingEnv | Any) -> dict[str, Any]:
    racing_env = _as_racing_env(env)
    observation_config = racing_env.observation_config
    return {
        "observation_dim": int(racing_env.observation_space.shape[0]),
        "action_dim": int(racing_env.action_space.shape[0]),
        "action_low": np.asarray(racing_env.action_space.low, dtype=np.float32).tolist(),
        "action_high": np.asarray(racing_env.action_space.high, dtype=np.float32).tolist(),
        "include_prev_action": bool(observation_config.include_prev_action),
        "include_grip_scale": bool(observation_config.include_grip_scale),
        "include_tire_usage": bool(observation_config.include_tire_usage),
        "lookahead_distances": [float(v) for v in observation_config.lookahead_distances],
        "curvature_scale": float(observation_config.curvature_scale),
        "prev_action_source": (
            "smoothed_action" if observation_config.include_prev_action else None
        ),
        "feature_count": observation_feature_count(observation_config),
    }


def _as_racing_env(env: Any) -> RacingEnv:
    if isinstance(env, RacingEnv):
        return env
    current = getattr(env, "current_env", None)
    if isinstance(current, RacingEnv):
        return current
    nested = getattr(env, "envs", None)
    if nested:
        return _as_racing_env(nested[0])
    raise TypeError(f"Expected RacingEnv or curriculum wrapper, got {type(env)!r}")


def schema_mismatch_errors(
    dataset_schema: dict[str, Any],
    env_schema: dict[str, Any],
) -> list[str]:
    errors: list[str] = []
    for key in (
        "observation_dim",
        "action_dim",
        "include_prev_action",
        "include_grip_scale",
        "include_tire_usage",
        "lookahead_distances",
        "prev_action_source",
    ):
        if key in dataset_schema and dataset_schema[key] != env_schema.get(key):
            errors.append(f"{key}: dataset={dataset_schema[key]!r} env={env_schema.get(key)!r}")
    if "action_low" in dataset_schema and "action_high" in dataset_schema:
        if not np.allclose(dataset_schema["action_low"], env_schema["action_low"]):
            errors.append("action_low mismatch")
        if not np.allclose(dataset_schema["action_high"], env_schema["action_high"]):
            errors.append("action_high mismatch")
    return errors
