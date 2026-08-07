"""Resource and project-root path resolution for RaceSim."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Packaged data lives next to the Python package when installed as a wheel.
_PACKAGE_DATA = Path(__file__).resolve().parent / "_data"


@lru_cache(maxsize=1)
def project_root() -> Path:
    """Return the workspace root or packaged data root for configs/assets.

    Resolution order:
    1. ``RACESIM_ROOT`` environment variable
    2. Walk upward from CWD for a directory containing ``configs/`` and ``assets/``
    3. Walk upward from this package for a development checkout
    4. Packaged ``racesim/_data`` directory (wheel installs)
    """
    env_root = os.environ.get("RACESIM_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _looks_like_root(candidate):
            return candidate

    for start in (Path.cwd(), Path(__file__).resolve().parent):
        root = _find_root_upwards(start)
        if root is not None:
            return root

    if _looks_like_root(_PACKAGE_DATA):
        return _PACKAGE_DATA

    # Last resort for editable installs that still keep assets at repo root.
    repo_from_src = Path(__file__).resolve().parents[2]
    if _looks_like_root(repo_from_src):
        return repo_from_src

    return Path.cwd()


def _looks_like_root(path: Path) -> bool:
    return (path / "configs").is_dir() and (path / "assets").is_dir()


def _find_root_upwards(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if _looks_like_root(candidate):
            return candidate
    return None


def resolve_resource(path: str | Path) -> Path:
    """Resolve a config/asset path against ``project_root()`` when needed."""
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    if candidate.exists():
        return candidate.resolve()
    packaged = project_root() / candidate
    if packaged.exists():
        return packaged.resolve()
    return candidate


def default_env_config() -> Path:
    return resolve_resource("configs/env.yaml")


def default_benchmark_pad_config() -> Path:
    pad = resolve_resource("configs/env_benchmark_pad.yaml")
    if pad.exists():
        return pad
    return default_env_config()


def default_track_catalog() -> Path:
    return resolve_resource("configs/track_catalog.yaml")


def default_train_config() -> Path:
    return resolve_resource("configs/train_ppo_oval.yaml")
