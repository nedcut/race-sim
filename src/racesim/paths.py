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
    1. ``RACESIM_ROOT`` if it is a strict RaceSim root
    2. Walk upward from this package file (development checkout)
    3. Packaged ``racesim/_data`` directory (wheel installs)
    4. Current working directory only if it is a strict RaceSim root

    A strict root contains ``configs/env.yaml`` and ``assets/mjcf/world.xml``,
    not merely directories named ``configs`` and ``assets``.
    """
    env_root = os.environ.get("RACESIM_ROOT")
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if _is_strict_root(candidate):
            return candidate

    package_root = _find_root_upwards(Path(__file__).resolve().parent)
    if package_root is not None:
        return package_root

    if _is_strict_root(_PACKAGE_DATA):
        return _PACKAGE_DATA

    cwd = Path.cwd().resolve()
    if _is_strict_root(cwd):
        return cwd

    if _PACKAGE_DATA.is_dir():
        return _PACKAGE_DATA
    return cwd


def _is_strict_root(path: Path) -> bool:
    return (path / "configs" / "env.yaml").is_file() and (
        path / "assets" / "mjcf" / "world.xml"
    ).is_file()


def _find_root_upwards(start: Path) -> Path | None:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if _is_strict_root(candidate):
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
