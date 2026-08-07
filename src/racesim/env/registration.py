"""Gymnasium environment registration for RaceSim."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from racesim.env.racing_env import RacingEnv

DEFAULT_ENV_CONFIG = "configs/env.yaml"
ENV_ID = "RaceSim-v0"
_REGISTERED = False


def resolve_env_config(
    config: str | Path | None = None,
    config_path: str | Path | None = None,
    **_kwargs: Any,
) -> Path:
    """Resolve the env YAML path from kwargs or ``RACESIM_CONFIG``.

    Precedence: ``config`` / ``config_path`` kwargs, then ``RACESIM_CONFIG``,
    then ``configs/env.yaml``.
    """
    path = config if config is not None else config_path
    if path is None:
        path = os.environ.get("RACESIM_CONFIG", DEFAULT_ENV_CONFIG)
    return Path(path)


def make_race_sim_env(
    config: str | Path | None = None,
    config_path: str | Path | None = None,
    **kwargs: Any,
) -> RacingEnv:
    """Factory used by ``gymnasium.make("RaceSim-v0", ...)``.

    Extra kwargs are reserved for future Gym API compatibility and ignored
    unless they provide a config path.
    """
    del kwargs  # Gymnasium may pass render_mode etc.; RacingEnv has no render yet.
    return RacingEnv(resolve_env_config(config=config, config_path=config_path))


def register_racesim_envs() -> None:
    """Register RaceSim-v0 with Gymnasium (idempotent)."""
    global _REGISTERED
    if _REGISTERED:
        return

    import gymnasium as gym

    if ENV_ID not in gym.envs.registry:
        gym.register(
            id=ENV_ID,
            entry_point="racesim.env.registration:make_race_sim_env",
        )
    _REGISTERED = True
