"""Gymnasium environment registration for RaceSim."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from racesim.env.racing_env import RacingEnv
from racesim.paths import default_env_config, resolve_resource

ENV_ID = "RaceSim-v0"
_REGISTERED = False


def resolve_env_config(
    config: str | Path | None = None,
    config_path: str | Path | None = None,
    **_kwargs: Any,
) -> Path:
    """Resolve the env YAML path from kwargs or ``RACESIM_CONFIG``.

    Precedence: ``config`` / ``config_path`` kwargs, then ``RACESIM_CONFIG``,
    then the packaged/default oval env config.
    """
    path = config if config is not None else config_path
    if path is None:
        path = os.environ.get("RACESIM_CONFIG")
    if path is None:
        return default_env_config()
    return resolve_resource(path)


def make_race_sim_env(
    config: str | Path | None = None,
    config_path: str | Path | None = None,
    render_mode: str | None = None,
    **kwargs: Any,
) -> RacingEnv:
    """Factory used by ``gymnasium.make("RaceSim-v0", ...)``."""
    # Drop unknown kwargs Gymnasium may pass without failing older callers.
    del kwargs
    return RacingEnv(
        resolve_env_config(config=config, config_path=config_path),
        render_mode=render_mode,
    )


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
