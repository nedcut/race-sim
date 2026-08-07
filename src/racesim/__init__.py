from __future__ import annotations

from racesim.env.racing_env import RacingEnv
from racesim.env.registration import ENV_ID, make_race_sim_env, register_racesim_envs
from racesim.env.track import ClosedTrack
from racesim.paths import default_env_config, project_root, resolve_resource
from racesim.version import __version__

__all__ = [
    "ClosedTrack",
    "ENV_ID",
    "RacingEnv",
    "__version__",
    "default_env_config",
    "make_race_sim_env",
    "project_root",
    "register_racesim_envs",
    "resolve_resource",
]

register_racesim_envs()
