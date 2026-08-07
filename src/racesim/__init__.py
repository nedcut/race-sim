"""Simplified racing simulation tools."""

from racesim.env.registration import register_racesim_envs

__all__ = ["__version__", "register_racesim_envs"]

__version__ = "0.1.0"

register_racesim_envs()
