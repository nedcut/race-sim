from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import racesim  # noqa: F401
from racesim.env.registration import ENV_ID, make_race_sim_env, resolve_env_config


def test_resolve_env_config_defaults() -> None:
    assert resolve_env_config() == Path("configs/env.yaml")


def test_resolve_env_config_from_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RACESIM_CONFIG", "configs/env_s_curve.yaml")
    assert resolve_env_config() == Path("configs/env_s_curve.yaml")


def test_resolve_env_config_kwargs_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RACESIM_CONFIG", "configs/env_s_curve.yaml")
    assert resolve_env_config(config="configs/env_hairpin.yaml") == Path(
        "configs/env_hairpin.yaml"
    )


def test_gym_make_race_sim_v0_reset_and_step() -> None:
    env = gym.make(ENV_ID, config="configs/env.yaml")
    try:
        observation, info = env.reset(seed=0)
        assert observation.shape == (19,)
        assert "progress" in info

        observation, reward, terminated, truncated, info = env.step(
            np.array([0.0, 0.3, 0.0], dtype=np.float32)
        )
        assert observation.shape == (19,)
        assert isinstance(reward, float)
        assert not terminated
        assert not truncated
        assert info["progress"] > 0.0
    finally:
        env.close()


def test_make_race_sim_env_factory() -> None:
    env = make_race_sim_env(config="configs/env.yaml")
    observation, _info = env.reset(seed=0)
    assert env.observation_space.contains(observation)


def test_import_racesim_registers_env() -> None:
    assert racesim.__version__
    spec = gym.spec(ENV_ID)
    assert spec.id == ENV_ID
