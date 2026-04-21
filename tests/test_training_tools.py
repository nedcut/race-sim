from __future__ import annotations

from pathlib import Path

import numpy as np

from racesim.eval.evaluate_policy import evaluate_policy_model
from racesim.training.live_dashboard import LiveDashboard
from racesim.training.train_ppo import load_train_config, make_training_env


class DummyPolicy:
    def predict(self, _observation, deterministic: bool = True):
        del deterministic
        return np.array([0.0, 0.2, 0.0], dtype=np.float32), None


def test_train_config_builds_curriculum_env() -> None:
    config = load_train_config(Path("configs/train_ppo_oval.yaml"))
    env = make_training_env(config)

    observation, info = env.reset(seed=0)

    assert observation.shape == env.observation_space.shape
    assert info["track_name"] == "oval"


def test_staged_progress_config_builds_curriculum_env() -> None:
    config = load_train_config(Path("configs/train_ppo_staged_progress_1m.yaml"))
    env = make_training_env(config)

    assert len(config["curriculum"]["stages"]) == 2
    assert len(env.envs) == 5
    np.testing.assert_allclose(env.probabilities.sum(), 1.0)


def test_evaluate_policy_model_with_dummy_policy() -> None:
    result = evaluate_policy_model(
        model=DummyPolicy(),
        config_path=Path("configs/env.yaml"),
        episodes=1,
        max_steps=2,
        lap_target=1.0,
        deterministic=True,
        record_trajectory=True,
    )

    assert result["controller"] == "ppo"
    assert len(result["episodes"][0]["trajectory"]) == 2


def test_live_dashboard_waiting_update_does_not_crash(tmp_path) -> None:
    dashboard = LiveDashboard(tmp_path / "missing.json")

    dashboard.update(0)

    assert dashboard.current is None
