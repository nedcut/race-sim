from __future__ import annotations

from pathlib import Path

import numpy as np

from racesim.eval.evaluate_policy import evaluate_policy_model, evaluate_predict
from racesim.training.live_dashboard import LiveDashboard
from racesim.training.live_eval import best_score
from racesim.training.train_ppo import load_train_config, make_training_env
from racesim.utils.device import apply_device_to_ppo_config


class DummyPolicy:
    def predict(self, _observation, deterministic: bool = True):
        del deterministic
        return np.array([0.0, 0.2, 0.0], dtype=np.float32), None


def _constant_predict(_observation, _info):
    return np.array([0.0, 0.15, 0.0], dtype=np.float32)


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


def test_nominal_beefy_config_forces_grip_one() -> None:
    config = load_train_config(Path("configs/train_ppo_nominal_beefy_1m.yaml"))
    env = make_training_env(config)

    _observation, info = env.reset(seed=0)

    assert info["grip_scale"] == 1.0
    assert config["ppo"]["policy_kwargs"]["net_arch"]["pi"] == [256, 256, 128]
    assert config["env"]["n_envs"] == 4
    assert config["env"]["vec_env"] == "subproc"
    assert config["eval"]["save_best"] is True


def test_blind_grip_config_uses_tight_grip_window() -> None:
    config = load_train_config(Path("configs/train_ppo_blind_grip_095_105_beefy_1m.yaml"))
    env = make_training_env(config)

    _observation, info = env.reset(seed=0)

    assert 0.95 <= info["grip_scale"] <= 1.05
    assert "grip_scale" not in env.reset_options
    assert config["env"]["n_envs"] == 4


def test_best_score_prefers_completion_then_lap_time_then_reward() -> None:
    incomplete = {
        "completion_rate": 0.0,
        "mean_completed_lap_time": None,
        "mean_reward": 500.0,
    }
    slower_complete = {
        "completion_rate": 1.0,
        "mean_completed_lap_time": 42.0,
        "mean_reward": 200.0,
    }
    faster_complete = {
        "completion_rate": 1.0,
        "mean_completed_lap_time": 40.0,
        "mean_reward": 180.0,
    }

    assert best_score(slower_complete) > best_score(incomplete)
    assert best_score(faster_complete) > best_score(slower_complete)


def test_evaluate_policy_model_with_dummy_policy() -> None:
    result = evaluate_policy_model(
        model=DummyPolicy(),
        config_path=Path("configs/env.yaml"),
        episodes=1,
        max_steps=2,
        lap_target=1.0,
        deterministic=True,
        record_trajectory=True,
        reset_options={"grip_scale": 0.95},
    )

    assert result["controller"] == "ppo"
    assert result["reset_options"]["grip_scale"] == 0.95
    assert len(result["episodes"][0]["trajectory"]) == 2


def test_evaluate_predict_callable() -> None:
    result = evaluate_predict(
        predict=_constant_predict,
        config_path=Path("configs/env.yaml"),
        episodes=1,
        max_steps=3,
        lap_target=1.0,
        record_trajectory=False,
        controller_name="custom",
    )
    assert result["controller"] == "custom"
    assert result["summary"]["episodes"] == 1


def test_beefy_train_configs_use_auto_device() -> None:
    for path in (
        Path("configs/train_ppo_nominal_beefy_1m.yaml"),
        Path("configs/train_ppo_blind_grip_095_105_beefy_1m.yaml"),
    ):
        config = load_train_config(path)
        assert config["ppo"]["device"] == "auto"
        resolved = apply_device_to_ppo_config(dict(config["ppo"]))
        assert resolved["device"] in {"cpu", "cuda", "mps"}


def test_live_dashboard_waiting_update_does_not_crash(tmp_path) -> None:
    dashboard = LiveDashboard(tmp_path / "missing.json")

    dashboard.update(0)

    assert dashboard.current is None
