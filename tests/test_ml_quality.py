from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from racesim.env.racing_env import RacingEnv, observation_feature_count, yaw_to_quat
from racesim.training.agent_bundle import (
    AgentBundleError,
    load_agent_bundle,
    normalize_enabled,
    save_agent_bundle,
)
from racesim.training.curriculum_env import TrackCurriculumEnv
from racesim.training.live_eval import LiveEvalCallback
from racesim.training.schedules import parse_schedule, prepare_ppo_schedules
from racesim.training.seeds import (
    DEFAULT_FINAL_EVAL_SEED_BASE,
    DEFAULT_LIVE_EVAL_SEED_BASE,
    DEFAULT_TRAIN_SEED,
)
from racesim.training.train_ppo import (
    load_train_config,
    make_training_env,
    resolve_init_mode,
    timestep_to_callback_calls,
)


def test_parse_linear_schedule_anneals_to_zero() -> None:
    schedule = parse_schedule("linear_3e-4", name="learning_rate")
    assert callable(schedule)
    assert schedule(1.0) == pytest.approx(3e-4)
    assert schedule(0.5) == pytest.approx(1.5e-4)
    assert schedule(0.0) == pytest.approx(0.0)


def test_prepare_ppo_schedules_leaves_constants() -> None:
    prepared = prepare_ppo_schedules({"learning_rate": 1e-3, "clip_range": 0.2, "n_steps": 64})
    assert prepared["learning_rate"] == 1e-3
    assert prepared["clip_range"] == 0.2
    assert prepared["n_steps"] == 64


def test_quality_env_has_extended_observation() -> None:
    env = RacingEnv("configs/env_rl_quality.yaml")
    expected = observation_feature_count(env.observation_config)
    assert expected == 25
    assert env.observation_space.shape == (25,)

    observation, _info = env.reset(seed=0)
    assert observation.shape == (25,)

    action = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    next_obs, _reward, _term, _trunc, info = env.step(action)
    assert next_obs.shape == (25,)
    np.testing.assert_allclose(next_obs[19:22], env.smoothed_action, atol=1e-5)
    assert not np.allclose(next_obs[19:22], action, atol=1e-3)
    np.testing.assert_allclose(info["smoothed_action"], env.smoothed_action)
    assert "lap_complete" in info["reward_terms"]
    assert "tire_usage" in info["reward_terms"]
    assert "action_rate" in info["reward_terms"]
    assert info["reward_terms"]["action_rate"] <= 0.0
    env.close()


def test_unknown_reward_key_raises(tmp_path: Path) -> None:
    text = Path("configs/env.yaml").read_text(encoding="utf-8")
    text = text.replace("off_track: 35.0", "off_track: 35.0\n  lap_complet: 100.0")
    path = tmp_path / "typo.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="lap_complet"):
        RacingEnv(path)


def test_nonfinite_reward_weight_raises(tmp_path: Path) -> None:
    text = Path("configs/env.yaml").read_text(encoding="utf-8")
    text = text.replace("progress: 1.0", "progress: .nan")
    path = tmp_path / "nan.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        RacingEnv(path)


def test_classic_env_yaml_still_loads() -> None:
    env = RacingEnv("configs/env.yaml")
    assert env.schema_version == 1
    assert env.reward_config.lap_complete == 0.0
    env.close()


def test_lap_complete_bonus_fires() -> None:
    env = RacingEnv("configs/env_rl_quality.yaml")
    env.reset(seed=0)
    env.reward_config = replace(
        env.reward_config,
        lap_complete=50.0,
        off_track=0.0,
        progress=0.0,
        time=0.0,
        lateral_error=0.0,
        heading_error=0.0,
        boundary_margin=0.0,
        progress_gate=0.0,
        speed_excess=0.0,
        no_progress=0.0,
        tire_usage=0.0,
        action_rate=0.0,
    )
    point, tangent, _normal = env.track.sample_at(env.track.length - 0.2)
    heading = float(np.arctan2(tangent[1], tangent[0]))
    quat = yaw_to_quat(heading)
    env.data.qpos[env.root_qpos_adr : env.root_qpos_adr + 3] = [point[0], point[1], 0.32]
    env.data.qpos[env.root_qpos_adr + 3 : env.root_qpos_adr + 7] = quat
    env.data.qvel[env.root_dof_adr : env.root_dof_adr + 3] = 8.0 * np.array(
        [tangent[0], tangent[1], 0.0]
    )
    mujoco.mj_forward(env.model, env.data)
    env.previous_progress = env.track.length - 0.2
    env.cumulative_forward_progress = env.track.length - 0.2

    crossed = False
    for _ in range(40):
        _obs, reward, terminated, _trunc, info = env.step(
            np.array([0.0, 1.0, 0.0], dtype=np.float32)
        )
        if info["lap_complete"]:
            crossed = True
            assert info["reward_terms"]["lap_complete"] == pytest.approx(50.0)
            assert terminated
            assert reward >= 49.0
            break
    assert crossed, "expected lap complete within throttle rollout near finish"
    env.close()


def test_quality_train_config_builds_env() -> None:
    config = load_train_config(Path("configs/train_ppo_quality.yaml"))
    env = make_training_env(config)
    observation, info = env.reset(seed=0)
    assert observation.shape == (25,)
    assert info["track_name"] == "oval"
    assert config["normalize"]["enabled"] is True
    assert str(config["ppo"]["learning_rate"]).startswith("linear")
    assert config["seed"] == DEFAULT_TRAIN_SEED
    assert config["eval"]["seed_base"] == DEFAULT_LIVE_EVAL_SEED_BASE
    assert config["eval"]["seed_base"] != config["seed"]
    assert DEFAULT_FINAL_EVAL_SEED_BASE != config["seed"]
    assert DEFAULT_FINAL_EVAL_SEED_BASE != config["eval"]["seed_base"]
    env.close()


def test_curriculum_rejects_mismatched_observation_shapes() -> None:
    with pytest.raises(ValueError, match="matching observation"):
        TrackCurriculumEnv(["configs/env.yaml", "configs/env_rl_quality.yaml"])


def test_classic_env_still_19d() -> None:
    env = RacingEnv("configs/env.yaml")
    assert env.observation_space.shape == (19,)
    env.close()


def test_collect_expert_dry_shape() -> None:
    from racesim.training.expert_data import collect_expert

    result = collect_expert(
        config_path="configs/env.yaml",
        controller_name="racing_line",
        episodes=1,
        max_steps=5,
        seed=0,
    )
    assert result["observations"].shape[0] >= 1
    assert result["observations"].shape[1] == 19
    assert result["actions"].shape[1] == 3
    assert "observation_action_schema" in result["metadata"]
    assert "dataset_digest" in result["metadata"]
    assert "env_config_hashes" in result["metadata"]


def test_expert_collector_horizon_marks_last_step_done() -> None:
    from racesim.training.expert_data import collect_expert

    result = collect_expert(
        config_path="configs/env.yaml",
        controller_name="racing_line",
        episodes=1,
        max_steps=5,
        seed=0,
    )
    assert result["dones"].shape == (5,)
    assert not result["dones"][:-1].any()
    assert bool(result["dones"][-1]) is True


def test_pretrained_and_resume_are_distinct() -> None:
    assert resolve_init_mode({}) == "fresh"
    assert resolve_init_mode({"pretrained": "results/bc/bc_quality"}) == "pretrained"
    assert resolve_init_mode({"resume": "results/ppo/agent_bundle"}) == "resume"
    with pytest.raises(ValueError, match="only one"):
        resolve_init_mode({"pretrained": "a.zip", "resume": "b.zip"})
    assert normalize_enabled({"normalize": {"enabled": True}})
    assert not normalize_enabled({"normalize": {"enabled": False}})


def test_dry_run_does_not_write_train_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from racesim.training import train_ppo

    config = yaml.safe_load(
        Path("configs/train_ppo_quality_smoke.yaml").read_text(encoding="utf-8")
    )
    output_dir = tmp_path / "out"
    config["output_dir"] = str(output_dir)
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["train_ppo", "--config", str(cfg_path), "--dry-run"])
    train_ppo.main()
    assert not (output_dir / "train_config.json").exists()
    assert not (output_dir / "agent_bundle").exists()


def test_bundle_manifest_requires_normalizer_when_normalize_enabled(tmp_path: Path) -> None:
    model = tmp_path / "model.zip"
    model.write_bytes(b"not-a-real-sb3-zip")
    schema = {"observation_dim": 25, "action_dim": 3}
    train_config = {"normalize": {"enabled": True}, "output_dir": str(tmp_path)}
    with pytest.raises(AgentBundleError, match="VecNormalize"):
        save_agent_bundle(
            tmp_path / "bundle_missing",
            model_path=model,
            vecnormalize_path=None,
            train_config=train_config,
            observation_action_schema=schema,
        )

    stats = tmp_path / "vecnormalize.pkl"
    stats.write_bytes(b"dummy-vecnormalize")
    bundle_dir = save_agent_bundle(
        tmp_path / "bundle_ok",
        model_path=model,
        vecnormalize_path=stats,
        train_config=train_config,
        observation_action_schema=schema,
        timesteps=128,
        seed=1000,
        init_mode="fresh",
    )
    loaded = load_agent_bundle(bundle_dir)
    assert loaded.vecnormalize_path is not None
    assert loaded.manifest["normalize_enabled"] is True
    assert "sha256" in loaded.manifest["files"]["vecnormalize.pkl"]

    loaded.vecnormalize_path.unlink()
    with pytest.raises(AgentBundleError, match="vecnormalize.pkl is missing"):
        load_agent_bundle(bundle_dir)


def test_callback_frequencies_are_timestep_based() -> None:
    assert timestep_to_callback_calls(50_000, n_envs=4) == 12_500
    assert timestep_to_callback_calls(50_000, n_envs=1) == 50_000
    callback = LiveEvalCallback(
        eval_env_config="configs/env.yaml",
        output_dir=Path("results/unused"),
        eval_freq=25_000,
        episodes=1,
        max_steps=10,
        enabled=False,
    )
    assert callback.next_eval_timestep == 25_000
    assert callback.next_eval_timestep != 1


def test_bc_rejects_mismatched_dataset_metadata() -> None:
    from racesim.training.bc_pretrain import assert_expert_matches_train_env

    config = load_train_config(Path("configs/train_ppo_quality.yaml"))
    env = make_training_env(config)
    try:
        metadata = {
            "observation_action_schema": {
                "observation_dim": 19,
                "action_dim": 3,
                "include_prev_action": False,
                "include_grip_scale": False,
                "include_tire_usage": False,
                "lookahead_distances": [6.0, 12.0, 24.0, 40.0],
                "prev_action_source": None,
                "action_low": [-1.0, 0.0, 0.0],
                "action_high": [1.0, 1.0, 1.0],
            }
        }
        with pytest.raises(ValueError, match="schema does not match"):
            assert_expert_matches_train_env(metadata, env, config)
    finally:
        env.close()


def test_evaluate_policy_cli_seed_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from racesim.eval.evaluate_policy import parse_args

    monkeypatch.setattr(sys, "argv", ["evaluate_policy", "--predict-demo"])
    args = parse_args()
    assert args.seed_base == DEFAULT_FINAL_EVAL_SEED_BASE
    assert args.bundle is None
