from __future__ import annotations

import numpy as np
import pytest

from racesim.training.curriculum_env import TrackCurriculumEnv, normalize_probabilities


def test_curriculum_env_resets_and_steps() -> None:
    env = TrackCurriculumEnv(["configs/env.yaml"], randomize_reset=True)

    observation, info = env.reset(seed=0)
    next_observation, reward, terminated, truncated, next_info = env.step(
        np.array([0.0, 0.2, 0.0], dtype=np.float32)
    )

    assert observation.shape == env.observation_space.shape
    assert next_observation.shape == env.observation_space.shape
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert info["track_name"] == "oval"
    assert next_info["track_name"] == "oval"


def test_normalize_probabilities_validates_shape() -> None:
    with pytest.raises(ValueError):
        normalize_probabilities([1.0, 2.0], size=1)

    np.testing.assert_allclose(normalize_probabilities([1.0, 3.0], size=2), [0.25, 0.75])


def test_curriculum_env_can_fix_reset_options() -> None:
    env = TrackCurriculumEnv(
        ["configs/env.yaml"],
        randomize_reset=True,
        reset_options={"grip_scale": 1.0},
    )

    _observation, info = env.reset(seed=0)

    assert info["grip_scale"] == 1.0


def test_curriculum_env_can_sample_reset_option_ranges() -> None:
    env = TrackCurriculumEnv(
        ["configs/env.yaml"],
        randomize_reset=True,
        reset_option_ranges={"grip_scale": [0.95, 1.05]},
    )

    _observation, info = env.reset(seed=0)

    assert 0.95 <= info["grip_scale"] <= 1.05


def test_same_seed_matches_when_a_new_track_first_appears() -> None:
    configs = ["configs/env.yaml", "configs/env_s_curve.yaml"]
    probabilities = [0.55, 0.45]
    n_resets = 40

    def rollout() -> tuple[list[int], list[np.ndarray]]:
        env = TrackCurriculumEnv(
            configs,
            randomize_reset=True,
            probabilities=probabilities,
        )
        track_indexes: list[int] = []
        observations: list[np.ndarray] = []
        observation, info = env.reset(seed=7)
        track_indexes.append(int(info["track_index"]))
        observations.append(np.asarray(observation, dtype=float).copy())
        for _ in range(n_resets - 1):
            observation, info = env.reset(seed=None)
            track_indexes.append(int(info["track_index"]))
            observations.append(np.asarray(observation, dtype=float).copy())
        return track_indexes, observations

    tracks_a, observations_a = rollout()
    tracks_b, observations_b = rollout()

    assert tracks_a == tracks_b
    assert set(tracks_a) == {0, 1}
    assert len(observations_a) == n_resets
    for observation_a, observation_b in zip(observations_a, observations_b, strict=True):
        np.testing.assert_allclose(observation_a, observation_b)
