from __future__ import annotations

from dataclasses import replace

import pytest

from racesim.env.racing_env import RacingEnv


def _gated_env(spacing_m: float = 10.0) -> RacingEnv:
    env = RacingEnv("configs/env.yaml")
    env.reward_config = replace(
        env.reward_config,
        progress_gate=1.0,
        progress_gate_spacing=spacing_m,
    )
    env.reset(seed=0)
    return env


def test_oscillation_does_not_inflate_progress_or_complete_a_lap() -> None:
    env = _gated_env(spacing_m=5.0)
    section = 4.0
    gates = 0
    peak = 0.0

    for _ in range(200):
        gates += env._apply_progress_delta(section)
        peak = max(peak, env.cumulative_forward_progress)
        gates += env._apply_progress_delta(-section)

    assert env.cumulative_forward_progress == pytest.approx(0.0)
    assert peak == pytest.approx(section)
    assert peak < env.track.length
    assert not env._lap_complete()
    assert gates == 0


def test_oscillation_does_not_keep_firing_progress_gates() -> None:
    env = _gated_env(spacing_m=10.0)

    first_crossing = env._apply_progress_delta(12.0)
    assert first_crossing == 1

    extra_gates = 0
    for _ in range(80):
        extra_gates += env._apply_progress_delta(-6.0)
        extra_gates += env._apply_progress_delta(6.0)

    assert extra_gates == 0
    assert env.cumulative_forward_progress == pytest.approx(12.0)
    assert not env._lap_complete()


def test_net_forward_track_length_completes_a_lap() -> None:
    env = RacingEnv("configs/env.yaml")
    env.reset(seed=0)

    remaining = env.track.length
    while remaining > 0.0:
        chunk = min(3.0, remaining)
        env._apply_progress_delta(chunk)
        remaining -= chunk

    assert env.cumulative_forward_progress == pytest.approx(env.track.length)
    assert env._lap_complete()


def test_backward_motion_reduces_cumulative_progress_to_zero() -> None:
    env = RacingEnv("configs/env.yaml")
    env.reset(seed=0)

    env._apply_progress_delta(7.5)
    assert env.cumulative_forward_progress == pytest.approx(7.5)

    env._apply_progress_delta(-3.0)
    assert env.cumulative_forward_progress == pytest.approx(4.5)

    env._apply_progress_delta(-10.0)
    assert env.cumulative_forward_progress == pytest.approx(0.0)
    assert not env._lap_complete()
