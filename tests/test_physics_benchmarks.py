from __future__ import annotations

import json
from pathlib import Path

import pytest

from racesim.eval.physics_benchmarks import (
    acceleration_benchmark,
    braking_benchmark,
    default_physics_config,
    main,
    make_open_loop_env,
    render_markdown,
    repeatability_benchmark,
    resolve_physics_config,
    run_physics_benchmarks,
    skidpad_benchmark,
    soft_range_flags,
    steady_turning_benchmark,
)
from racesim.paths import default_benchmark_pad_config


def test_physics_benchmarks_return_json_friendly_metrics() -> None:
    result = run_physics_benchmarks(
        config_path="configs/env.yaml",
        seed=11,
        acceleration_steps=3,
        braking_steps=3,
        turning_steps=3,
        repeatability_steps=3,
    )

    json.dumps(result)
    assert set(result["benchmarks"]) == {
        "acceleration",
        "braking",
        "steady_turning",
        "skidpad",
        "step_steer",
        "slalom",
        "braking_turn",
        "throttle_exit",
        "repeatability",
    }
    assert isinstance(result["sanity_flags"], list)


def test_acceleration_benchmark_reports_forward_speed_gain() -> None:
    metrics = acceleration_benchmark("configs/env.yaml", seed=0, steps=8)

    assert metrics["steps"] == 8
    assert metrics["final_speed_mps"] > metrics["initial_speed_mps"]
    assert metrics["average_acceleration_mps2"] > 0.0
    assert metrics["distance_m"] > 0.0


def test_braking_benchmark_reports_deceleration() -> None:
    metrics = braking_benchmark("configs/env.yaml", seed=0, steps=8, initial_speed=5.0)

    assert metrics["steps"] >= 1
    assert metrics["final_speed_mps"] < metrics["initial_speed_mps"]
    assert metrics["average_deceleration_mps2"] > 0.0
    assert metrics["braking_distance_m"] > 0.0


def test_steady_turning_benchmark_reports_yaw_response() -> None:
    metrics = steady_turning_benchmark("configs/env.yaml", seed=0, steps=8)

    assert metrics["steps"] == 8
    assert metrics["mean_abs_steady_yaw_rate_radps"] > 0.0
    assert metrics["estimated_turn_radius_m"] is None or metrics["estimated_turn_radius_m"] > 0.0


def test_skidpad_benchmark_reports_tire_usage() -> None:
    metrics = skidpad_benchmark("configs/env.yaml", seed=0, steps=8)

    assert metrics["steps"] == 8
    assert metrics["peak_tire_usage"] >= 0.0
    assert "peak_abs_slip_angle_rad" in metrics


def test_repeatability_benchmark_compares_matching_rollouts() -> None:
    metrics = repeatability_benchmark("configs/env.yaml", seed=0, steps=8)

    assert metrics["same_length"]
    assert metrics["same_terminal_state"]
    assert metrics["max_abs_observation_delta"] == 0.0
    assert metrics["max_abs_reward_delta"] == 0.0
    assert metrics["deterministic"]


def test_physics_benchmarks_cli_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "benchmarks.json"
    markdown_output = tmp_path / "benchmarks.md"

    main(
        [
            "--config",
            "configs/env.yaml",
            "--seed",
            "3",
            "--acceleration-steps",
            "2",
            "--braking-steps",
            "2",
            "--turning-steps",
            "2",
            "--repeatability-steps",
            "2",
            "--output",
            str(output),
            "--markdown-output",
            str(markdown_output),
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["seed"] == 3
    assert "acceleration" in result["benchmarks"]
    assert "RaceSim Physics Benchmarks" in markdown_output.read_text(encoding="utf-8")


def test_physics_benchmark_markdown_includes_sanity_flags() -> None:
    result = run_physics_benchmarks(
        config_path="configs/env.yaml",
        seed=2,
        benchmarks=["acceleration"],
        acceleration_steps=2,
    )

    markdown = render_markdown(result)

    assert "## Sanity Flags" in markdown
    assert "| acceleration |" in markdown


def test_benchmark_env_off_track_respects_disabled_termination() -> None:
    """Open-loop env inflates off_track_margin; info must match termination logic."""
    result = run_physics_benchmarks(
        config_path="configs/env.yaml",
        seed=0,
        benchmarks=["acceleration"],
        acceleration_steps=12,
    )
    metrics = result["benchmarks"]["acceleration"]
    assert metrics["off_track"] is False
    assert not any("off-track" in flag for flag in result["sanity_flags"])


def test_default_physics_config_prefers_benchmark_pad() -> None:
    pad = default_benchmark_pad_config()
    assert pad.exists()
    assert default_physics_config() == pad
    assert resolve_physics_config(None) == pad
    assert resolve_physics_config("configs/env.yaml").name == "env.yaml"


def test_make_open_loop_env_defaults_to_pad() -> None:
    env = make_open_loop_env(None, max_steps=4)
    assert env.off_track_margin == pytest.approx(1e6)
    result = run_physics_benchmarks(
        seed=0,
        benchmarks=["acceleration"],
        acceleration_steps=2,
    )
    assert Path(result["config"]).resolve() == default_benchmark_pad_config().resolve()


def test_physics_benchmarks_cli_enforce_exits_on_flags(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "benchmarks.json"

    def fake_run(**_kwargs):
        return {
            "config": "configs/env_benchmark_pad.yaml",
            "seed": 0,
            "benchmarks": {"acceleration": {"steps": 1}},
            "sanity_flags": ["acceleration did not increase speed"],
        }

    monkeypatch.setattr(
        "racesim.eval.physics_benchmarks.run_physics_benchmarks",
        fake_run,
    )
    with pytest.raises(SystemExit) as exc_info:
        main(["--output", str(output), "--enforce", "--benchmarks", "acceleration"])
    assert exc_info.value.code == 1


def test_physics_benchmarks_cli_without_enforce_keeps_exit_zero(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "benchmarks.json"

    def fake_run(**_kwargs):
        return {
            "config": "configs/env_benchmark_pad.yaml",
            "seed": 0,
            "benchmarks": {"acceleration": {"steps": 1}},
            "sanity_flags": ["acceleration did not increase speed"],
        }

    monkeypatch.setattr(
        "racesim.eval.physics_benchmarks.run_physics_benchmarks",
        fake_run,
    )
    main(["--output", str(output), "--benchmarks", "acceleration"])
    assert output.exists()


def test_soft_range_flags_detect_out_of_band_accel() -> None:
    flags = soft_range_flags(
        {"acceleration": {"average_acceleration_mps2": 0.01}},
    )
    assert flags
    assert "outside" in flags[0]

    clean = soft_range_flags(
        {"acceleration": {"average_acceleration_mps2": 5.0}},
    )
    assert clean == []
