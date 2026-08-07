from __future__ import annotations

import json
from pathlib import Path

from racesim.eval.physics_benchmarks import (
    acceleration_benchmark,
    braking_benchmark,
    main,
    render_markdown,
    repeatability_benchmark,
    run_physics_benchmarks,
    skidpad_benchmark,
    steady_turning_benchmark,
)


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
