from __future__ import annotations

import json
from pathlib import Path

from racesim.eval.physics_benchmarks import (
    acceleration_benchmark,
    braking_benchmark,
    main,
    repeatability_benchmark,
    run_physics_benchmarks,
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
        "repeatability",
    }


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


def test_repeatability_benchmark_compares_matching_rollouts() -> None:
    metrics = repeatability_benchmark("configs/env.yaml", seed=0, steps=8)

    assert metrics["same_length"]
    assert metrics["same_terminal_state"]
    assert metrics["max_abs_observation_delta"] == 0.0
    assert metrics["max_abs_reward_delta"] == 0.0
    assert metrics["deterministic"]


def test_physics_benchmarks_cli_writes_json(tmp_path: Path) -> None:
    output = tmp_path / "benchmarks.json"

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
        ]
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["seed"] == 3
    assert "acceleration" in result["benchmarks"]
