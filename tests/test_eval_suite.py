from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from racesim.eval.suite import (
    QUICK_PROFILE,
    main,
    quality_gates_failed,
    render_markdown,
    resolve_profile,
    run_gate,
    run_quality_gates,
    smoke_table,
)


class _Args:
    def __init__(self, **kwargs):
        defaults = {
            "quick": False,
            "profile": None,
            "smoke_steps": None,
            "eval_episodes": None,
            "eval_max_steps": None,
            "policy_episodes": None,
            "policy_max_steps": None,
            "skip_gates": False,
            "skip_policy": False,
        }
        defaults.update(kwargs)
        self.__dict__.update(defaults)


def test_smoke_table_formats_rows() -> None:
    rows = [
        {
            "track": "oval",
            "status": "complete",
            "steps": 10,
            "lap_fraction": 1.0,
            "speed": 2.5,
        }
    ]

    table = smoke_table(rows)

    assert "| oval | complete | 10 | 1.00 | 2.50 |" in table


def test_quick_profile_skips_pytest_and_policy() -> None:
    profile = resolve_profile(_Args(quick=True))

    assert profile.name == "quick"
    assert profile.run_pytest is False
    assert profile.run_policy is False
    assert profile.run_catalog_smoke is False
    assert profile.controllers == ("racing_line",)
    assert profile.eval_episodes == 1
    assert profile.eval_max_steps <= 500
    assert QUICK_PROFILE.name == "quick"


def test_profile_quick_alias() -> None:
    profile = resolve_profile(_Args(profile="quick"))
    assert profile.name == "quick"


def test_render_markdown_includes_threshold_decision() -> None:
    markdown = render_markdown(
        {
            "generated_at": "2026-04-23T00:00:00+00:00",
            "profile": "full",
            "catalog": "configs/track_catalog.yaml",
            "quality_gates": [],
            "controller_eval": {
                "racing_line": {
                    "completion_rate": 1.0,
                    "off_track_rate": 0.0,
                    "mean_completed_lap_time": 70.4,
                    "best_completed_lap_time": 70.4,
                },
                "heuristic": {
                    "completion_rate": 1.0,
                    "off_track_rate": 0.0,
                    "mean_completed_lap_time": 72.4,
                    "best_completed_lap_time": 72.4,
                },
            },
            "smoke_steps": 1,
            "catalog_smoke": {
                "racing_line": [
                    {
                        "track": "oval",
                        "status": "complete",
                        "steps": 1,
                        "lap_fraction": 1.0,
                        "speed": 2.5,
                    }
                ],
                "heuristic": [
                    {
                        "track": "oval",
                        "status": "complete",
                        "steps": 1,
                        "lap_fraction": 1.0,
                        "speed": 2.5,
                    }
                ],
            },
            "policy_model": "results/model.zip",
            "policy_matrix": {"status": "skipped", "reason": "missing model"},
            "physics_benchmarks": {
                "acceleration": {
                    "final_speed_mps": 1.0,
                    "average_acceleration_mps2": 0.5,
                    "off_track": False,
                },
                "braking": {
                    "stopped": True,
                    "braking_distance_m": 2.0,
                    "average_deceleration_mps2": 1.0,
                },
                "steady_turning": {
                    "mean_abs_steady_yaw_rate_radps": 0.1,
                    "estimated_turn_radius_m": 10.0,
                    "off_track": False,
                },
                "repeatability": {
                    "deterministic": True,
                    "max_abs_observation_delta": 0.0,
                },
            },
            "physics_sanity_flags": [],
            "physics_config": "configs/env_benchmark_pad.yaml",
            "failure_cases": [],
        }
    )

    assert "Physics benchmarks stay telemetry-only" in markdown
    assert "| acceleration |" in markdown
    assert "## Failure Cases" in markdown
    assert "Profile: `full`" in markdown
    assert "racesim-physics-benchmarks --enforce" in markdown


def test_render_markdown_marks_skipped_catalog_smoke() -> None:
    markdown = render_markdown(
        {
            "generated_at": "2026-04-23T00:00:00+00:00",
            "profile": "quick",
            "catalog": "configs/track_catalog.yaml",
            "quality_gates": [],
            "controller_eval": {
                "racing_line": {
                    "completion_rate": 1.0,
                    "off_track_rate": 0.0,
                    "mean_completed_lap_time": None,
                    "best_completed_lap_time": None,
                },
            },
            "smoke_steps": 200,
            "catalog_smoke": {"racing_line": [], "heuristic": [], "status": "skipped"},
            "policy_model": "results/model.zip",
            "policy_matrix": {"status": "skipped", "reason": "disabled"},
            "physics_benchmarks": {},
            "physics_sanity_flags": [],
            "failure_cases": [],
        }
    )

    assert "skipped (quick profile)" in markdown


def test_run_gate_uses_argv_list_not_shell(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def fake_run(args, **kwargs):
        seen["args"] = args
        seen["shell"] = kwargs.get("shell")
        return subprocess.CompletedProcess(args, 0, stdout="ok\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_gate("demo", [sys.executable, "-c", "print(1)"])

    assert isinstance(seen["args"], list)
    assert not isinstance(seen["args"], str)
    assert seen["shell"] is False
    assert result.passed
    assert isinstance(result.command, str)


def test_run_gate_rejects_shell_string() -> None:
    with pytest.raises(TypeError, match="argv sequence"):
        run_gate("ruff", "python -m ruff check .")  # type: ignore[arg-type]


def test_run_gate_failure_is_recorded() -> None:
    result = run_gate("failing", [sys.executable, "-c", "raise SystemExit(2)"])
    assert result.passed is False


def test_run_quality_gates_passes_catalog_as_argv_element(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[str, list[str]]] = []

    def fake_run_gate(name: str, argv):
        calls.append((name, list(argv)))
        return type("Gate", (), {"name": name, "command": " ".join(argv), "passed": True})()

    monkeypatch.setattr("racesim.eval.suite.run_gate", fake_run_gate)
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text("tracks: {}\n", encoding="utf-8")
    run_quality_gates(catalog, run_pytest=True)

    assert calls
    for _name, argv in calls:
        assert isinstance(argv, list)
        assert argv[0] == sys.executable
        assert all(isinstance(part, str) for part in argv)

    track_argv = next(argv for name, argv in calls if name == "track validation")
    assert "--catalog" in track_argv
    assert str(catalog) in track_argv
    assert track_argv[track_argv.index("--catalog") + 1] == str(catalog)


def test_failed_quality_gate_fails_the_suite() -> None:
    passed = {"quality_gates": [{"name": "ruff", "passed": True}]}
    failed = {"quality_gates": [{"name": "ruff", "passed": False}]}
    skipped = {"quality_gates": []}

    assert quality_gates_failed(passed) is False
    assert quality_gates_failed(failed) is True
    assert quality_gates_failed(skipped) is False


def test_main_exits_nonzero_when_quality_gate_fails(monkeypatch, tmp_path: Path) -> None:
    output = tmp_path / "baselines.md"
    result = {
        "generated_at": "2026-04-23T00:00:00+00:00",
        "profile": "quick",
        "catalog": "configs/track_catalog.yaml",
        "quality_gates": [
            {"name": "ruff", "command": "ruff check .", "passed": False, "output_tail": "boom"}
        ],
        "controller_eval": {
            "racing_line": {
                "completion_rate": 1.0,
                "off_track_rate": 0.0,
                "mean_completed_lap_time": None,
                "best_completed_lap_time": None,
            }
        },
        "smoke_steps": 200,
        "catalog_smoke": {"racing_line": [], "heuristic": [], "status": "skipped"},
        "policy_model": "results/model.zip",
        "policy_matrix": {"status": "skipped", "reason": "disabled"},
        "physics_benchmarks": {},
        "physics_sanity_flags": [],
        "failure_cases": [],
    }
    monkeypatch.setattr(
        sys,
        "argv",
        ["racesim-eval-suite", "--output", str(output), "--quick", "--skip-gates"],
    )
    monkeypatch.setattr("racesim.eval.suite.run_eval_suite", lambda **_kwargs: result)

    with pytest.raises(SystemExit) as excinfo:
        main()

    assert excinfo.value.code == 1
    markdown = output.read_text(encoding="utf-8")
    assert "ruff: fail" in markdown
