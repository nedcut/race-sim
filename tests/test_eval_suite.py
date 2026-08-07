from __future__ import annotations

from racesim.eval.suite import QUICK_PROFILE, render_markdown, resolve_profile, smoke_table


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
