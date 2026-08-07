from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from racesim.eval.evaluate import evaluate
from racesim.eval.evaluate_policy import evaluate_policy_model
from racesim.eval.physics_benchmarks import run_physics_benchmarks
from racesim.eval.smoke_tracks import smoke_tracks

DEFAULT_POLICY_MODEL = Path("results/ppo_blind_grip_095_105_beefy_1m/best_model.zip")


@dataclass(frozen=True)
class GateResult:
    name: str
    command: str
    passed: bool
    output_tail: str


@dataclass(frozen=True)
class SuiteProfile:
    name: str
    smoke_steps: int
    eval_episodes: int
    eval_max_steps: int
    policy_episodes: int
    policy_max_steps: int
    run_gates: bool
    run_pytest: bool
    run_policy: bool
    run_catalog_smoke: bool
    controllers: tuple[str, ...]
    physics_acceleration_steps: int
    physics_braking_steps: int
    physics_turning_steps: int
    physics_maneuver_steps: int
    physics_repeatability_steps: int
    physics_benchmarks: tuple[str, ...] | None


FULL_PROFILE = SuiteProfile(
    name="full",
    smoke_steps=5000,
    eval_episodes=5,
    eval_max_steps=3000,
    policy_episodes=3,
    policy_max_steps=2000,
    run_gates=True,
    run_pytest=True,
    run_policy=True,
    run_catalog_smoke=True,
    controllers=("racing_line", "heuristic"),
    physics_acceleration_steps=120,
    physics_braking_steps=120,
    physics_turning_steps=160,
    physics_maneuver_steps=180,
    physics_repeatability_steps=80,
    physics_benchmarks=None,
)

# Target: complete in under ~2 minutes on a typical laptop (no PPO matrix).
QUICK_PROFILE = SuiteProfile(
    name="quick",
    smoke_steps=200,
    eval_episodes=1,
    eval_max_steps=400,
    policy_episodes=1,
    policy_max_steps=200,
    run_gates=True,
    run_pytest=False,
    run_policy=False,
    run_catalog_smoke=False,
    controllers=("racing_line",),
    physics_acceleration_steps=40,
    physics_braking_steps=40,
    physics_turning_steps=40,
    physics_maneuver_steps=40,
    physics_repeatability_steps=20,
    physics_benchmarks=("acceleration", "braking", "steady_turning", "repeatability"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project evaluation suite.")
    parser.add_argument("--catalog", type=Path, default=Path("configs/track_catalog.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/baselines.md"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast profile: lint + track validation + short physics + short racing_line eval.",
    )
    parser.add_argument(
        "--profile",
        choices=("full", "quick"),
        default=None,
        help="Named suite profile. --quick is an alias for --profile quick.",
    )
    parser.add_argument("--smoke-steps", type=int, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-max-steps", type=int, default=None)
    parser.add_argument("--policy-model", type=Path, default=DEFAULT_POLICY_MODEL)
    parser.add_argument("--policy-episodes", type=int, default=None)
    parser.add_argument("--policy-max-steps", type=int, default=None)
    parser.add_argument("--skip-gates", action="store_true")
    parser.add_argument("--skip-policy", action="store_true")
    return parser.parse_args()


def resolve_profile(args: argparse.Namespace) -> SuiteProfile:
    name = args.profile or ("quick" if args.quick else "full")
    base = QUICK_PROFILE if name == "quick" else FULL_PROFILE
    return SuiteProfile(
        name=base.name,
        smoke_steps=args.smoke_steps if args.smoke_steps is not None else base.smoke_steps,
        eval_episodes=args.eval_episodes if args.eval_episodes is not None else base.eval_episodes,
        eval_max_steps=args.eval_max_steps
        if args.eval_max_steps is not None
        else base.eval_max_steps,
        policy_episodes=args.policy_episodes
        if args.policy_episodes is not None
        else base.policy_episodes,
        policy_max_steps=args.policy_max_steps
        if args.policy_max_steps is not None
        else base.policy_max_steps,
        run_gates=base.run_gates and not args.skip_gates,
        run_pytest=base.run_pytest,
        run_policy=base.run_policy and not args.skip_policy,
        run_catalog_smoke=base.run_catalog_smoke,
        controllers=base.controllers,
        physics_acceleration_steps=base.physics_acceleration_steps,
        physics_braking_steps=base.physics_braking_steps,
        physics_turning_steps=base.physics_turning_steps,
        physics_maneuver_steps=base.physics_maneuver_steps,
        physics_repeatability_steps=base.physics_repeatability_steps,
        physics_benchmarks=base.physics_benchmarks,
    )


def main() -> None:
    args = parse_args()
    profile = resolve_profile(args)
    result = run_eval_suite(
        catalog_path=args.catalog,
        policy_model=args.policy_model,
        profile=profile,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(result), encoding="utf-8")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


def run_eval_suite(
    catalog_path: Path = Path("configs/track_catalog.yaml"),
    smoke_steps: int | None = None,
    eval_episodes: int | None = None,
    eval_max_steps: int | None = None,
    policy_model: Path = DEFAULT_POLICY_MODEL,
    policy_episodes: int | None = None,
    policy_max_steps: int | None = None,
    run_gates: bool | None = None,
    run_policy: bool | None = None,
    profile: SuiteProfile | None = None,
) -> dict[str, Any]:
    active = profile or FULL_PROFILE
    if smoke_steps is not None:
        active = SuiteProfile(**{**active.__dict__, "smoke_steps": smoke_steps})
    if eval_episodes is not None:
        active = SuiteProfile(**{**active.__dict__, "eval_episodes": eval_episodes})
    if eval_max_steps is not None:
        active = SuiteProfile(**{**active.__dict__, "eval_max_steps": eval_max_steps})
    if policy_episodes is not None:
        active = SuiteProfile(**{**active.__dict__, "policy_episodes": policy_episodes})
    if policy_max_steps is not None:
        active = SuiteProfile(**{**active.__dict__, "policy_max_steps": policy_max_steps})
    if run_gates is not None:
        active = SuiteProfile(**{**active.__dict__, "run_gates": run_gates})
    if run_policy is not None:
        active = SuiteProfile(**{**active.__dict__, "run_policy": run_policy})

    gates = (
        run_quality_gates(catalog_path, run_pytest=active.run_pytest) if active.run_gates else []
    )
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["tracks"]

    controller_eval = {
        controller: evaluate(
            config_path="configs/env.yaml",
            controller_name=controller,
            episodes=active.eval_episodes,
            max_steps=active.eval_max_steps,
            seed=0,
        )["summary"]
        for controller in active.controllers
    }
    if active.run_catalog_smoke:
        smoke = {
            "racing_line": smoke_tracks(
                catalog_path, "racing_line", active.smoke_steps, lap_target=1.0
            ),
            "heuristic": smoke_tracks(
                catalog_path, "heuristic", active.smoke_steps, lap_target=1.0
            ),
        }
    else:
        smoke = {"racing_line": [], "heuristic": [], "status": "skipped"}

    physics_kwargs: dict[str, Any] = {
        "acceleration_steps": active.physics_acceleration_steps,
        "braking_steps": active.physics_braking_steps,
        "turning_steps": active.physics_turning_steps,
        "maneuver_steps": active.physics_maneuver_steps,
        "repeatability_steps": active.physics_repeatability_steps,
    }
    if active.physics_benchmarks is not None:
        physics_kwargs["benchmarks"] = active.physics_benchmarks
    physics = run_physics_benchmarks(**physics_kwargs)
    policy = run_policy_matrix(
        catalog=catalog,
        policy_model=policy_model,
        episodes=active.policy_episodes,
        max_steps=active.policy_max_steps,
        enabled=active.run_policy,
    )

    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "profile": active.name,
        "catalog": str(catalog_path),
        "quality_gates": [gate.__dict__ for gate in gates],
        "controller_eval": controller_eval,
        "smoke_steps": active.smoke_steps,
        "catalog_smoke": smoke,
        "physics_benchmarks": physics["benchmarks"],
        "physics_sanity_flags": physics["sanity_flags"],
        "physics_config": physics["config"],
        "physics_threshold_policy": "telemetry_only",
        "policy_model": str(policy_model),
        "policy_matrix": policy,
        "failure_cases": failure_cases(smoke, policy),
    }


def run_quality_gates(catalog_path: Path, run_pytest: bool = True) -> list[GateResult]:
    gates: list[GateResult] = []
    if run_pytest:
        gates.append(run_gate("pytest", "python -m pytest"))
    gates.extend(
        [
            run_gate("ruff", "python -m ruff check ."),
            run_gate(
                "track validation",
                f"python -m racesim.scripts.validate_tracks --catalog {catalog_path}",
            ),
        ]
    )
    return gates


def run_gate(name: str, command: str) -> GateResult:
    completed = subprocess.run(
        command,
        shell=True,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return GateResult(
        name=name,
        command=command,
        passed=completed.returncode == 0,
        output_tail="\n".join(completed.stdout.strip().splitlines()[-8:]),
    )


def run_policy_matrix(
    catalog: dict[str, dict[str, str]],
    policy_model: Path,
    episodes: int,
    max_steps: int,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled:
        return {"status": "skipped", "reason": "disabled"}
    if not policy_model.exists():
        return {"status": "skipped", "reason": f"missing model: {policy_model}"}

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        return {"status": "skipped", "reason": f"missing stable-baselines3: {exc}"}

    model = PPO.load(policy_model)
    rows = []
    for track_name, entry in catalog.items():
        result = evaluate_policy_model(
            model=model,
            config_path=Path(entry["env"]),
            episodes=episodes,
            max_steps=max_steps,
            lap_target=1.0,
            deterministic=True,
            record_trajectory=False,
        )
        rows.append({"track": track_name, **result["summary"]})
    return {"status": "ok", "episodes": episodes, "rows": rows}


def failure_cases(smoke: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, str]]:
    cases = []
    for controller, rows in smoke.items():
        if controller == "status" or not isinstance(rows, list):
            continue
        for row in rows:
            if row["status"] != "complete":
                cases.append(
                    {
                        "track": row["track"],
                        "surface": f"{controller} controller",
                        "status": row["status"],
                        "note": f"reached lap {row['lap_fraction']:.2f}",
                    }
                )

    if policy.get("status") == "ok":
        for row in policy["rows"]:
            if row["completion_rate"] < 1.0:
                cases.append(
                    {
                        "track": row["track"],
                        "surface": "ppo policy",
                        "status": "incomplete",
                        "note": (
                            f"completion {row['completion_rate']:.2f}, "
                            f"off-track {row['off_track_rate']:.2f}"
                        ),
                    }
                )
    return cases


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# RaceSim Baselines",
        "",
        f"Generated: `{result['generated_at']}`",
        f"Profile: `{result.get('profile', 'full')}`",
        f"Catalog: `{result['catalog']}`",
        "",
        "## Quality Gates",
        "",
    ]
    if result["quality_gates"]:
        lines.extend(
            f"- {gate['name']}: {'pass' if gate['passed'] else 'fail'}"
            for gate in result["quality_gates"]
        )
    else:
        lines.append("- skipped")

    lines.extend(
        [
            "",
            "## Default Oval Eval",
            "",
            "| controller | completion | off-track | mean lap time | best lap time |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for controller, summary in result["controller_eval"].items():
        lines.append(summary_row(controller, summary))

    smoke = result["catalog_smoke"]
    if smoke.get("status") == "skipped":
        lines.extend(
            [
                "",
                f"## Catalog Smoke ({result['smoke_steps']} Steps)",
                "",
                "- skipped (quick profile)",
            ]
        )
    else:
        lines.extend(
            [
                "",
                f"## Catalog Smoke ({result['smoke_steps']} Steps)",
                "",
                "### Racing Line",
                "",
            ]
        )
        lines.extend(smoke_table(smoke["racing_line"]))
        lines.extend(["", "### Heuristic", ""])
        lines.extend(smoke_table(smoke["heuristic"]))

    lines.extend(["", "## PPO Policy Matrix", ""])
    policy = result["policy_matrix"]
    if policy["status"] == "ok":
        lines.extend(
            [
                f"Model: `{result['policy_model']}`",
                "",
                "| track | completion | off-track | mean lap time |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for row in policy["rows"]:
            lines.append(
                f"| {row['track']} | {row['completion_rate']:.2f} | "
                f"{row['off_track_rate']:.2f} | {format_optional(row['mean_completed_lap_time'])} |"
            )
    else:
        lines.append(f"- skipped: {policy['reason']}")

    lines.extend(
        [
            "",
            "## Physics Benchmarks",
            "",
            f"Config: `{result.get('physics_config', 'configs/env.yaml')}`",
            "",
            "These are telemetry baselines, not pass/fail thresholds yet.",
            "",
        ]
    )
    physics = result["physics_benchmarks"]
    lines.extend(
        [
            "| benchmark | steps | final speed | yaw rate | tire usage | notes |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for name, metrics in physics.items():
        lines.append(physics_row(name, metrics))

    lines.extend(["", "### Sanity Flags", ""])
    flags = result.get("physics_sanity_flags", [])
    if flags:
        lines.extend(f"- {flag}" for flag in flags)
    else:
        lines.append("- none")

    lines.extend(["", "## Failure Cases", ""])
    if result["failure_cases"]:
        for case in result["failure_cases"]:
            lines.append(
                f"- `{case['track']}` via {case['surface']}: {case['status']} ({case['note']})"
            )
    else:
        lines.append("- none in this suite run")

    lines.extend(
        [
            "",
            "## Threshold Decision",
            "",
            "Physics benchmarks stay telemetry-only for now. The only hard gates are tests, lint, "
            "and catalog geometry validation; controller and policy results are baseline numbers "
            "used to catch regressions during review. Use `racesim-physics-benchmarks --enforce` "
            "when optional sanity-flag gating is desired.",
            "",
        ]
    )
    return "\n".join(lines)


def summary_row(name: str, summary: dict[str, Any]) -> str:
    return (
        f"| {name} | {summary['completion_rate']:.2f} | {summary['off_track_rate']:.2f} | "
        f"{format_optional(summary['mean_completed_lap_time'])} | "
        f"{format_optional(summary['best_completed_lap_time'])} |"
    )


def smoke_table(rows: list[dict]) -> list[str]:
    lines = [
        "| track | status | steps | lap fraction | speed |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['track']} | {row['status']} | {row['steps']} | "
            f"{row['lap_fraction']:.2f} | {row['speed']:.2f} |"
        )
    return lines


def physics_row(name: str, metrics: dict[str, Any]) -> str:
    final_speed = metrics.get("final_speed_mps")
    yaw_rate = metrics.get("mean_abs_yaw_rate_radps") or metrics.get(
        "mean_abs_steady_yaw_rate_radps"
    )
    tire_usage = metrics.get("peak_tire_usage")
    notes = []
    if metrics.get("off_track"):
        notes.append("off-track")
    if metrics.get("terminated"):
        notes.append("terminated")
    if metrics.get("deterministic") is not None:
        notes.append(f"deterministic={metrics['deterministic']}")
    if metrics.get("stopped") is not None:
        notes.append(f"stopped={metrics['stopped']}")
    return (
        f"| {name} | {metrics.get('steps', metrics.get('first_steps', 0))} | "
        f"{format_optional(final_speed)} | {format_optional(yaw_rate)} | "
        f"{format_optional(tire_usage)} | {', '.join(notes) or '-'} |"
    )


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


if __name__ == "__main__":
    main()
