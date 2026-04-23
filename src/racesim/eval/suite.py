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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the project evaluation suite.")
    parser.add_argument("--catalog", type=Path, default=Path("configs/track_catalog.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/baselines.md"))
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument("--smoke-steps", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-max-steps", type=int, default=3000)
    parser.add_argument("--policy-model", type=Path, default=DEFAULT_POLICY_MODEL)
    parser.add_argument("--policy-episodes", type=int, default=3)
    parser.add_argument("--policy-max-steps", type=int, default=2000)
    parser.add_argument("--skip-gates", action="store_true")
    parser.add_argument("--skip-policy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_eval_suite(
        catalog_path=args.catalog,
        smoke_steps=args.smoke_steps,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        policy_model=args.policy_model,
        policy_episodes=args.policy_episodes,
        policy_max_steps=args.policy_max_steps,
        run_gates=not args.skip_gates,
        run_policy=not args.skip_policy,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(result), encoding="utf-8")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote {args.output}")


def run_eval_suite(
    catalog_path: Path = Path("configs/track_catalog.yaml"),
    smoke_steps: int = 5000,
    eval_episodes: int = 5,
    eval_max_steps: int = 3000,
    policy_model: Path = DEFAULT_POLICY_MODEL,
    policy_episodes: int = 3,
    policy_max_steps: int = 2000,
    run_gates: bool = True,
    run_policy: bool = True,
) -> dict[str, Any]:
    gates = run_quality_gates(catalog_path) if run_gates else []
    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))["tracks"]

    controller_eval = {
        "racing_line": evaluate(
            config_path="configs/env.yaml",
            controller_name="racing_line",
            episodes=eval_episodes,
            max_steps=eval_max_steps,
            seed=0,
        )["summary"],
        "heuristic": evaluate(
            config_path="configs/env.yaml",
            controller_name="heuristic",
            episodes=eval_episodes,
            max_steps=eval_max_steps,
            seed=0,
        )["summary"],
    }
    smoke = {
        "racing_line": smoke_tracks(catalog_path, "racing_line", smoke_steps, lap_target=1.0),
        "heuristic": smoke_tracks(catalog_path, "heuristic", smoke_steps, lap_target=1.0),
    }
    physics = run_physics_benchmarks(config_path="configs/env.yaml")
    policy = run_policy_matrix(
        catalog=catalog,
        policy_model=policy_model,
        episodes=policy_episodes,
        max_steps=policy_max_steps,
        enabled=run_policy,
    )

    return {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "catalog": str(catalog_path),
        "quality_gates": [gate.__dict__ for gate in gates],
        "controller_eval": controller_eval,
        "smoke_steps": smoke_steps,
        "catalog_smoke": smoke,
        "physics_benchmarks": physics["benchmarks"],
        "physics_threshold_policy": "telemetry_only",
        "policy_model": str(policy_model),
        "policy_matrix": policy,
        "failure_cases": failure_cases(smoke, policy),
    }


def run_quality_gates(catalog_path: Path) -> list[GateResult]:
    return [
        run_gate("pytest", "python -m pytest"),
        run_gate("ruff", "python -m ruff check ."),
        run_gate(
            "track validation",
            f"python -m racesim.scripts.validate_tracks --catalog {catalog_path}",
        ),
    ]


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


def failure_cases(smoke: dict[str, list[dict]], policy: dict[str, Any]) -> list[dict[str, str]]:
    cases = []
    for controller, rows in smoke.items():
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

    lines.extend(
        [
            "",
            f"## Catalog Smoke ({result['smoke_steps']} Steps)",
            "",
            "### Racing Line",
            "",
        ]
    )
    lines.extend(smoke_table(result["catalog_smoke"]["racing_line"]))
    lines.extend(["", "### Heuristic", ""])
    lines.extend(smoke_table(result["catalog_smoke"]["heuristic"]))

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
            "These are telemetry baselines, not pass/fail thresholds yet.",
            "",
        ]
    )
    physics = result["physics_benchmarks"]
    acceleration = physics["acceleration"]
    braking = physics["braking"]
    turning = physics["steady_turning"]
    repeatability = physics["repeatability"]
    lines.extend(
        [
            f"- acceleration: final `{acceleration['final_speed_mps']:.2f} m/s`, "
            f"avg `{acceleration['average_acceleration_mps2']:.2f} m/s^2`, "
            f"off-track `{acceleration['off_track']}`",
            f"- braking: stopped `{braking['stopped']}`, distance "
            f"`{braking['braking_distance_m']:.2f} m`, avg decel "
            f"`{braking['average_deceleration_mps2']:.2f} m/s^2`",
            f"- steady turning: yaw rate `{turning['mean_abs_steady_yaw_rate_radps']:.3f} rad/s`, "
            f"radius `{turning['estimated_turn_radius_m']:.1f} m`, "
            f"off-track `{turning['off_track']}`",
            f"- repeatability: deterministic `{repeatability['deterministic']}`, "
            f"max obs delta `{repeatability['max_abs_observation_delta']}`",
        ]
    )

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
            "used to catch regressions during review.",
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


def format_optional(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


if __name__ == "__main__":
    main()
