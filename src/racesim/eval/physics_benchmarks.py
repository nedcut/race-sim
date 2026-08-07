from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from racesim.env.racing_env import RacingEnv

BenchmarkName = str

DEFAULT_BENCHMARKS: tuple[BenchmarkName, ...] = (
    "acceleration",
    "braking",
    "steady_turning",
    "skidpad",
    "step_steer",
    "slalom",
    "braking_turn",
    "throttle_exit",
    "repeatability",
)

BENCHMARK_PAD_CONFIG = Path("configs/env_benchmark_pad.yaml")
FALLBACK_PHYSICS_CONFIG = Path("configs/env.yaml")


def default_physics_config() -> Path:
    """Prefer the wide open-loop pad when present; keep oval env as fallback."""
    if BENCHMARK_PAD_CONFIG.exists():
        return BENCHMARK_PAD_CONFIG
    return FALLBACK_PHYSICS_CONFIG


def resolve_physics_config(config_path: str | Path | None = None) -> Path:
    if config_path is None:
        return default_physics_config()
    return Path(config_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-loop physics quality benchmarks for RacingEnv."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Env config for open-loop runs. Defaults to configs/env_benchmark_pad.yaml when "
            "present, otherwise configs/env.yaml."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--benchmarks",
        nargs="+",
        choices=DEFAULT_BENCHMARKS,
        default=list(DEFAULT_BENCHMARKS),
    )
    parser.add_argument("--acceleration-steps", type=int, default=120)
    parser.add_argument("--braking-steps", type=int, default=120)
    parser.add_argument("--turning-steps", type=int, default=160)
    parser.add_argument("--maneuver-steps", type=int, default=180)
    parser.add_argument("--repeatability-steps", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("results/physics_benchmarks.json"))
    parser.add_argument("--markdown-output", type=Path, default=None)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit with code 1 when any sanity_flags are raised.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    result = run_physics_benchmarks(
        config_path=args.config,
        seed=args.seed,
        benchmarks=args.benchmarks,
        acceleration_steps=args.acceleration_steps,
        braking_steps=args.braking_steps,
        turning_steps=args.turning_steps,
        maneuver_steps=args.maneuver_steps,
        repeatability_steps=args.repeatability_steps,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result["benchmarks"], indent=2))
    print(f"Wrote {args.output}")
    if args.enforce and result["sanity_flags"]:
        for flag in result["sanity_flags"]:
            print(f"sanity flag: {flag}")
        raise SystemExit(1)


def run_physics_benchmarks(
    config_path: str | Path | None = None,
    seed: int = 0,
    benchmarks: list[BenchmarkName] | tuple[BenchmarkName, ...] = DEFAULT_BENCHMARKS,
    acceleration_steps: int = 120,
    braking_steps: int = 120,
    turning_steps: int = 160,
    maneuver_steps: int = 180,
    repeatability_steps: int = 80,
) -> dict[str, Any]:
    config_path = resolve_physics_config(config_path)
    benchmark_set = set(benchmarks)
    unknown = benchmark_set.difference(DEFAULT_BENCHMARKS)
    if unknown:
        raise ValueError(f"Unknown physics benchmarks: {sorted(unknown)}")

    results: dict[str, Any] = {}
    if "acceleration" in benchmark_set:
        results["acceleration"] = acceleration_benchmark(config_path, seed, acceleration_steps)
    if "braking" in benchmark_set:
        results["braking"] = braking_benchmark(config_path, seed + 1, braking_steps)
    if "steady_turning" in benchmark_set:
        results["steady_turning"] = steady_turning_benchmark(
            config_path,
            seed + 2,
            turning_steps,
        )
    if "skidpad" in benchmark_set:
        results["skidpad"] = skidpad_benchmark(config_path, seed + 3, maneuver_steps)
    if "step_steer" in benchmark_set:
        results["step_steer"] = step_steer_benchmark(config_path, seed + 4, maneuver_steps)
    if "slalom" in benchmark_set:
        results["slalom"] = slalom_benchmark(config_path, seed + 5, maneuver_steps)
    if "braking_turn" in benchmark_set:
        results["braking_turn"] = braking_turn_benchmark(config_path, seed + 6, maneuver_steps)
    if "throttle_exit" in benchmark_set:
        results["throttle_exit"] = throttle_exit_benchmark(
            config_path,
            seed + 7,
            maneuver_steps,
        )
    if "repeatability" in benchmark_set:
        results["repeatability"] = repeatability_benchmark(
            config_path,
            seed + 8,
            repeatability_steps,
        )

    return {
        "config": str(config_path),
        "seed": int(seed),
        "benchmarks": results,
        "sanity_flags": sanity_flags(results),
    }


def acceleration_benchmark(
    config_path: str | Path,
    seed: int = 0,
    steps: int = 120,
    throttle: float = 1.0,
) -> dict[str, Any]:
    env = make_open_loop_env(config_path, steps)
    dt = step_dt(env)
    _observation, info = env.reset(seed=seed, options={"initial_speed": 0.0})

    previous_position = np.asarray(info["position"], dtype=float)
    speeds = [float(info["speed"])]
    distance = 0.0
    final_info = info
    terminated = False
    truncated = False
    step_count = 0
    action = np.array([0.0, throttle, 0.0], dtype=np.float32)

    for step in range(1, steps + 1):
        _observation, _reward, terminated, truncated, final_info = env.step(action)
        position = np.asarray(final_info["position"], dtype=float)
        distance += float(np.linalg.norm(position - previous_position))
        previous_position = position
        speeds.append(float(final_info["speed"]))
        step_count = step
        if terminated or truncated:
            break

    return {
        "steps": int(step_count),
        "sim_time_s": float(step_count * dt),
        "initial_speed_mps": float(speeds[0]),
        "final_speed_mps": float(speeds[-1]),
        "peak_speed_mps": float(max(speeds)),
        "distance_m": float(distance),
        "average_acceleration_mps2": rate(speeds[-1] - speeds[0], step_count * dt),
        "peak_step_acceleration_mps2": peak_positive_rate(speeds, dt),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "off_track": bool(final_info["off_track"]),
        "lap_complete": bool(final_info["lap_complete"]),
    }


def braking_benchmark(
    config_path: str | Path,
    seed: int = 0,
    steps: int = 120,
    initial_speed: float = 9.0,
    stop_speed: float = 0.25,
) -> dict[str, Any]:
    env = make_open_loop_env(config_path, steps)
    dt = step_dt(env)
    _observation, info = env.reset(seed=seed, options={"initial_speed": initial_speed})

    previous_position = np.asarray(info["position"], dtype=float)
    speeds = [float(info["speed"])]
    distance = 0.0
    final_info = info
    terminated = False
    truncated = False
    step_count = 0
    action = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    for step in range(1, steps + 1):
        _observation, _reward, terminated, truncated, final_info = env.step(action)
        position = np.asarray(final_info["position"], dtype=float)
        distance += float(np.linalg.norm(position - previous_position))
        previous_position = position
        speed = float(final_info["speed"])
        speeds.append(speed)
        step_count = step
        if speed <= stop_speed or terminated or truncated:
            break

    stopped = speeds[-1] <= stop_speed
    return {
        "steps": int(step_count),
        "sim_time_s": float(step_count * dt),
        "initial_speed_mps": float(speeds[0]),
        "final_speed_mps": float(speeds[-1]),
        "stop_speed_mps": float(stop_speed),
        "stopped": bool(stopped),
        "braking_distance_m": float(distance),
        "average_deceleration_mps2": rate(speeds[0] - speeds[-1], step_count * dt),
        "peak_step_deceleration_mps2": peak_positive_rate([-speed for speed in speeds], dt),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "off_track": bool(final_info["off_track"]),
        "lap_complete": bool(final_info["lap_complete"]),
    }


def steady_turning_benchmark(
    config_path: str | Path,
    seed: int = 0,
    steps: int = 160,
    steering: float = 0.45,
    throttle: float = 0.28,
    initial_speed: float = 5.0,
) -> dict[str, Any]:
    env = make_open_loop_env(config_path, steps)
    dt = step_dt(env)
    _observation, info = env.reset(seed=seed, options={"initial_speed": initial_speed})

    rows: list[dict[str, float]] = [telemetry_from_info(info)]
    final_info = info
    terminated = False
    truncated = False
    step_count = 0
    action = np.array([steering, throttle, 0.0], dtype=np.float32)

    for step in range(1, steps + 1):
        _observation, _reward, terminated, truncated, final_info = env.step(action)
        rows.append(telemetry_from_info(final_info))
        step_count = step
        if terminated or truncated:
            break

    steady_rows = rows[max(0, len(rows) // 2) :]
    speeds = np.array([row["speed"] for row in steady_rows], dtype=float)
    yaw_rates = np.array([row["yaw_rate"] for row in steady_rows], dtype=float)
    lateral_speeds = np.array([row["lateral_speed"] for row in steady_rows], dtype=float)
    lateral_errors = np.array([row["lateral_error"] for row in rows], dtype=float)
    mean_speed = float(np.mean(speeds)) if speeds.size else 0.0
    mean_abs_yaw_rate = float(np.mean(np.abs(yaw_rates))) if yaw_rates.size else 0.0

    return {
        "steps": int(step_count),
        "sim_time_s": float(step_count * dt),
        "steering": float(steering),
        "throttle": float(throttle),
        "initial_speed_mps": float(rows[0]["speed"]),
        "mean_steady_speed_mps": mean_speed,
        "mean_abs_steady_yaw_rate_radps": mean_abs_yaw_rate,
        "mean_steady_lateral_acceleration_mps2": float(mean_speed * mean_abs_yaw_rate),
        "estimated_turn_radius_m": safe_divide(mean_speed, mean_abs_yaw_rate),
        "mean_abs_steady_lateral_speed_mps": float(np.mean(np.abs(lateral_speeds)))
        if lateral_speeds.size
        else 0.0,
        "max_abs_lateral_error_m": float(np.max(np.abs(lateral_errors)))
        if lateral_errors.size
        else 0.0,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "off_track": bool(final_info["off_track"]),
        "lap_complete": bool(final_info["lap_complete"]),
    }


def skidpad_benchmark(
    config_path: str | Path,
    seed: int = 0,
    steps: int = 180,
    steering: float = 0.62,
    throttle: float = 0.34,
    initial_speed: float = 6.0,
) -> dict[str, Any]:
    rows, final_info, terminated, truncated, dt = action_rollout(
        config_path,
        seed,
        steps,
        constant_action(steering, throttle, 0.0),
        initial_speed=initial_speed,
    )
    steady_rows = rows[max(0, len(rows) // 2) :]
    return maneuver_summary(
        "skidpad",
        rows,
        steady_rows,
        final_info,
        terminated,
        truncated,
        dt,
        {"steering": steering, "throttle": throttle},
    )


def step_steer_benchmark(
    config_path: str | Path,
    seed: int = 0,
    steps: int = 180,
    steering: float = 0.7,
    throttle: float = 0.25,
    initial_speed: float = 7.0,
) -> dict[str, Any]:
    def action(step: int) -> np.ndarray:
        steer = 0.0 if step < steps // 3 else steering
        return np.array([steer, throttle, 0.0], dtype=np.float32)

    rows, final_info, terminated, truncated, dt = action_rollout(
        config_path,
        seed,
        steps,
        action,
        initial_speed=initial_speed,
    )
    post_step = rows[min(len(rows) - 1, steps // 3) :]
    summary = maneuver_summary(
        "step_steer",
        rows,
        post_step,
        final_info,
        terminated,
        truncated,
        dt,
        {"steering": steering, "throttle": throttle},
    )
    summary["peak_abs_yaw_rate_radps"] = max_abs(row["yaw_rate"] for row in post_step)
    summary["peak_understeer_score"] = max_abs(row["understeer_score"] for row in post_step)
    return summary


def slalom_benchmark(
    config_path: str | Path,
    seed: int = 0,
    steps: int = 180,
    steering: float = 0.55,
    throttle: float = 0.28,
    period_steps: int = 28,
    initial_speed: float = 6.0,
) -> dict[str, Any]:
    def action(step: int) -> np.ndarray:
        steer = steering * np.sin(2.0 * np.pi * step / max(period_steps, 1))
        return np.array([steer, throttle, 0.0], dtype=np.float32)

    rows, final_info, terminated, truncated, dt = action_rollout(
        config_path,
        seed,
        steps,
        action,
        initial_speed=initial_speed,
    )
    return maneuver_summary(
        "slalom",
        rows,
        rows,
        final_info,
        terminated,
        truncated,
        dt,
        {"steering": steering, "throttle": throttle, "period_steps": period_steps},
    )


def braking_turn_benchmark(
    config_path: str | Path,
    seed: int = 0,
    steps: int = 180,
    steering: float = 0.45,
    brake: float = 0.8,
    initial_speed: float = 9.0,
) -> dict[str, Any]:
    def action(step: int) -> np.ndarray:
        coast_steps = steps // 4
        return np.array(
            [steering, 0.2 if step < coast_steps else 0.0, 0.0 if step < coast_steps else brake],
            dtype=np.float32,
        )

    rows, final_info, terminated, truncated, dt = action_rollout(
        config_path,
        seed,
        steps,
        action,
        initial_speed=initial_speed,
    )
    braking_rows = rows[min(len(rows) - 1, steps // 4) :]
    summary = maneuver_summary(
        "braking_turn",
        rows,
        braking_rows,
        final_info,
        terminated,
        truncated,
        dt,
        {"steering": steering, "brake": brake},
    )
    summary["speed_loss_mps"] = float(rows[0]["speed"] - rows[-1]["speed"])
    return summary


def throttle_exit_benchmark(
    config_path: str | Path,
    seed: int = 0,
    steps: int = 180,
    steering: float = 0.38,
    throttle: float = 0.95,
    initial_speed: float = 5.0,
) -> dict[str, Any]:
    def action(step: int) -> np.ndarray:
        settle_steps = steps // 3
        throttle_now = 0.22 if step < settle_steps else throttle
        return np.array([steering, throttle_now, 0.0], dtype=np.float32)

    rows, final_info, terminated, truncated, dt = action_rollout(
        config_path,
        seed,
        steps,
        action,
        initial_speed=initial_speed,
    )
    exit_rows = rows[min(len(rows) - 1, steps // 3) :]
    summary = maneuver_summary(
        "throttle_exit",
        rows,
        exit_rows,
        final_info,
        terminated,
        truncated,
        dt,
        {"steering": steering, "throttle": throttle},
    )
    summary["exit_speed_gain_mps"] = float(rows[-1]["speed"] - exit_rows[0]["speed"])
    return summary


def repeatability_benchmark(
    config_path: str | Path,
    seed: int = 0,
    steps: int = 80,
) -> dict[str, Any]:
    actions = repeatability_actions(seed, steps)
    first = rollout_trace(config_path, seed, actions)
    second = rollout_trace(config_path, seed, actions)
    comparisons = compare_traces(first, second)
    same_length = len(first) == len(second)
    same_terminal = terminal_signature(first) == terminal_signature(second)

    return {
        "steps_requested": int(steps),
        "first_steps": int(max(len(first) - 1, 0)),
        "second_steps": int(max(len(second) - 1, 0)),
        "same_length": bool(same_length),
        "same_terminal_state": bool(same_terminal),
        "max_abs_observation_delta": float(comparisons["observation"]),
        "max_abs_reward_delta": float(comparisons["reward"]),
        "max_abs_info_delta": comparisons["info"],
        "deterministic": bool(
            same_length
            and same_terminal
            and comparisons["observation"] == 0.0
            and comparisons["reward"] == 0.0
            and max(comparisons["info"].values(), default=0.0) == 0.0
        ),
    }


def action_rollout(
    config_path: str | Path,
    seed: int,
    steps: int,
    action_fn,
    initial_speed: float,
) -> tuple[list[dict[str, float]], dict[str, Any], bool, bool, float]:
    env = make_open_loop_env(config_path, steps)
    dt = step_dt(env)
    _observation, info = env.reset(seed=seed, options={"initial_speed": initial_speed})
    rows = [telemetry_from_info(info)]
    final_info = info
    terminated = False
    truncated = False

    for step in range(1, steps + 1):
        _observation, _reward, terminated, truncated, final_info = env.step(action_fn(step))
        rows.append(telemetry_from_info(final_info))
        if terminated or truncated:
            break

    return rows, final_info, terminated, truncated, dt


def constant_action(steering: float, throttle: float, brake: float):
    def action(_step: int) -> np.ndarray:
        return np.array([steering, throttle, brake], dtype=np.float32)

    return action


def maneuver_summary(
    name: str,
    rows: list[dict[str, float]],
    analysis_rows: list[dict[str, float]],
    final_info: dict[str, Any],
    terminated: bool,
    truncated: bool,
    dt: float,
    inputs: dict[str, float | int],
) -> dict[str, Any]:
    speeds = np.array([row["speed"] for row in analysis_rows], dtype=float)
    yaw_rates = np.array([row["yaw_rate"] for row in analysis_rows], dtype=float)
    tire_usage = np.array(
        [max(row["front_tire_usage"], row["rear_tire_usage"]) for row in analysis_rows],
        dtype=float,
    )
    lateral_errors = np.array([row["lateral_error"] for row in rows], dtype=float)
    slip_angles = np.array(
        [max(abs(row["front_slip_angle"]), abs(row["rear_slip_angle"])) for row in analysis_rows],
        dtype=float,
    )
    mean_speed = float(np.mean(speeds)) if speeds.size else 0.0
    mean_abs_yaw_rate = float(np.mean(np.abs(yaw_rates))) if yaw_rates.size else 0.0
    return {
        "name": name,
        "steps": int(max(len(rows) - 1, 0)),
        "sim_time_s": float(max(len(rows) - 1, 0) * dt),
        "inputs": inputs,
        "initial_speed_mps": float(rows[0]["speed"]),
        "final_speed_mps": float(rows[-1]["speed"]),
        "mean_speed_mps": mean_speed,
        "mean_abs_yaw_rate_radps": mean_abs_yaw_rate,
        "mean_lateral_acceleration_mps2": float(mean_speed * mean_abs_yaw_rate),
        "estimated_turn_radius_m": safe_divide(mean_speed, mean_abs_yaw_rate),
        "peak_tire_usage": float(np.max(tire_usage)) if tire_usage.size else 0.0,
        "mean_tire_usage": float(np.mean(tire_usage)) if tire_usage.size else 0.0,
        "peak_abs_slip_angle_rad": float(np.max(slip_angles)) if slip_angles.size else 0.0,
        "max_abs_lateral_error_m": float(np.max(np.abs(lateral_errors)))
        if lateral_errors.size
        else 0.0,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "off_track": bool(final_info["off_track"]),
        "lap_complete": bool(final_info["lap_complete"]),
    }


def rollout_trace(config_path: str | Path, seed: int, actions: np.ndarray) -> list[dict[str, Any]]:
    env = make_open_loop_env(config_path, len(actions))
    observation, info = env.reset(seed=seed, options={"randomize": True})
    trace = [
        {
            "observation": observation.astype(float).tolist(),
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "info": telemetry_from_info(info),
        }
    ]

    for action in actions:
        observation, reward, terminated, truncated, info = env.step(action)
        trace.append(
            {
                "observation": observation.astype(float).tolist(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": telemetry_from_info(info),
            }
        )
        if terminated or truncated:
            break
    return trace


def repeatability_actions(seed: int, steps: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steering = rng.uniform(-0.5, 0.5, size=steps)
    throttle = rng.uniform(0.0, 0.8, size=steps)
    brake = rng.uniform(0.0, 0.25, size=steps)
    return np.stack([steering, throttle, brake], axis=1).astype(np.float32)


def compare_traces(first: list[dict[str, Any]], second: list[dict[str, Any]]) -> dict[str, Any]:
    pair_count = min(len(first), len(second))
    observation_delta = 0.0
    reward_delta = 0.0
    info_deltas: dict[str, float] = {}

    for left, right in zip(first[:pair_count], second[:pair_count], strict=False):
        observation_delta = max(
            observation_delta,
            max_abs_delta(left["observation"], right["observation"]),
        )
        reward_delta = max(reward_delta, abs(float(left["reward"]) - float(right["reward"])))
        for key, left_value in left["info"].items():
            right_value = right["info"][key]
            info_deltas[key] = max(
                info_deltas.get(key, 0.0),
                max_abs_delta(left_value, right_value),
            )

    return {
        "observation": float(observation_delta),
        "reward": float(reward_delta),
        "info": {key: float(value) for key, value in sorted(info_deltas.items())},
    }


def telemetry_from_info(info: dict[str, Any]) -> dict[str, float]:
    from racesim.eval.telemetry import flatten_info_telemetry

    flat = flatten_info_telemetry(info)
    # Open-loop traces keep the float-compatible subset.
    return {
        key: value
        for key, value in flat.items()
        if isinstance(value, (int, float, bool))
        and key
        not in {
            "progress_delta",
            "raw_progress_delta",
            "progress_delta_clipped",
            "off_track",
            "lap_complete",
        }
    }


def terminal_signature(trace: list[dict[str, Any]]) -> tuple[bool, bool]:
    if not trace:
        return (False, False)
    last = trace[-1]
    return (bool(last["terminated"]), bool(last["truncated"]))


def step_dt(env: RacingEnv) -> float:
    return float(env.frame_skip * env.model.opt.timestep)


def make_open_loop_env(config_path: str | Path | None, max_steps: int) -> RacingEnv:
    """Build an env that ignores lap/off-track termination for open-loop telemetry."""
    env = RacingEnv(resolve_physics_config(config_path))
    env.max_episode_steps = max_steps
    env.lap_target = float("inf")
    env.off_track_margin = 1e6
    return env


def rate(delta: float, elapsed: float) -> float:
    if elapsed <= 0.0:
        return 0.0
    return float(delta / elapsed)


def peak_positive_rate(values: list[float], dt: float) -> float:
    if len(values) < 2 or dt <= 0.0:
        return 0.0
    deltas = np.diff(np.asarray(values, dtype=float)) / dt
    return float(max(np.max(deltas), 0.0))


def safe_divide(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def max_abs(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(max(abs(value) for value in values))


def sanity_flags(benchmarks: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    acceleration = benchmarks.get("acceleration")
    if acceleration and acceleration["average_acceleration_mps2"] <= 0.0:
        flags.append("acceleration did not increase speed")
    braking = benchmarks.get("braking")
    if braking and not braking["stopped"]:
        flags.append("braking run did not reach stop speed")
    for name in (
        "steady_turning",
        "skidpad",
        "step_steer",
        "slalom",
        "braking_turn",
        "throttle_exit",
    ):
        metrics = benchmarks.get(name)
        if not metrics:
            continue
        yaw_rate = metrics.get("mean_abs_yaw_rate_radps")
        if yaw_rate is None:
            yaw_rate = metrics.get("mean_abs_steady_yaw_rate_radps", 0.0)
        if yaw_rate <= 0.01:
            flags.append(f"{name} has almost no yaw response")
        if metrics.get("off_track"):
            flags.append(f"{name} ended off-track (termination margin)")
    for name in ("acceleration", "braking"):
        metrics = benchmarks.get(name)
        if metrics and metrics.get("off_track"):
            flags.append(f"{name} ended off-track (termination margin)")
    repeatability = benchmarks.get("repeatability")
    if repeatability and not repeatability["deterministic"]:
        flags.append("repeatability benchmark was not deterministic")
    return flags


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# RaceSim Physics Benchmarks",
        "",
        f"Config: `{result['config']}`",
        f"Seed: `{result['seed']}`",
        "",
        "These numbers are telemetry baselines. Sanity flags are warnings, not CI gates.",
        "",
        "## Maneuvers",
        "",
        "| benchmark | steps | final speed | yaw rate | tire usage | notes |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for name, metrics in result["benchmarks"].items():
        lines.append(benchmark_row(name, metrics))

    lines.extend(["", "## Sanity Flags", ""])
    flags = result.get("sanity_flags", [])
    if flags:
        lines.extend(f"- {flag}" for flag in flags)
    else:
        lines.append("- none")
    lines.append("")
    return "\n".join(lines)


def benchmark_row(name: str, metrics: dict[str, Any]) -> str:
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
        f"{format_metric(final_speed)} | {format_metric(yaw_rate)} | "
        f"{format_metric(tire_usage)} | {', '.join(notes) or '-'} |"
    )


def format_metric(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int | float):
        return f"{float(value):.3f}"
    return str(value)


def max_abs_delta(left: Any, right: Any) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape:
        return float("inf")
    return float(np.max(np.abs(left_values - right_values))) if left_values.size else 0.0


if __name__ == "__main__":
    main()
