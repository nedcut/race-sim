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
    "repeatability",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run open-loop physics quality benchmarks for RacingEnv."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/env.yaml"))
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
    parser.add_argument("--repeatability-steps", type=int, default=80)
    parser.add_argument("--output", type=Path, default=Path("results/physics_benchmarks.json"))
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
        repeatability_steps=args.repeatability_steps,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["benchmarks"], indent=2))
    print(f"Wrote {args.output}")


def run_physics_benchmarks(
    config_path: str | Path = "configs/env.yaml",
    seed: int = 0,
    benchmarks: list[BenchmarkName] | tuple[BenchmarkName, ...] = DEFAULT_BENCHMARKS,
    acceleration_steps: int = 120,
    braking_steps: int = 120,
    turning_steps: int = 160,
    repeatability_steps: int = 80,
) -> dict[str, Any]:
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
    if "repeatability" in benchmark_set:
        results["repeatability"] = repeatability_benchmark(
            config_path,
            seed + 3,
            repeatability_steps,
        )

    return {
        "config": str(config_path),
        "seed": int(seed),
        "benchmarks": results,
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
    position = np.asarray(info["position"], dtype=float)
    return {
        "x": float(position[0]),
        "y": float(position[1]),
        "heading": float(info["heading"]),
        "progress": float(info["progress"]),
        "lap_fraction": float(info["lap_fraction"]),
        "cumulative_lap_fraction": float(info["cumulative_lap_fraction"]),
        "speed": float(info["speed"]),
        "longitudinal_speed": float(info["longitudinal_speed"]),
        "lateral_speed": float(info["lateral_speed"]),
        "yaw_rate": float(info["yaw_rate"]),
        "lateral_error": float(info["lateral_error"]),
        "heading_error": float(info["heading_error"] or 0.0),
        "grip_scale": float(info["grip_scale"]),
    }


def terminal_signature(trace: list[dict[str, Any]]) -> tuple[bool, bool]:
    if not trace:
        return (False, False)
    last = trace[-1]
    return (bool(last["terminated"]), bool(last["truncated"]))


def step_dt(env: RacingEnv) -> float:
    return float(env.frame_skip * env.model.opt.timestep)


def make_open_loop_env(config_path: str | Path, max_steps: int) -> RacingEnv:
    env = RacingEnv(config_path)
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


def max_abs_delta(left: Any, right: Any) -> float:
    left_values = np.asarray(left, dtype=float)
    right_values = np.asarray(right, dtype=float)
    if left_values.shape != right_values.shape:
        return float("inf")
    return float(np.max(np.abs(left_values - right_values))) if left_values.size else 0.0


if __name__ == "__main__":
    main()
