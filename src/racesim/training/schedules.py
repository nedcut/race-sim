"""Learning-rate / clip-range schedule helpers for Stable-Baselines3."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

Schedule = float | Callable[[float], float]


def parse_schedule(value: Any, *, name: str = "schedule") -> Schedule:
    """Parse a constant or linear anneal schedule.

    Accepted forms:
    - ``0.0003`` / ``3e-4`` (float)
    - ``"linear_3e-4"``, ``"linear:3e-4"``, ``"linear,3e-4"``
    - callable already

    Linear schedules map SB3 ``progress_remaining`` ∈ [0, 1] → ``remaining * initial``.
    """
    if callable(value):
        return value  # type: ignore[return-value]
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a number, schedule string, or callable; got {type(value)}")

    text = value.strip().lower()
    if text.startswith("linear"):
        rest = text[len("linear") :].lstrip("_: ,")
        if not rest:
            raise ValueError(f"{name} linear schedule requires an initial value, e.g. linear_3e-4")
        initial = float(rest)
        return lambda progress_remaining, initial=initial: float(progress_remaining) * initial

    if re.fullmatch(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?", text):
        return float(text)
    raise ValueError(f"Unrecognized {name} schedule: {value!r}")


def prepare_ppo_schedules(ppo_config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *ppo_config* with schedule fields parsed for SB3."""
    prepared = dict(ppo_config)
    if "learning_rate" in prepared:
        prepared["learning_rate"] = parse_schedule(prepared["learning_rate"], name="learning_rate")
    if "clip_range" in prepared:
        prepared["clip_range"] = parse_schedule(prepared["clip_range"], name="clip_range")
    if "clip_range_vf" in prepared and prepared["clip_range_vf"] is not None:
        prepared["clip_range_vf"] = parse_schedule(prepared["clip_range_vf"], name="clip_range_vf")
    return prepared
