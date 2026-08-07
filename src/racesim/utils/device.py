"""Portable training device resolution for PyTorch / Stable-Baselines3."""

from __future__ import annotations

from typing import Any


def resolve_torch_device(requested: str | None = "auto") -> str:
    """Map a requested device string to an available PyTorch device.

    Supported values:
    - ``auto`` (default): prefer CUDA, then MPS, else CPU.
    - ``cpu``, ``cuda``, ``mps``: use that backend when available; fall back
      with a reasonable path when the preferred backend is missing
      (``mps``/``cuda`` → ``cpu``, ``auto`` always returns something runnable).

    Returns a device name suitable for Stable-Baselines3 ``device=...``.
    """
    name = (requested or "auto").strip().lower()
    if name in {"", "auto"}:
        return _best_available_device()
    if name == "cpu":
        return "cpu"
    if name.startswith("cuda"):
        return name if _cuda_available() else "cpu"
    if name == "mps":
        return "mps" if _mps_available() else "cpu"
    # Pass through opaque torch device strings (e.g. "cuda:1") after a light check.
    if name.startswith("cuda") and not _cuda_available():
        return "cpu"
    return name


def apply_device_to_ppo_config(ppo_config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of PPO kwargs with ``device`` resolved for training."""
    resolved = dict(ppo_config)
    requested = resolved.pop("device", "auto")
    resolved["device"] = resolve_torch_device(requested)
    return resolved


def _best_available_device() -> str:
    if _cuda_available():
        return "cuda"
    if _mps_available():
        return "mps"
    return "cpu"


def _cuda_available() -> bool:
    torch = _try_import_torch()
    if torch is None:
        return False
    return bool(torch.cuda.is_available())


def _mps_available() -> bool:
    torch = _try_import_torch()
    if torch is None:
        return False
    backend = getattr(torch.backends, "mps", None)
    if backend is None:
        return False
    return bool(backend.is_available() and backend.is_built())


def _try_import_torch() -> Any | None:
    try:
        import torch
    except ImportError:  # pragma: no cover - optional RL dependency
        return None
    return torch
