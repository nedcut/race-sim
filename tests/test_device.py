from __future__ import annotations

from racesim.utils.device import apply_device_to_ppo_config, resolve_torch_device


def test_resolve_auto_returns_known_device() -> None:
    device = resolve_torch_device("auto")
    assert device in {"cpu", "cuda", "mps"}


def test_resolve_cpu_always_cpu() -> None:
    assert resolve_torch_device("cpu") == "cpu"
    assert resolve_torch_device(None) in {"cpu", "cuda", "mps"}


def test_resolve_mps_falls_back_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("racesim.utils.device._mps_available", lambda: False)
    monkeypatch.setattr("racesim.utils.device._cuda_available", lambda: False)
    assert resolve_torch_device("mps") == "cpu"
    assert resolve_torch_device("auto") == "cpu"


def test_resolve_cuda_falls_back_when_unavailable(monkeypatch) -> None:
    monkeypatch.setattr("racesim.utils.device._cuda_available", lambda: False)
    assert resolve_torch_device("cuda") == "cpu"


def test_resolve_mps_when_available(monkeypatch) -> None:
    monkeypatch.setattr("racesim.utils.device._mps_available", lambda: True)
    assert resolve_torch_device("mps") == "mps"


def test_resolve_auto_prefers_cuda_then_mps(monkeypatch) -> None:
    monkeypatch.setattr("racesim.utils.device._cuda_available", lambda: True)
    monkeypatch.setattr("racesim.utils.device._mps_available", lambda: True)
    assert resolve_torch_device("auto") == "cuda"

    monkeypatch.setattr("racesim.utils.device._cuda_available", lambda: False)
    assert resolve_torch_device("auto") == "mps"


def test_apply_device_to_ppo_config() -> None:
    resolved = apply_device_to_ppo_config({"device": "cpu", "n_steps": 64})
    assert resolved["device"] == "cpu"
    assert resolved["n_steps"] == 64

    missing = apply_device_to_ppo_config({"learning_rate": 1e-3})
    assert missing["device"] in {"cpu", "cuda", "mps"}
    assert missing["learning_rate"] == 1e-3
