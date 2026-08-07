from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from racesim.env import racing_env
from racesim.env.racing_env import ControlConfig, RacingEnv

VEHICLE_CONFIGS = [
    Path("configs/vehicles/kart.yaml"),
    Path("configs/vehicles/touring.yaml"),
    Path("configs/vehicles/formula.yaml"),
]


def test_vehicle_presets_map_to_runtime_config_fields() -> None:
    expected_control_fields = {field.name for field in fields(ControlConfig)}
    tire_model_config = getattr(racing_env, "TireModelConfig", None)
    expected_tire_model_fields = (
        {field.name for field in fields(tire_model_config)}
        if tire_model_config is not None
        else set()
    )
    expected_fields = expected_control_fields | expected_tire_model_fields

    for path in VEHICLE_CONFIGS:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        control = config["control"]
        tire_model = config["tire_model"]
        control_kwargs = {
            key: value
            for key, value in (control | tire_model).items()
            if key in expected_control_fields
        }

        assert set(control) <= expected_control_fields
        assert set(control) | set(tire_model) == expected_fields
        ControlConfig(**control_kwargs)
        if tire_model_config is not None:
            tire_model_config(**tire_model)


def test_env_loads_vehicle_preset_without_inline_control(tmp_path: Path) -> None:
    env_config = tmp_path / "env_vehicle.yaml"
    env_config.write_text(
        f"""
track: {Path("configs/tracks/oval.yaml").resolve()}
model:
  xml: {Path("assets/mjcf/world.xml").resolve()}
vehicle: {Path("configs/vehicles/kart.yaml").resolve()}
""".lstrip(),
        encoding="utf-8",
    )

    env = RacingEnv(env_config)

    assert env.control.max_drive_force == 1250.0
    assert env.control.steer_rate == 8.0
    tire_config = getattr(env, "tire_model", env.control)
    assert tire_config.center_of_mass_height == 0.22
    assert env._drive_split() == (0.0, 1.0)


def test_inline_control_overrides_vehicle_preset(tmp_path: Path) -> None:
    env_config = tmp_path / "env_vehicle_override.yaml"
    env_config.write_text(
        f"""
track: {Path("configs/tracks/oval.yaml").resolve()}
model:
  xml: {Path("assets/mjcf/world.xml").resolve()}
vehicle: {Path("configs/vehicles/formula.yaml").resolve()}
control:
  drivetrain: fwd
  max_drive_force: 321.0
""".lstrip(),
        encoding="utf-8",
    )

    env = RacingEnv(env_config)

    assert env.control.max_drive_force == 321.0
    assert env.control.max_lateral_force == 8200.0
    tire_config = getattr(env, "tire_model", env.control)
    assert tire_config.aero_downforce_coefficient == 18.0
    assert env._drive_split() == (1.0, 0.0)


def test_catalog_env_yamls_prefer_vehicle_preset() -> None:
    paths = [Path("configs/env.yaml"), *sorted(Path("configs").glob("env_*.yaml"))]
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "vehicle" in config, path
        assert "control" not in config, path
        assert "tire_model" not in config, path

    env = RacingEnv("configs/env.yaml")
    assert env.control.max_drive_force == 2500.0


def test_simulation_dt_overrides_model_timestep(tmp_path: Path) -> None:
    env_config = tmp_path / "env_dt.yaml"
    base = yaml.safe_load(Path("configs/env.yaml").read_text(encoding="utf-8"))
    base["simulation"]["dt"] = 0.01
    env_config.write_text(yaml.safe_dump(base), encoding="utf-8")
    env = RacingEnv(env_config)
    assert abs(env.model.opt.timestep - 0.01) < 1e-12
    assert abs(env.control_timestep() - 0.04) < 1e-12
