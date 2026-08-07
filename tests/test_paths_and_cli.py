from __future__ import annotations

from pathlib import Path

from racesim.cli import main as cli_main
from racesim.paths import project_root, resolve_resource
from racesim.version import __version__


def test_version_is_semver_like() -> None:
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts[:2])


def test_cli_version_prints(capsys) -> None:
    assert cli_main(["version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_project_root_finds_configs_and_assets() -> None:
    root = project_root()
    assert (root / "configs").is_dir()
    assert (root / "assets").is_dir()


def test_resolve_resource_finds_default_env() -> None:
    path = resolve_resource("configs/env.yaml")
    assert path.exists()
    assert path.name == "env.yaml"


def test_racesim_root_env_override(monkeypatch, tmp_path: Path) -> None:
    configs = tmp_path / "configs"
    assets = tmp_path / "assets"
    configs.mkdir()
    assets.mkdir()
    monkeypatch.setenv("RACESIM_ROOT", str(tmp_path))
    project_root.cache_clear()
    try:
        assert project_root() == tmp_path.resolve()
    finally:
        project_root.cache_clear()
