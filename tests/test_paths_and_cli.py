from __future__ import annotations

from pathlib import Path

from racesim.cli import main as cli_main
from racesim.paths import project_root, resolve_resource
from racesim.version import __version__


def _strict_root(path: Path) -> Path:
    (path / "configs").mkdir()
    (path / "configs" / "env.yaml").write_text("track: configs/tracks/oval.yaml\n")
    (path / "assets" / "mjcf").mkdir(parents=True)
    (path / "assets" / "mjcf" / "world.xml").write_text("<mujoco/>\n")
    return path


def test_version_is_semver_like() -> None:
    parts = __version__.split(".")
    assert len(parts) >= 2
    assert all(part.isdigit() for part in parts[:2])


def test_cli_version_prints(capsys) -> None:
    assert cli_main(["version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_project_root_finds_configs_and_assets() -> None:
    root = project_root()
    assert (root / "configs" / "env.yaml").is_file()
    assert (root / "assets" / "mjcf" / "world.xml").is_file()


def test_resolve_resource_finds_default_env() -> None:
    path = resolve_resource("configs/env.yaml")
    assert path.exists()
    assert path.name == "env.yaml"


def test_racesim_root_env_override(monkeypatch, tmp_path: Path) -> None:
    _strict_root(tmp_path)
    monkeypatch.setenv("RACESIM_ROOT", str(tmp_path))
    project_root.cache_clear()
    try:
        assert project_root() == tmp_path.resolve()
    finally:
        project_root.cache_clear()


def test_empty_configs_assets_cwd_does_not_win(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "assets").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RACESIM_ROOT", raising=False)
    project_root.cache_clear()
    try:
        root = project_root()
        assert root != tmp_path.resolve()
        assert (root / "configs" / "env.yaml").is_file()
        assert (root / "assets" / "mjcf" / "world.xml").is_file()
    finally:
        project_root.cache_clear()
