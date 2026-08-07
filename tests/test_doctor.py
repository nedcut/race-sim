from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from racesim.scripts.doctor import (
    check_display,
    check_imports,
    check_optional_rl,
    check_python_version,
    check_repo_assets,
    doctor_ok,
    format_report,
    main,
    run_checks,
)


def test_check_python_version_accepts_311_plus() -> None:
    ok = check_python_version((3, 11, 0, "final", 0))
    assert ok.ok
    assert ok.critical

    bad = check_python_version((3, 10, 12, "final", 0))
    assert not bad.ok
    assert "too old" in bad.message


def test_check_imports_reports_required_modules() -> None:
    results = check_imports()
    names = {r.name for r in results}
    assert "import:gymnasium" in names
    assert "import:mujoco" in names
    assert "import:numpy" in names
    assert "import:yaml" in names
    # In a normal dev install these should all pass.
    assert all(r.ok for r in results)


def test_check_optional_rl_is_non_critical() -> None:
    results = check_optional_rl()
    assert results
    assert all(not r.critical for r in results)


def test_check_repo_assets_ok_at_repo_root() -> None:
    root = Path.cwd()
    results = check_repo_assets(root)
    assert all(r.ok for r in results)
    assert any(r.name == "configs/env.yaml" for r in results)
    assert any(r.name == "assets/mjcf" for r in results)


def test_check_repo_assets_fails_with_tmp_paths(tmp_path: Path) -> None:
    results = check_repo_assets(tmp_path)
    assert not doctor_ok(results)
    messages = " ".join(r.message for r in results if not r.ok)
    assert "missing" in messages


def test_check_repo_assets_partial_assets(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "env.yaml").write_text("track: configs/tracks/oval.yaml\n")
    assets = tmp_path / "assets"
    assets.mkdir()
    # No mjcf xmls
    results = check_repo_assets(tmp_path)
    by_name = {r.name: r for r in results}
    assert by_name["configs/env.yaml"].ok
    assert by_name["assets"].ok
    assert not by_name["assets/mjcf"].ok


def test_check_display_linux_without_display() -> None:
    with patch("racesim.scripts.doctor.platform.system", return_value="Linux"):
        result = check_display({})
    assert not result.ok
    assert not result.critical
    assert "headless" in result.message.lower() or "DISPLAY" in result.message


def test_check_display_linux_with_display() -> None:
    with patch("racesim.scripts.doctor.platform.system", return_value="Linux"):
        result = check_display({"DISPLAY": ":0"})
    assert result.ok


def test_check_display_macos_local() -> None:
    with patch("racesim.scripts.doctor.platform.system", return_value="Darwin"):
        result = check_display({"TERM_PROGRAM": "iTerm.app"})
    assert result.ok
    assert not result.critical


def test_run_checks_and_format_report_with_bad_python(tmp_path: Path) -> None:
    results = run_checks(repo_root=tmp_path, version_info=(3, 9, 0))
    report = format_report(results)
    assert "[FAIL]" in report
    assert not doctor_ok(results)
    assert "critical failure" in report


def test_main_exits_nonzero_for_missing_repo(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["--repo-root", str(tmp_path)])
    captured = capsys.readouterr()
    assert code == 1
    assert "FAIL" in captured.out or "missing" in captured.out.lower()


def test_main_exits_zero_at_repo_root(capsys: pytest.CaptureFixture[str]) -> None:
    code = main([])
    captured = capsys.readouterr()
    assert code == 0
    assert "critical checks passed" in captured.out
