from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


MIN_PYTHON = (3, 11)
REQUIRED_MODULES = ("gymnasium", "mujoco", "numpy", "yaml")
OPTIONAL_RL_MODULES = ("stable_baselines3", "torch")
DEFAULT_ENV_CONFIG = Path("configs/env.yaml")
DEFAULT_ASSETS_DIR = Path("assets")
DEFAULT_MJCF_DIR = Path("assets/mjcf")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    message: str
    critical: bool = True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose RaceSim environment, imports, assets, and display readiness.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Directory containing configs/ and assets/ (default: current working directory).",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=DEFAULT_ENV_CONFIG,
        help="Path to env YAML relative to --repo-root (default: configs/env.yaml).",
    )
    parser.add_argument(
        "--assets-dir",
        type=Path,
        default=DEFAULT_ASSETS_DIR,
        help="Assets directory relative to --repo-root (default: assets).",
    )
    return parser.parse_args(argv)


def check_python_version(version_info: tuple[int, ...] | None = None) -> CheckResult:
    info = version_info if version_info is not None else sys.version_info
    version = f"{info[0]}.{info[1]}.{info[2]}" if len(info) >= 3 else f"{info[0]}.{info[1]}"
    ok = (info[0], info[1]) >= MIN_PYTHON
    required = f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]}+"
    if ok:
        return CheckResult("python", True, f"Python {version} (>= {required})", critical=True)
    return CheckResult(
        "python",
        False,
        f"Python {version} is too old; need {required}",
        critical=True,
    )


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def check_imports(modules: tuple[str, ...] = REQUIRED_MODULES) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name in modules:
        if module_available(name):
            results.append(CheckResult(f"import:{name}", True, f"{name} importable", critical=True))
        else:
            results.append(
                CheckResult(f"import:{name}", False, f"{name} not importable", critical=True)
            )
    return results


def check_optional_rl(modules: tuple[str, ...] = OPTIONAL_RL_MODULES) -> list[CheckResult]:
    results: list[CheckResult] = []
    for name in modules:
        if module_available(name):
            results.append(
                CheckResult(f"optional:{name}", True, f"{name} available (RL extra)", critical=False)
            )
        else:
            results.append(
                CheckResult(
                    f"optional:{name}",
                    False,
                    f"{name} missing; install with pip install -e '.[rl]' for train-ppo",
                    critical=False,
                )
            )
    return results


def check_path_exists(path: Path, name: str, *, critical: bool = True) -> CheckResult:
    if path.exists():
        kind = "dir" if path.is_dir() else "file"
        return CheckResult(name, True, f"{path} exists ({kind})", critical=critical)
    return CheckResult(name, False, f"{path} missing", critical=critical)


def check_repo_assets(
    repo_root: Path,
    env_config: Path = DEFAULT_ENV_CONFIG,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
) -> list[CheckResult]:
    env_path = env_config if env_config.is_absolute() else repo_root / env_config
    assets_path = assets_dir if assets_dir.is_absolute() else repo_root / assets_dir
    mjcf_path = assets_path / "mjcf" if assets_path.name != "mjcf" else assets_path
    if assets_dir == DEFAULT_ASSETS_DIR:
        mjcf_path = repo_root / DEFAULT_MJCF_DIR

    results = [
        check_path_exists(env_path, "configs/env.yaml", critical=True),
        check_path_exists(assets_path, "assets", critical=True),
    ]
    if assets_path.is_dir():
        has_mjcf = mjcf_path.is_dir() and any(mjcf_path.glob("*.xml"))
        if has_mjcf:
            results.append(
                CheckResult(
                    "assets/mjcf",
                    True,
                    f"{mjcf_path} has MJCF XML files",
                    critical=True,
                )
            )
        else:
            results.append(
                CheckResult(
                    "assets/mjcf",
                    False,
                    f"{mjcf_path} missing or has no .xml files",
                    critical=True,
                )
            )
    return results


def check_display(env: dict[str, str] | None = None) -> CheckResult:
    """Heuristic display/OpenGL readiness check (soft warning when unavailable)."""
    environ = env if env is not None else os.environ
    system = platform.system()

    if system == "Darwin":
        # macOS desktop sessions typically can open a viewer; headless CI often lacks Aqua.
        session = environ.get("TERM_PROGRAM") or environ.get("SSH_CONNECTION")
        if environ.get("SSH_CONNECTION") and not environ.get("DISPLAY"):
            return CheckResult(
                "display",
                False,
                "macOS SSH session without DISPLAY; MuJoCo viewer may be unavailable",
                critical=False,
            )
        return CheckResult(
            "display",
            True,
            f"macOS display heuristic OK ({session or 'local desktop'})",
            critical=False,
        )

    if system == "Linux":
        if environ.get("DISPLAY") or environ.get("WAYLAND_DISPLAY"):
            return CheckResult(
                "display",
                True,
                f"DISPLAY/WAYLAND set (DISPLAY={environ.get('DISPLAY')!r})",
                critical=False,
            )
        return CheckResult(
            "display",
            False,
            "No DISPLAY/WAYLAND_DISPLAY; use headless evaluate/physics (no viewer)",
            critical=False,
        )

    # Windows and others: assume OK; viewer support varies by package build.
    return CheckResult(
        "display",
        True,
        f"{system}: no additional display checks",
        critical=False,
    )


def run_checks(
    repo_root: Path | None = None,
    env_config: Path = DEFAULT_ENV_CONFIG,
    assets_dir: Path = DEFAULT_ASSETS_DIR,
    environ: dict[str, str] | None = None,
    version_info: tuple[int, ...] | None = None,
) -> list[CheckResult]:
    root = (repo_root or Path.cwd()).resolve()
    results: list[CheckResult] = []
    results.append(check_python_version(version_info))
    results.extend(check_imports())
    results.extend(check_optional_rl())
    results.extend(check_repo_assets(root, env_config=env_config, assets_dir=assets_dir))
    results.append(check_display(environ))
    return results


def format_report(results: list[CheckResult]) -> str:
    lines: list[str] = []
    for result in results:
        if result.ok:
            status = "OK"
        elif result.critical:
            status = "FAIL"
        else:
            status = "WARN"
        lines.append(f"[{status}] {result.name}: {result.message}")
    critical_failed = sum(1 for r in results if r.critical and not r.ok)
    soft_failed = sum(1 for r in results if not r.critical and not r.ok)
    lines.append("")
    if critical_failed:
        lines.append(f"Doctor found {critical_failed} critical failure(s).")
    else:
        lines.append("Doctor: critical checks passed.")
    if soft_failed:
        lines.append(f"{soft_failed} optional warning(s) (viewer/RL may be limited).")
    return "\n".join(lines)


def doctor_ok(results: list[CheckResult]) -> bool:
    return all(r.ok for r in results if r.critical)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    results = run_checks(
        repo_root=args.repo_root,
        env_config=args.env_config,
        assets_dir=args.assets_dir,
    )
    print(format_report(results))
    return 0 if doctor_ok(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
