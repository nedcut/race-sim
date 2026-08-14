"""Unified ``racesim`` command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from typing import Any


def _run(module_main: Callable[..., Any], argv: list[str]) -> int:
    saved = sys.argv
    try:
        sys.argv = [saved[0], *argv]
        result = module_main()
        if result is None:
            return 0
        return int(result)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    finally:
        sys.argv = saved


def _cmd_version(_args: argparse.Namespace) -> int:
    from racesim.version import __version__

    print(f"racesim {__version__}")
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    from racesim.scripts.doctor import main as doctor_main

    argv: list[str] = []
    if args.repo_root is not None:
        argv.extend(["--repo-root", str(args.repo_root)])
    return _run(doctor_main, argv)


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from racesim.eval.evaluate import main as evaluate_main

    return _run(evaluate_main, list(args.argv))


def _cmd_evaluate_policy(args: argparse.Namespace) -> int:
    from racesim.eval.evaluate_policy import main as evaluate_policy_main

    return _run(evaluate_policy_main, list(args.argv))


def _cmd_physics(args: argparse.Namespace) -> int:
    from racesim.eval.physics_benchmarks import main as physics_main

    return _run(physics_main, list(args.argv))


def _cmd_suite(args: argparse.Namespace) -> int:
    from racesim.eval.suite import main as suite_main

    return _run(suite_main, list(args.argv))


def _cmd_train(args: argparse.Namespace) -> int:
    from racesim.training.train_ppo import main as train_main

    return _run(train_main, list(args.argv))


def _cmd_collect_expert(args: argparse.Namespace) -> int:
    from racesim.training.expert_data import main as collect_main

    return _run(collect_main, list(args.argv))


def _cmd_bc_pretrain(args: argparse.Namespace) -> int:
    from racesim.training.bc_pretrain import main as bc_main

    return _run(bc_main, list(args.argv))


def _cmd_keyboard(args: argparse.Namespace) -> int:
    from racesim.scripts.keyboard_drive import main as keyboard_main

    return _run(keyboard_main, list(args.argv))


def _cmd_validate_tracks(args: argparse.Namespace) -> int:
    from racesim.scripts.validate_tracks import main as validate_main

    return _run(validate_main, list(args.argv))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="racesim",
        description="RaceSim — MuJoCo racing environment for control and RL research.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    version = sub.add_parser("version", help="Print package version")
    version.set_defaults(func=_cmd_version)

    doctor = sub.add_parser("doctor", help="Check install and environment")
    doctor.add_argument("--repo-root", type=str, default=None)
    doctor.set_defaults(func=_cmd_doctor)

    # Forward remaining argv to legacy CLIs so flags stay identical.
    for name, help_text, func in (
        ("evaluate", "Evaluate a builtin controller", _cmd_evaluate),
        ("evaluate-policy", "Evaluate a trained policy or predict demo", _cmd_evaluate_policy),
        ("physics", "Run open-loop physics benchmarks", _cmd_physics),
        ("suite", "Run the quality / baseline suite", _cmd_suite),
        ("train", "Train PPO (requires [rl] extra)", _cmd_train),
        ("collect-expert", "Collect expert transitions for BC", _cmd_collect_expert),
        ("bc-pretrain", "Behavioral cloning warm-start for PPO", _cmd_bc_pretrain),
        ("keyboard", "Interactive keyboard drive", _cmd_keyboard),
        ("validate-tracks", "Validate the track catalog", _cmd_validate_tracks),
    ):
        command = sub.add_parser(name, help=help_text)
        command.add_argument("argv", nargs=argparse.REMAINDER, help=argparse.SUPPRESS)
        command.set_defaults(func=func)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # REMAINDER may leave a leading "--" for: racesim evaluate -- --episodes 1
    if hasattr(args, "argv") and args.argv[:1] == ["--"]:
        args.argv = args.argv[1:]
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
