#!/usr/bin/env python3
"""Train a tiny PPO checkpoint for onboarding demos.

Writes artifacts/ppo_oval_quick.zip (also via racesim-train-ppo).

Usage (from repo root, with [rl] extras installed):

    python scripts/train_quick_checkpoint.py
    # or:
    racesim-train-ppo --config configs/train_ppo_oval_quick.yaml
    # then copy:
    #   cp results/ppo_oval_quick/final_model.zip artifacts/ppo_oval_quick.zip
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "train_ppo_oval_quick.yaml"
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
ARTIFACT_NAME = "ppo_oval_quick.zip"


def main() -> int:
    if not CONFIG_PATH.is_file():
        print(f"Missing config: {CONFIG_PATH}", file=sys.stderr)
        return 1

    try:
        from stable_baselines3 import PPO  # noqa: F401
    except ImportError:
        print(
            "stable-baselines3 not installed. Install with:\n"
            '  pip install -e ".[rl]"\n'
            "Then run:\n"
            "  racesim-train-ppo --config configs/train_ppo_oval_quick.yaml\n"
            f"  cp results/ppo_oval_quick/final_model.zip artifacts/{ARTIFACT_NAME}",
            file=sys.stderr,
        )
        return 1

    sys.path.insert(0, str(REPO_ROOT / "src"))
    # Ensure relative paths resolve from repo root when invoked from elsewhere.
    os.chdir(REPO_ROOT)

    from racesim.training.train_ppo import load_train_config
    from racesim.training.train_ppo import main as train_main

    config = load_train_config(CONFIG_PATH)
    output_dir = Path(config.get("output_dir", "results/ppo_oval_quick"))
    print(
        f"Quick PPO train: timesteps={config.get('total_timesteps')}, "
        f"output_dir={output_dir}"
    )

    # Re-parse via train CLI by temporarily adjusting sys.argv
    old_argv = sys.argv
    try:
        sys.argv = ["racesim-train-ppo", "--config", str(CONFIG_PATH)]
        train_main()
    finally:
        sys.argv = old_argv

    source = output_dir / "final_model.zip"
    if not source.is_file():
        print(f"Training finished but {source} not found.", file=sys.stderr)
        return 1

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = ARTIFACTS_DIR / ARTIFACT_NAME
    shutil.copy2(source, dest)
    size_kb = dest.stat().st_size / 1024
    print(f"Wrote {dest} ({size_kb:.1f} KiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
