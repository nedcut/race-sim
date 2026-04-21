from __future__ import annotations

import subprocess
import sys


def test_keyboard_drive_help_mentions_camera_option() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "racesim.scripts.keyboard_drive", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--camera" in completed.stdout
