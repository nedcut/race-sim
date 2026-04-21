from __future__ import annotations

import argparse

from racesim.scripts.keyboard_drive import maybe_reexec_with_mjpython


def test_no_reexec_flag_returns_without_relaunching() -> None:
    args = argparse.Namespace(no_reexec=True)

    assert maybe_reexec_with_mjpython(args) is None
