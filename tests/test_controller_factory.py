from __future__ import annotations

import numpy as np

from racesim.controllers.factory import make_controller
from racesim.controllers.heuristic import RacingLineHeuristicController
from racesim.env.racing_env import RacingEnv


def test_factory_creates_racing_line_controller() -> None:
    env = RacingEnv("configs/env.yaml")

    controller = make_controller("racing_line", env.track)

    assert isinstance(controller, RacingLineHeuristicController)


def test_racing_line_controller_targets_nonzero_offset_near_corner() -> None:
    env = RacingEnv("configs/env.yaml")
    controller = RacingLineHeuristicController(env.track)

    offsets = [controller.target_lateral_offset(progress) for progress in np.linspace(0, 80, 20)]

    assert max(abs(offset) for offset in offsets) > 0.5
