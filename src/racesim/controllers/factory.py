from __future__ import annotations

import numpy as np

from racesim.controllers.heuristic import HeuristicController, RacingLineHeuristicController
from racesim.env.track import ClosedTrack

CONTROLLERS = ("centerline", "heuristic", "racing_line", "open_loop")


def make_controller(name: str, track: ClosedTrack) -> object:
    if name in {"centerline", "heuristic"}:
        return HeuristicController(track)
    if name == "racing_line":
        return RacingLineHeuristicController(track)
    if name == "open_loop":
        return OpenLoopController()
    raise ValueError(f"Unsupported controller: {name}")


class OpenLoopController:
    def act(self, _observation: np.ndarray, _info: dict) -> np.ndarray:
        return np.array([0.05, 0.35, 0.0], dtype=np.float32)
