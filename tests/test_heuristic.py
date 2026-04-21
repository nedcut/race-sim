from __future__ import annotations

from racesim.controllers.heuristic import HeuristicController
from racesim.env.racing_env import RacingEnv


def test_heuristic_action_is_inside_action_space() -> None:
    env = RacingEnv("configs/env.yaml")
    controller = HeuristicController(env.track)
    observation, info = env.reset(seed=0)

    action = controller.act(observation, info)

    assert env.action_space.contains(action)


def test_heuristic_rollout_runs_without_immediate_failure() -> None:
    env = RacingEnv("configs/env.yaml")
    controller = HeuristicController(env.track)
    observation, info = env.reset(seed=0)

    steps = 0
    terminated = False
    for _ in range(20):
        action = controller.act(observation, info)
        observation, _reward, terminated, _truncated, info = env.step(action)
        steps += 1
        if terminated:
            break

    assert steps == 20
    assert not terminated
