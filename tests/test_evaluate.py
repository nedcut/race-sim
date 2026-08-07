from __future__ import annotations

import json
from pathlib import Path

from racesim.eval.evaluate import episode_seeds, evaluate, evaluate_from_config, main


def test_episode_seeds_cycles_when_shorter_than_episodes() -> None:
    assert episode_seeds(5, seed=0, seeds=[1, 2]) == [1, 2, 1, 2, 1]
    assert episode_seeds(3, seed=10, seeds=None) == [10, 11, 12]


def test_evaluate_uses_explicit_seeds() -> None:
    result = evaluate(
        config_path="configs/env.yaml",
        controller_name="open_loop",
        episodes=2,
        max_steps=2,
        seed=0,
        seeds=[7, 8],
    )
    assert result["seeds"] == [7, 8]
    assert result["summary"]["episodes"] == 2
    assert [episode["seed"] for episode in result["episodes"]] == [7, 8]


def test_evaluate_from_config_runs_registered_controllers(tmp_path: Path) -> None:
    config = tmp_path / "eval.yaml"
    config.write_text(
        "\n".join(
            [
                "env_config: configs/env.yaml",
                "controllers:",
                "  - open_loop",
                "  - missing_controller",
                "episodes: 1",
                "seeds: [3]",
            ]
        ),
        encoding="utf-8",
    )
    result = evaluate_from_config(
        eval_config_path=config,
        max_steps=2,
    )
    assert "open_loop" in result["by_controller"]
    assert result["skipped_controllers"] == ["missing_controller"]
    assert result["by_controller"]["open_loop"]["seeds"] == [3]


def test_evaluate_cli_eval_config(tmp_path: Path) -> None:
    config = tmp_path / "eval.yaml"
    config.write_text(
        "\n".join(
            [
                "env_config: configs/env.yaml",
                "controllers:",
                "  - open_loop",
                "episodes: 1",
                "seeds: [0]",
            ]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "out.json"
    main(
        [
            "--eval-config",
            str(config),
            "--max-steps",
            "2",
            "--output",
            str(output),
        ]
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert "open_loop" in payload["by_controller"]
