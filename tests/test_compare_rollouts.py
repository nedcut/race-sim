from __future__ import annotations

from racesim.eval.compare_rollouts import load_run


def test_load_run_reads_recorded_controller_label(tmp_path) -> None:
    path = tmp_path / "run.json"
    path.write_text(
        """
        {
          "controller": "centerline",
          "config": "configs/env.yaml",
          "episodes": [
            {
              "metrics": {
                "sim_time": 1.0,
                "total_reward": 2.0,
                "lap_complete": true,
                "off_track": false
              },
              "trajectory": [
                {"x": 0, "y": 0, "speed": 1, "cumulative_lap_fraction": 0}
              ]
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    run = load_run(path)

    assert run["label"] == "centerline"
