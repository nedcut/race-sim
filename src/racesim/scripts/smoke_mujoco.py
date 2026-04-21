from __future__ import annotations

from pathlib import Path


def main() -> None:
    import mujoco

    model_path = Path(__file__).resolve().parents[3] / "assets" / "mjcf" / "world.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    for _ in range(10):
        mujoco.mj_step(model, data)

    print(
        "MuJoCo smoke test passed: "
        f"nbody={model.nbody}, ngeom={model.ngeom}, sim_time={data.time:.3f}s"
    )


if __name__ == "__main__":
    main()
