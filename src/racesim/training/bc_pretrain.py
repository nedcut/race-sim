"""Behavioral cloning warm-start that exports a Stable-Baselines3 PPO zip."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from racesim.paths import default_train_config
from racesim.training.agent_bundle import normalize_enabled, sibling_vecnormalize
from racesim.training.provenance import (
    hash_env_config_tree,
    observation_action_schema,
    schema_mismatch_errors,
)
from racesim.training.train_ppo import load_train_config, make_training_env, make_vec_env

# Near-deterministic Gaussian after BC so PPO's first rollouts resemble the expert.
BC_ACTION_LOG_STD = -3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pretrain a PPO actor via supervised BC on expert .npz data."
    )
    parser.add_argument("--data", type=Path, required=True, help="Expert .npz from collect-expert")
    parser.add_argument(
        "--train-config",
        type=Path,
        default=None,
        help="Train YAML used for env / policy architecture (default train config)",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/bc/bc_policy"),
        help="SB3 save path (without .zip)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate data / env shapes without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_config = load_train_config(args.train_config or default_train_config())
    data = load_expert_npz(args.data)
    result = pretrain_bc(
        train_config=train_config,
        observations=data["observations"],
        actions=data["actions"],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        output=args.output,
        dry_run=args.dry_run,
        metadata=data.get("metadata"),
    )
    print(json.dumps(result, indent=2, default=str))


def load_expert_npz(path: Path) -> dict[str, Any]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as payload:
        result: dict[str, Any] = {
            "observations": np.asarray(payload["observations"], dtype=np.float32),
            "actions": np.asarray(payload["actions"], dtype=np.float32),
        }
        if "metadata_json" in payload.files:
            raw = payload["metadata_json"]
            text = raw.item() if getattr(raw, "ndim", 1) == 0 else str(raw)
            result["metadata"] = json.loads(str(text))
    sidecar = path.with_suffix(".json")
    if sidecar.exists():
        result["metadata"] = json.loads(sidecar.read_text(encoding="utf-8"))
    return result


def assert_expert_matches_train_env(
    metadata: dict[str, Any] | None,
    env: Any,
    train_config: dict[str, Any],
) -> None:
    """Reject datasets whose recorded schema/hashes disagree with the train env."""
    if not metadata:
        return
    env_schema = observation_action_schema(env)
    dataset_schema = metadata.get("observation_action_schema")
    if dataset_schema:
        errors = schema_mismatch_errors(dataset_schema, env_schema)
        if errors:
            raise ValueError(
                "Expert dataset schema does not match the train env:\n  " + "\n  ".join(errors)
            )
    dataset_hashes = metadata.get("env_config_hashes") or {}
    if not dataset_hashes:
        return
    train_hashes: dict[str, str] = {}
    env_block = train_config.get("env", {}) or {}
    env_paths = list(env_block.get("env_configs") or [])
    if not env_paths and train_config.get("env_config"):
        env_paths = [train_config["env_config"]]
    for env_path in env_paths:
        train_hashes.update(hash_env_config_tree(env_path))
    dataset_values = set(dataset_hashes.values())
    train_env_values = {
        digest
        for path, digest in train_hashes.items()
        if Path(path).suffix in {".yaml", ".yml"} and "tracks" not in Path(path).parts
    }
    if train_env_values and train_env_values.isdisjoint(dataset_values):
        raise ValueError(
            "Expert dataset env config hash does not match the train env YAML. "
            "Re-collect experts with the same env config used for PPO."
        )


def pretrain_bc(
    train_config: dict[str, Any],
    observations: np.ndarray,
    actions: np.ndarray,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    output: Path,
    dry_run: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if observations.ndim != 2 or actions.ndim != 2:
        raise ValueError("observations and actions must be 2-D arrays")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions must have the same length")

    env = make_training_env(train_config)
    try:
        obs_dim = int(env.observation_space.shape[0])
        act_dim = int(env.action_space.shape[0])
        assert_expert_matches_train_env(metadata, env, train_config)
    finally:
        env.close()

    if observations.shape[1] != obs_dim:
        raise ValueError(f"Expert obs dim {observations.shape[1]} != train env obs dim {obs_dim}")
    if actions.shape[1] != act_dim:
        raise ValueError(f"Expert act dim {actions.shape[1]} != train env act dim {act_dim}")

    if dry_run:
        return {
            "dry_run": True,
            "n_samples": int(observations.shape[0]),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "normalize_enabled": normalize_enabled(train_config),
        }

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from racesim.training.schedules import prepare_ppo_schedules
    from racesim.utils.device import apply_device_to_ppo_config, resolve_torch_device

    set_random_seed(seed)
    slim = dict(train_config)
    env_cfg = dict(slim.get("env", {}))
    env_cfg["n_envs"] = 1
    env_cfg["vec_env"] = "dummy"
    slim["env"] = env_cfg
    vec_env = make_vec_env(slim, Monitor, DummyVecEnv, DummyVecEnv)

    normalize_cfg = train_config.get("normalize", {})
    if isinstance(normalize_cfg, bool):
        normalize_cfg = {"enabled": normalize_cfg}
    used_normalize = bool(normalize_cfg.get("enabled", False))
    if used_normalize:
        vec_env = VecNormalize(
            vec_env,
            training=True,
            norm_obs=bool(normalize_cfg.get("norm_obs", True)),
            norm_reward=False,
            clip_obs=float(normalize_cfg.get("clip_obs", 10.0)),
            clip_reward=float(normalize_cfg.get("clip_reward", 10.0)),
            gamma=float(train_config.get("ppo", {}).get("gamma", 0.99)),
        )
        if vec_env.norm_obs:
            vec_env.obs_rms.update(np.asarray(observations, dtype=np.float32))
        vec_env.training = False
        observations = np.asarray(vec_env.normalize_obs(observations), dtype=np.float32)

    ppo_raw = prepare_ppo_schedules(dict(train_config.get("ppo", {})))
    ppo_config = apply_device_to_ppo_config(ppo_raw)
    policy = ppo_config.pop("policy", "MlpPolicy")
    if callable(ppo_config.get("learning_rate")):
        ppo_config["learning_rate"] = learning_rate
    device = ppo_config.get("device", resolve_torch_device("auto"))
    model = PPO(
        policy,
        vec_env,
        seed=seed,
        verbose=0,
        **{k: v for k, v in ppo_config.items() if k != "learning_rate"},
        learning_rate=learning_rate,
    )

    observations_t = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
    actions_t = torch.as_tensor(actions, dtype=torch.float32, device=model.device)
    n = observations_t.shape[0]
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=learning_rate)
    history: list[float] = []

    model.policy.train()
    for epoch in range(epochs):
        permutation = torch.randperm(n, device=model.device)
        epoch_losses: list[float] = []
        for start in range(0, n, batch_size):
            index = permutation[start : start + batch_size]
            batch_obs = observations_t[index]
            batch_act = actions_t[index]
            distribution = model.policy.get_distribution(batch_obs)
            if hasattr(distribution.distribution, "mean"):
                pred = distribution.distribution.mean
            else:
                pred = distribution.mode()
            loss = torch.mean((pred - batch_act) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        history.append(mean_loss)
        if epoch == 0 or (epoch + 1) % max(1, epochs // 5) == 0 or epoch + 1 == epochs:
            print(f"BC epoch {epoch + 1}/{epochs}: mse={mean_loss:.6f}")

    if hasattr(model.policy, "log_std"):
        with torch.no_grad():
            model.policy.log_std.fill_(float(BC_ACTION_LOG_STD))
        print(f"Set policy log_std to {BC_ACTION_LOG_STD} (near-deterministic BC actor)")

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    vecnormalize_path = None
    if used_normalize:
        vec_env.training = True
        vecnormalize_path = output.parent / "vecnormalize.pkl"
        paired = Path(str(output) + "_vecnormalize.pkl")
        vec_env.save(str(vecnormalize_path))
        vec_env.save(str(paired))
        vecnormalize_path = paired if sibling_vecnormalize(output) else vecnormalize_path

    meta = {
        "output": str(output) + ".zip",
        "vecnormalize": str(vecnormalize_path) if vecnormalize_path is not None else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "n_samples": int(n),
        "final_mse": history[-1] if history else None,
        "device": str(device),
        "normalize_enabled": used_normalize,
        "action_log_std": BC_ACTION_LOG_STD if hasattr(model.policy, "log_std") else None,
    }
    meta_path = Path(str(output) + "_bc_meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    vec_env.close()
    return meta


def load_train_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    main()
