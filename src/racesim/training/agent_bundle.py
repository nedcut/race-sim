"""Versioned agent bundle: SB3 zip + VecNormalize stats + schema + manifest."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from racesim.training.provenance import (
    dependency_versions,
    git_sha,
    hash_env_config_tree,
    sha256_file,
)

BUNDLE_SCHEMA_VERSION = 1
MANIFEST_NAME = "manifest.json"
MODEL_NAME = "model.zip"
VECNORMALIZE_NAME = "vecnormalize.pkl"
TRAIN_CONFIG_NAME = "train_config.json"
SCHEMA_NAME = "observation_action_schema.json"


class AgentBundleError(ValueError):
    """Fail-closed bundle validation or I/O error."""


@dataclass(frozen=True)
class AgentBundle:
    path: Path
    manifest: dict[str, Any]
    model_path: Path
    vecnormalize_path: Path | None
    train_config: dict[str, Any]
    observation_action_schema: dict[str, Any]


def normalize_enabled(config: dict[str, Any] | None) -> bool:
    payload = (config or {}).get("normalize", {})
    if isinstance(payload, bool):
        return payload
    if not isinstance(payload, dict):
        return False
    return bool(payload.get("enabled", False))


def sibling_vecnormalize(model_path: Path | str) -> Path | None:
    path = Path(model_path)
    stem = path.with_suffix("") if path.suffix == ".zip" else path
    candidates = [
        stem.parent / VECNORMALIZE_NAME,
        Path(str(stem) + "_vecnormalize.pkl"),
        stem.with_name(stem.name + "_vecnormalize.pkl"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def nearby_train_config(model_path: Path | str) -> dict[str, Any] | None:
    path = Path(model_path)
    directory = path if path.is_dir() else path.parent
    for candidate in (directory / TRAIN_CONFIG_NAME, directory.parent / TRAIN_CONFIG_NAME):
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return None


def is_agent_bundle(path: Path | str) -> bool:
    candidate = Path(path)
    return candidate.is_dir() and (candidate / MANIFEST_NAME).exists()


def save_agent_bundle(
    bundle_dir: Path | str,
    *,
    model_path: Path | str,
    vecnormalize_path: Path | str | None,
    train_config: dict[str, Any],
    observation_action_schema: dict[str, Any],
    env_config_paths: list[str | Path] | None = None,
    timesteps: int | None = None,
    seed: int | None = None,
    init_mode: str = "fresh",
) -> Path:
    """Write a directory bundle. VecNormalize is required when normalize.enabled."""
    destination = Path(bundle_dir)
    destination.mkdir(parents=True, exist_ok=True)

    model_src = Path(model_path)
    if model_src.suffix != ".zip":
        zipped = Path(str(model_src) + ".zip")
        if zipped.exists():
            model_src = zipped
    if not model_src.exists():
        raise AgentBundleError(f"SB3 model zip not found: {model_src}")

    model_dest = destination / MODEL_NAME
    shutil.copy2(model_src, model_dest)

    vec_dest: Path | None = None
    required = normalize_enabled(train_config)
    if required:
        if vecnormalize_path is None:
            raise AgentBundleError(
                "normalize.enabled is true but VecNormalize stats were not provided; "
                "refusing to save a bundle that cannot be evaluated correctly"
            )
        vec_src = Path(vecnormalize_path)
        if not vec_src.exists():
            raise AgentBundleError(f"VecNormalize stats missing: {vec_src}")
        vec_dest = destination / VECNORMALIZE_NAME
        shutil.copy2(vec_src, vec_dest)
    elif vecnormalize_path is not None:
        vec_src = Path(vecnormalize_path)
        if vec_src.exists():
            vec_dest = destination / VECNORMALIZE_NAME
            shutil.copy2(vec_src, vec_dest)

    train_dest = destination / TRAIN_CONFIG_NAME
    train_dest.write_text(json.dumps(train_config, indent=2, default=str), encoding="utf-8")
    schema_dest = destination / SCHEMA_NAME
    schema_dest.write_text(json.dumps(observation_action_schema, indent=2), encoding="utf-8")

    env_hashes: dict[str, str] = {}
    for env_path in env_config_paths or _env_paths_from_train_config(train_config):
        env_hashes.update(hash_env_config_tree(env_path))

    files = {
        MODEL_NAME: {"sha256": sha256_file(model_dest)},
        TRAIN_CONFIG_NAME: {"sha256": sha256_file(train_dest)},
        SCHEMA_NAME: {"sha256": sha256_file(schema_dest)},
    }
    if vec_dest is not None:
        files[VECNORMALIZE_NAME] = {"sha256": sha256_file(vec_dest)}

    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "normalize_enabled": required,
        "files": files,
        "env_config_hashes": env_hashes,
        "git_sha": git_sha(),
        "dependencies": dependency_versions(),
        "training": {
            "timesteps": timesteps,
            "seed": seed,
            "init_mode": init_mode,
            "output_dir": str(train_config.get("output_dir", "")),
        },
        "observation_action_schema": observation_action_schema,
    }
    (destination / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def load_agent_bundle(bundle_dir: Path | str) -> AgentBundle:
    """Load and fail-closed-validate an agent bundle directory."""
    path = Path(bundle_dir)
    if not path.is_dir():
        raise AgentBundleError(f"Agent bundle is not a directory: {path}")
    manifest_path = path / MANIFEST_NAME
    if not manifest_path.exists():
        raise AgentBundleError(f"Agent bundle missing {MANIFEST_NAME}: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = int(manifest.get("schema_version", 0))
    if schema_version != BUNDLE_SCHEMA_VERSION:
        raise AgentBundleError(
            f"Unsupported bundle schema_version {schema_version}; expected {BUNDLE_SCHEMA_VERSION}"
        )

    model_path = path / MODEL_NAME
    if not model_path.exists():
        raise AgentBundleError(f"Agent bundle missing {MODEL_NAME}")
    _assert_sha(model_path, manifest, MODEL_NAME)

    train_config_path = path / TRAIN_CONFIG_NAME
    if not train_config_path.exists():
        raise AgentBundleError(f"Agent bundle missing {TRAIN_CONFIG_NAME}")
    train_config = json.loads(train_config_path.read_text(encoding="utf-8"))
    _assert_sha(train_config_path, manifest, TRAIN_CONFIG_NAME)

    schema_path = path / SCHEMA_NAME
    if schema_path.exists():
        observation_schema = json.loads(schema_path.read_text(encoding="utf-8"))
        _assert_sha(schema_path, manifest, SCHEMA_NAME)
    else:
        observation_schema = dict(manifest.get("observation_action_schema") or {})

    required = bool(manifest.get("normalize_enabled", normalize_enabled(train_config)))
    vec_path = path / VECNORMALIZE_NAME
    if required and not vec_path.exists():
        raise AgentBundleError(
            "This bundle was trained with VecNormalize, but vecnormalize.pkl is missing. "
            "Refusing to evaluate on raw observations."
        )
    vecnormalize_path = vec_path if vec_path.exists() else None
    if vecnormalize_path is not None:
        _assert_sha(vecnormalize_path, manifest, VECNORMALIZE_NAME)

    return AgentBundle(
        path=path,
        manifest=manifest,
        model_path=model_path,
        vecnormalize_path=vecnormalize_path,
        train_config=train_config,
        observation_action_schema=observation_schema,
    )


def resolve_model_checkpoint(
    model_path: Path | str,
    vecnormalize_path: Path | str | None = None,
) -> tuple[Path, Path | None]:
    """Resolve a loose --model zip with fail-closed VecNormalize discovery."""
    path = Path(model_path)
    if is_agent_bundle(path):
        raise AgentBundleError(f"{path} is an agent bundle; load it with --bundle")
    if path.is_dir():
        raise AgentBundleError(f"{path} is a directory without {MANIFEST_NAME}")
    zip_path = path if path.suffix == ".zip" else Path(str(path) + ".zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"Model zip not found: {zip_path}")

    explicit = Path(vecnormalize_path) if vecnormalize_path is not None else None
    if explicit is not None and not explicit.exists():
        raise AgentBundleError(f"VecNormalize stats not found: {explicit}")

    sibling = sibling_vecnormalize(zip_path)
    train_config = nearby_train_config(zip_path)
    required = normalize_enabled(train_config) if train_config is not None else False

    if explicit is not None:
        return zip_path, explicit
    if required and sibling is None:
        raise AgentBundleError(
            f"Training config at {zip_path.parent} has normalize.enabled, but no "
            "VecNormalize stats were found next to the model. Pass --vecnormalize or use --bundle."
        )
    return zip_path, sibling


def _assert_sha(path: Path, manifest: dict[str, Any], name: str) -> None:
    expected = ((manifest.get("files") or {}).get(name) or {}).get("sha256")
    if expected is None:
        return
    actual = sha256_file(path)
    if actual != expected:
        raise AgentBundleError(f"SHA-256 mismatch for {name}: expected {expected}, got {actual}")


def _env_paths_from_train_config(train_config: dict[str, Any]) -> list[str]:
    env_config = train_config.get("env", {}) or {}
    paths = list(env_config.get("env_configs") or [])
    if not paths:
        legacy = train_config.get("env_config")
        if legacy:
            paths = [str(legacy)]
    eval_env = (train_config.get("eval") or {}).get("env_config")
    if eval_env and eval_env not in paths:
        paths.append(str(eval_env))
    return [str(path) for path in paths]
