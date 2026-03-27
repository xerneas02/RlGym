from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_tuning_overrides() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        from configs.tuning_constants import (
            CURRICULUM_OVERRIDES,
            REWARD_OVERRIDES,
            TRAINING_CONFIG_OVERRIDES,
        )
    except ImportError:
        return {}, {}, {}

    return (
        deepcopy(TRAINING_CONFIG_OVERRIDES),
        deepcopy(REWARD_OVERRIDES),
        deepcopy(CURRICULUM_OVERRIDES),
    )


def _normalize_override_bundle(overrides: dict[str, Any] | None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not overrides:
        return {}, {}, {}
    return (
        deepcopy(dict(overrides.get("config", {}))),
        deepcopy(dict(overrides.get("rewards", {}))),
        deepcopy(dict(overrides.get("curriculum", {}))),
    )


def load_override_bundle(overrides_path: Path | None = None, inline_overrides: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    bundle = {}
    if overrides_path is not None:
        path = Path(overrides_path)
        if not path.exists():
            raise FileNotFoundError(f"Override bundle not found: {path}")
        if path.suffix.lower() in {".yaml", ".yml"}:
            bundle = dict(load_yaml(path) or {})
        else:
            import json

            bundle = json.loads(path.read_text(encoding="utf-8"))
    inline_config, inline_rewards, inline_curriculum = _normalize_override_bundle(inline_overrides)
    file_config, file_rewards, file_curriculum = _normalize_override_bundle(bundle)
    _deep_update(file_config, inline_config)
    _deep_update(file_rewards, inline_rewards)
    _deep_update(file_curriculum, inline_curriculum)
    return file_config, file_rewards, file_curriculum


def load_project_configs(
    project_root: Path,
    overrides_path: Path | None = None,
    inline_overrides: dict[str, Any] | None = None,
):
    config = load_yaml(project_root / "configs" / "training.yaml")
    rewards = load_yaml(project_root / "configs" / "rewards.yaml")
    curriculum = load_yaml(project_root / "configs" / "curriculum.yaml")

    config_overrides, reward_overrides, curriculum_overrides = _load_tuning_overrides()
    _deep_update(config, config_overrides)
    _deep_update(rewards, reward_overrides)
    _deep_update(curriculum, curriculum_overrides)

    bundle_config, bundle_rewards, bundle_curriculum = load_override_bundle(overrides_path, inline_overrides)
    _deep_update(config, bundle_config)
    _deep_update(rewards, bundle_rewards)
    _deep_update(curriculum, bundle_curriculum)
    return config, rewards, curriculum
