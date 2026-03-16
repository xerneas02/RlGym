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


def load_project_configs(project_root: Path):
    config = load_yaml(project_root / "configs" / "training.yaml")
    rewards = load_yaml(project_root / "configs" / "rewards.yaml")
    curriculum = load_yaml(project_root / "configs" / "curriculum.yaml")

    config_overrides, reward_overrides, curriculum_overrides = _load_tuning_overrides()
    _deep_update(config, config_overrides)
    _deep_update(rewards, reward_overrides)
    _deep_update(curriculum, curriculum_overrides)
    return config, rewards, curriculum
