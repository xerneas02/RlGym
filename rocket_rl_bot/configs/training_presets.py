from __future__ import annotations

from copy import deepcopy
from typing import Any


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


TOUCH_FOCUS_TRAINING_OVERRIDES = {
    "project": {
        "run_name_prefix": "ppo_goal_finish_v1",
        "resume_latest": False,
    },
    "environment": {
        "timeout_steps": 600,
        "no_touch_timeout_steps": 180,
        "end_on_goal": True,
    },
    "training": {
        "total_steps": 120_000_000,
        "learning_rate": 0.00015,
        "entropy_coef": 0.0015,
        "eval_interval_steps": 2_000_000,
        "checkpoint_interval_steps": 2_000_000,
        "opponent_mix": {
            "train_blue_only": True,
            "history_pool_size": 8,
            "history_snapshot_interval_steps": 1_000_000,
            "seed_history_with_current_policy": True,
            "opponent_pool": [
                {"name": "self_play", "weight": 0.30},
                {"name": "historical_self_play", "weight": 0.30},
                {"name": "necto", "weight": 0.20},
                {"name": "seer", "weight": 0.20},
            ],
        },
    },
    "evaluation": {
        "protocol": "goal_finish_benchmark_v1",
        "benchmark_weights": {
            "kickoff_like": 0.12,
            "open_goal_attack": 0.26,
            "simple_defense": 0.12,
            "ball_center_random": 0.16,
            "wrong_side_recovery": 0.12,
            "misaligned_recovery": 0.10,
            "wall_ball": 0.04,
            "random_match_state": 0.08,
        },
    },
}


TOUCH_FOCUS_REWARD_OVERRIDES = {
    "weights": {
        "goal_reward": 14.0,
        "ball_goal_progress": 0.30,
        "velocity_to_ball": 0.10,
        "touch_reward": 0.45,
        "defense_position": 0.020,
        "boost_efficiency": 0.0,
    },
    "descriptions": {
        "goal_reward": "Primary objective during this stage: scoring and not conceding should dominate optimization.",
        "ball_goal_progress": "High dense shaping so the policy prefers dangerous forward ball movement over touch count.",
        "velocity_to_ball": "Keeps kickoff and challenge commitment high to avoid slowing down before contact.",
        "touch_reward": "Moderate touch signal: useful contacts stay rewarded, but touch farming loses priority.",
        "defense_position": "Slightly stronger stabilizer so the bot can transition between pressure and protection.",
        "boost_efficiency": "Disabled to avoid dampening aggressive attacking decisions.",
    },
}


TOUCH_FOCUS_CURRICULUM_OVERRIDES = {
    "stages": [
        {
            "name": "goal_finish_bootstrap_v1",
            "min_steps": 0,
            "weights": {
                "kickoff_like": 0.10,
                "open_goal_attack": 0.50,
                "simple_defense": 0.08,
                "ball_center_random": 0.14,
                "wrong_side_recovery": 0.08,
                "misaligned_recovery": 0.07,
                "wall_ball": 0.01,
                "random_match_state": 0.02,
            },
        },
        {
            "name": "goal_finish_expand_v1",
            "min_steps": 15_000_000,
            "weights": {
                "kickoff_like": 0.08,
                "open_goal_attack": 0.40,
                "simple_defense": 0.10,
                "ball_center_random": 0.16,
                "wrong_side_recovery": 0.09,
                "misaligned_recovery": 0.08,
                "wall_ball": 0.03,
                "random_match_state": 0.06,
            },
        },
        {
            "name": "goal_finish_generalize_v1",
            "min_steps": 40_000_000,
            "weights": {
                "kickoff_like": 0.06,
                "open_goal_attack": 0.28,
                "simple_defense": 0.12,
                "ball_center_random": 0.18,
                "wrong_side_recovery": 0.08,
                "misaligned_recovery": 0.07,
                "wall_ball": 0.05,
                "random_match_state": 0.16,
            },
        },
    ]
}


def apply_touch_focus_preset(
    config: dict[str, Any],
    rewards: dict[str, Any],
    curriculum: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_copy = deepcopy(config)
    rewards_copy = deepcopy(rewards)
    curriculum_copy = deepcopy(curriculum)
    _deep_update(config_copy, TOUCH_FOCUS_TRAINING_OVERRIDES)
    _deep_update(rewards_copy, TOUCH_FOCUS_REWARD_OVERRIDES)
    _deep_update(curriculum_copy, TOUCH_FOCUS_CURRICULUM_OVERRIDES)
    return config_copy, rewards_copy, curriculum_copy
