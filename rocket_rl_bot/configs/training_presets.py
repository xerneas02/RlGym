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
        "run_name_prefix": "ppo_touch_focus_v3",
        "resume_latest": False,
    },
    "environment": {
        "timeout_steps": 600,
        "no_touch_timeout_steps": 180,
        "end_on_goal": False,
    },
    "training": {
        "total_steps": 300_000_000,
        "learning_rate": 0.0002,
        "entropy_coef": 0.002,
        "eval_interval_steps": 2_000_000,
        "checkpoint_interval_steps": 2_000_000,
    },
    "evaluation": {
        "protocol": "touch_focus_benchmark_v3",
        "benchmark_weights": {
            "kickoff_like": 0.06,
            "open_goal_attack": 0.28,
            "simple_defense": 0.10,
            "ball_center_random": 0.16,
            "wrong_side_recovery": 0.16,
            "misaligned_recovery": 0.16,
            "wall_ball": 0.04,
            "random_match_state": 0.04,
        },
    },
}


TOUCH_FOCUS_REWARD_OVERRIDES = {
    "weights": {
        "goal_reward": 6.0,
        "ball_goal_progress": 0.18,
        "velocity_to_ball": 0.08,
        "touch_reward": 1.6,
        "defense_position": 0.004,
        "boost_efficiency": 0.0,
    },
    "descriptions": {
        "goal_reward": "Still meaningful, but held below the dense shaping terms so early learning focuses on movement and first contact.",
        "ball_goal_progress": "Stronger dense shaping to keep the ball moving toward the opponent goal after contact.",
        "velocity_to_ball": "Raised to make basic approach and alignment much easier to bootstrap.",
        "touch_reward": "Primary objective during touch-focus: strongly reward first contact, controlled ground touches, and forward touches.",
        "defense_position": "Kept only as a light stabilizer so it does not compete with contact learning.",
        "boost_efficiency": "Disabled during touch-focus to avoid penalizing early aggressive movement.",
    },
}


TOUCH_FOCUS_CURRICULUM_OVERRIDES = {
    "stages": [
        {
            "name": "touch_focus_bootstrap_v3",
            "min_steps": 0,
            "weights": {
                "kickoff_like": 0.04,
                "open_goal_attack": 0.48,
                "simple_defense": 0.06,
                "ball_center_random": 0.18,
                "wrong_side_recovery": 0.10,
                "misaligned_recovery": 0.12,
                "wall_ball": 0.01,
                "random_match_state": 0.01,
            },
        },
        {
            "name": "touch_focus_expand_v3",
            "min_steps": 18_000_000,
            "weights": {
                "kickoff_like": 0.04,
                "open_goal_attack": 0.38,
                "simple_defense": 0.08,
                "ball_center_random": 0.18,
                "wrong_side_recovery": 0.10,
                "misaligned_recovery": 0.10,
                "wall_ball": 0.03,
                "random_match_state": 0.09,
            },
        },
        {
            "name": "touch_focus_generalize_v3",
            "min_steps": 45_000_000,
            "weights": {
                "kickoff_like": 0.04,
                "open_goal_attack": 0.24,
                "simple_defense": 0.10,
                "ball_center_random": 0.18,
                "wrong_side_recovery": 0.08,
                "misaligned_recovery": 0.08,
                "wall_ball": 0.06,
                "random_match_state": 0.22,
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
