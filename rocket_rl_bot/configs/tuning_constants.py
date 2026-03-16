from __future__ import annotations

# Quick local overrides applied on top of the YAML files in configs/.
# Edit this file for fast experiments instead of changing several files by hand.
#
# Important:
# - RocketSim usually ignores simulation_speed_multiplier.
# - The most useful speed knobs are generally num_envs and torch_num_threads.

TRAINING_CONFIG_OVERRIDES = {
    "project": {
        "device": "auto",
    },
    "environment": {
        "num_envs": 6,
        "simulation_speed_multiplier": 1000,
        "tick_skip": 10,
        "timeout_steps": 600,
        "no_touch_timeout_steps": 225,
    },
    "training": {
        "total_steps": 100_000_000,
        "rollout_steps": 8192,
        "batch_size": 65_536,
        "minibatch_size": 4096,
        "epochs": 3,
        "learning_rate": 0.0003,
        "eval_interval_steps": 5_000_000,
        "checkpoint_interval_steps": 5_000_000,
    },
    "evaluation": {
        "num_matches": 16,
    },
    "runtime": {
        "torch_num_threads": 2,
        "mixed_precision": False,
        "deterministic_torch": False,
    },
}

REWARD_OVERRIDES = {
    "weights": {
        "goal_reward": 10.0,
        "ball_goal_progress": 0.05,
        "velocity_to_ball": 0.004,
        "touch_reward": 0.35,
        "defense_position": 0.012,
        "boost_efficiency": 0.0002,
    },
}

CURRICULUM_OVERRIDES = {}
