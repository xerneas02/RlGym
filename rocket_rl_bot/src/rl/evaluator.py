from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from src.env.actions import OptimizedDiscreteAction
from src.env.env_builder import build_vector_env
from src.env.observations import CompactObservationBuilder
from src.rl.model import ActorCritic
from src.rl.topdown_viewer import TopDown2DViewer, save_trajectory
from src.utils.checkpointing import load_checkpoint
from src.utils.seeding import preserved_random_state


@dataclass
class EvaluationResult:
    goal_rate: float
    average_reward: float
    touches_per_game: float
    time_of_possession: float
    concede_rate: float
    average_speed: float


class BronzeChaserPolicy:
    def __init__(self, action_parser: OptimizedDiscreteAction) -> None:
        self.action_parser = action_parser
        self.slices = CompactObservationBuilder.SLICES

    def act(self, observation: np.ndarray) -> int:
        rel_ball = observation[self.slices.relative_ball_position]
        ball_position = observation[self.slices.ball_position]
        on_ground = observation[25] > 0.5
        throttle = 1.0 if rel_ball[1] > -0.05 else -1.0
        steer = float(np.clip(rel_ball[0] * 2.5, -1.0, 1.0))
        yaw = steer
        pitch = -1.0 if ball_position[2] > 0.20 else 0.0
        boost = 1 if throttle > 0 and abs(steer) < 0.20 and rel_ball[1] > 0.18 else 0
        jump = 1 if on_ground and np.linalg.norm(rel_ball[:2]) < 0.08 and ball_position[2] > 0.12 else 0
        handbrake = 1 if abs(steer) > 0.75 and rel_ball[1] > 0.05 else 0
        action = [throttle, steer, pitch, yaw, 0.0, jump, boost, handbrake]
        return self.action_parser.nearest_action_index(action)


def _safe_mean(values) -> float:
    return float(np.mean(values)) if values else 0.0


def _resolve_evaluation_seed(evaluation_config: Dict) -> Optional[int]:
    if not bool(evaluation_config.get("deterministic", False)):
        return None
    return int(evaluation_config.get("seed", 1042))


def _is_benchmark_protocol(protocol: str) -> bool:
    normalized = str(protocol).lower()
    return normalized == "benchmark" or "benchmark" in normalized


def _build_evaluation_curriculum(curriculum_config: Dict, evaluation_config: Dict) -> Dict:
    protocol = str(evaluation_config.get("protocol", "benchmark")).lower()
    curriculum = copy.deepcopy(curriculum_config)
    if not _is_benchmark_protocol(protocol):
        return curriculum

    weights = evaluation_config.get("benchmark_weights")
    if not weights:
        weights = curriculum["stages"][0]["weights"]
    return {
        "stages": [
            {
                "name": "evaluation_benchmark",
                "min_steps": 0,
                "weights": {key: float(value) for key, value in weights.items()},
            }
        ]
    }


def _apply_evaluation_stage(env, evaluation_config: Dict, training_step: int) -> None:
    protocol = str(evaluation_config.get("protocol", "benchmark")).lower()
    if protocol == "follow_training":
        env.update_curriculum(int(training_step))
    else:
        env.update_curriculum(0)


def _resolve_opponent_mode(evaluation_config: Dict, opponent_mode: Optional[str]) -> str:
    mode = str(opponent_mode or evaluation_config.get("scripted_opponent", "bronze_chaser")).lower()
    if mode in {"bronze", "bronze_chaser", "scripted"}:
        return "bronze"
    if mode in {"self_play", "self", "mirror", "policy"}:
        return "self_play"
    raise ValueError(f"Unsupported evaluation opponent '{mode}'. Use one of: bronze, self_play.")


def evaluate_policy(
    model: ActorCritic,
    env_config: Dict,
    reward_config: Dict,
    curriculum_config: Dict,
    evaluation_config: Dict,
    num_matches: int,
    device: torch.device,
    training_step: int = 0,
    opponent_mode: Optional[str] = None,
    render_2d: bool = False,
    render_fps: int = 15,
    save_trajectory_path: Optional[Path] = None,
) -> Dict[str, float]:
    evaluation_config = dict(evaluation_config)
    eval_seed = _resolve_evaluation_seed(evaluation_config)
    eval_curriculum = _build_evaluation_curriculum(curriculum_config, evaluation_config)

    with preserved_random_state(seed=eval_seed, deterministic_torch=False):
        env = build_vector_env(env_config, reward_config, eval_curriculum, num_envs=1)
        _apply_evaluation_stage(env, evaluation_config, training_step)
        resolved_opponent = _resolve_opponent_mode(evaluation_config, opponent_mode)
        opponent = BronzeChaserPolicy(OptimizedDiscreteAction()) if resolved_opponent == "bronze" else None
        obs = env.reset()

        goals = []
        rewards = []
        touches = []
        possession = []
        concedes = []
        speeds = []

        viewer = TopDown2DViewer(title="Rocket RL Evaluation") if render_2d else None
        frames = [] if save_trajectory_path is not None else None
        keep_running = True

        try:
            while len(goals) < num_matches and keep_running:
                blue_obs = torch.as_tensor(obs[0:1], dtype=torch.float32, device=device)
                with torch.no_grad():
                    blue_action, _, _ = model.act(blue_obs, deterministic=True)

                if obs.shape[0] > 1:
                    if opponent is None:
                        orange_obs = torch.as_tensor(obs[1:2], dtype=torch.float32, device=device)
                        with torch.no_grad():
                            orange_action, _, _ = model.act(orange_obs, deterministic=True)
                        orange_action_value = int(orange_action.item())
                    else:
                        orange_action_value = int(opponent.act(obs[1]))
                else:
                    orange_action_value = int(blue_action.item())

                obs, _, _, infos = env.step([int(blue_action.item()), orange_action_value])

                frame = env.get_render_state(0)
                if frame is None and infos:
                    frame = infos[0].get("render_state")
                if frame is not None:
                    frame = dict(frame)
                    frame["meta"] = {
                        "completed_matches": len(goals),
                        "target_matches": int(num_matches),
                        "protocol": str(evaluation_config.get("protocol", "benchmark")),
                        "opponent": resolved_opponent,
                    }
                    if frames is not None:
                        frames.append(frame)
                    if viewer is not None:
                        keep_running = viewer.draw(frame, fps=render_fps)

                if infos and infos[0].get("episode"):
                    episode = infos[0]["episode"]
                    goals.append(float(episode["goal_rate"]))
                    rewards.append(float(episode["episode_reward"]))
                    touches.append(float(episode["touches"]))
                    possession.append(float(episode["time_of_possession"]))
                    concedes.append(float(episode["concede_rate"]))
                    speeds.append(float(episode["average_speed"]))
        finally:
            env.close()
            if viewer is not None:
                viewer.close()

    if frames is not None and save_trajectory_path is not None:
        save_trajectory(frames, save_trajectory_path)

    return {
        "goal_rate": _safe_mean(goals),
        "average_reward": _safe_mean(rewards),
        "touches_per_game": _safe_mean(touches),
        "time_of_possession": _safe_mean(possession),
        "concede_rate": _safe_mean(concedes),
        "average_speed": _safe_mean(speeds),
        "matches_completed": float(len(goals)),
        "evaluation_seed": float(eval_seed) if eval_seed is not None else -1.0,
    }


def evaluate_checkpoint(
    checkpoint_path,
    env_config: Dict,
    reward_config: Dict,
    curriculum_config: Dict,
    evaluation_config: Dict,
    device: torch.device,
    num_matches: int,
    opponent_mode: Optional[str] = None,
    render_2d: bool = False,
    render_fps: int = 15,
    save_trajectory_path: Optional[Path] = None,
) -> Dict[str, float]:
    eval_curriculum = _build_evaluation_curriculum(curriculum_config, evaluation_config)
    env = build_vector_env(env_config, reward_config, eval_curriculum, num_envs=1)
    _apply_evaluation_stage(env, evaluation_config, 0)
    obs = env.reset()
    env.close()
    model = ActorCritic(obs_dim=obs.shape[-1], action_dim=env.action_space_n).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    training_step = int(checkpoint.get("training_state", {}).get("total_steps", 0))
    return evaluate_policy(
        model,
        env_config,
        reward_config,
        curriculum_config,
        evaluation_config,
        num_matches,
        device,
        training_step=training_step,
        opponent_mode=opponent_mode,
        render_2d=render_2d,
        render_fps=render_fps,
        save_trajectory_path=save_trajectory_path,
    )
