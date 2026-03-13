from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch import nn

from src.env.env_builder import build_vector_env
from src.rl.evaluator import evaluate_policy
from src.rl.metrics import MetricWindow, compute_gae, explained_variance
from src.rl.model import ActorCritic
from src.utils.checkpointing import find_latest_checkpoint, load_checkpoint, save_checkpoint
from src.utils.logging_utils import TrainingLogger, create_run_directories, snapshot_config
from src.utils.seeding import set_global_seeds


@dataclass
class RolloutBuffer:
    observations: np.ndarray
    actions: np.ndarray
    log_probs: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    values: np.ndarray
    pointer: int = 0

    @classmethod
    def create(cls, rollout_steps: int, num_actors: int, obs_dim: int) -> "RolloutBuffer":
        return cls(
            observations=np.zeros((rollout_steps, num_actors, obs_dim), dtype=np.float32),
            actions=np.zeros((rollout_steps, num_actors), dtype=np.int64),
            log_probs=np.zeros((rollout_steps, num_actors), dtype=np.float32),
            rewards=np.zeros((rollout_steps, num_actors), dtype=np.float32),
            dones=np.zeros((rollout_steps, num_actors), dtype=np.float32),
            values=np.zeros((rollout_steps, num_actors), dtype=np.float32),
        )

    def add(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        log_prob: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        value: np.ndarray,
    ) -> None:
        self.observations[self.pointer] = observation
        self.actions[self.pointer] = action
        self.log_probs[self.pointer] = log_prob
        self.rewards[self.pointer] = reward
        self.dones[self.pointer] = done
        self.values[self.pointer] = value
        self.pointer += 1


class PPOTrainer:
    def __init__(
        self,
        project_root: Path,
        config: Dict[str, Any],
        reward_config: Dict[str, Any],
        curriculum_config: Dict[str, Any],
        resume_checkpoint: Optional[Path] = None,
    ) -> None:
        self.project_root = Path(project_root)
        self.config = copy.deepcopy(config)
        self.reward_config = copy.deepcopy(reward_config)
        self.curriculum_config = copy.deepcopy(curriculum_config)
        self.training_cfg = self.config["training"]
        self.env_cfg = self.config["environment"]
        self.evaluation_cfg = self.config["evaluation"]
        self.runtime_cfg = self.config["runtime"]
        self.seed = set_global_seeds(
            int(self.config["project"]["seed"]),
            deterministic_torch=bool(self.runtime_cfg.get("deterministic_torch", False)),
        )
        torch.set_num_threads(int(self.runtime_cfg.get("torch_num_threads", 8)))
        self.device = self._resolve_device(self.config["project"].get("device", "auto"))
        self.directories = create_run_directories(self.project_root, self.config)
        self.logger = TrainingLogger(self.directories)

        self.rocket_learn_available = False
        self.rocket_learn_version = "not_installed"
        try:
            import rocket_learn  # type: ignore

            self.rocket_learn_available = True
            self.rocket_learn_version = getattr(rocket_learn, "__version__", "unknown")
        except ImportError:
            pass

        self.resolved_config = {
            **self.config,
            "metadata": {
                "device": str(self.device),
                "rocket_learn_available": self.rocket_learn_available,
                "rocket_learn_version": self.rocket_learn_version,
            },
        }
        snapshot_config(self.directories.config_snapshot, self.resolved_config)

        self.env = build_vector_env(self.env_cfg, self.reward_config, self.curriculum_config)
        initial_obs = self.env.reset()
        self.obs_dim = int(initial_obs.shape[-1])
        self.num_actions = int(self.env.action_space_n)
        self.num_actors = int(self.env.num_actors)
        self.current_obs = initial_obs

        self.model = ActorCritic(self.obs_dim, self.num_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.training_cfg["learning_rate"]))
        self.total_steps = 0
        self.episodes_completed = 0
        self.last_eval_step = 0
        self.last_checkpoint_step = 0
        self.start_time = time.perf_counter()
        self.episode_window = MetricWindow(maxlen=200)
        self.reward_window = MetricWindow(maxlen=200)

        checkpoint_to_load = resume_checkpoint
        if checkpoint_to_load is None and bool(self.config["project"].get("resume_latest", False)):
            checkpoint_to_load = self._find_latest_global_checkpoint()
        if checkpoint_to_load is not None and Path(checkpoint_to_load).exists():
            self._load_resume_state(Path(checkpoint_to_load))

        self._print_startup_summary()

    def train(self) -> None:
        target_steps = int(self.training_cfg["total_steps"])
        rollout_steps = int(self.training_cfg["rollout_steps"])
        next_eval_step = max(int(self.training_cfg["eval_interval_steps"]), self.last_eval_step + int(self.training_cfg["eval_interval_steps"]))
        next_checkpoint_step = max(
            int(self.training_cfg["checkpoint_interval_steps"]),
            self.last_checkpoint_step + int(self.training_cfg["checkpoint_interval_steps"]),
        )

        while self.total_steps < target_steps:
            self.env.update_curriculum(self.total_steps)
            rollout = RolloutBuffer.create(rollout_steps, self.num_actors, self.obs_dim)
            rollout_start = time.perf_counter()
            collected_steps = 0

            for _ in range(rollout_steps):
                observation_tensor = torch.as_tensor(self.current_obs, dtype=torch.float32, device=self.device)
                with torch.no_grad():
                    actions, log_probs, values = self.model.act(observation_tensor)
                next_obs, rewards, dones, infos = self.env.step(actions.cpu().numpy())
                rollout.add(
                    self.current_obs,
                    actions.cpu().numpy(),
                    log_probs.cpu().numpy(),
                    rewards,
                    dones,
                    values.cpu().numpy(),
                )
                self.current_obs = next_obs
                step_increment = int(self.num_actors)
                collected_steps += step_increment
                self.total_steps += step_increment
                self._ingest_infos(infos)
                if self.total_steps >= target_steps:
                    break

            next_obs_tensor = torch.as_tensor(self.current_obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                _, _, next_values = self.model.act(next_obs_tensor, deterministic=True)
            advantages, returns = compute_gae(
                rollout.rewards[: rollout.pointer],
                rollout.values[: rollout.pointer],
                rollout.dones[: rollout.pointer],
                next_values.cpu().numpy(),
                gamma=float(self.training_cfg["gamma"]),
                gae_lambda=float(self.training_cfg["gae_lambda"]),
            )
            update_metrics = self._ppo_update(rollout, advantages, returns)
            rollout_duration = max(time.perf_counter() - rollout_start, 1e-6)
            sps = collected_steps / rollout_duration
            training_time_hours = (time.perf_counter() - self.start_time) / 3600.0

            log_payload = {
                **self.episode_window.means(),
                **update_metrics,
                "steps_per_second": float(sps),
                "environment_fps": float(sps * self.env_cfg["tick_skip"]),
                "training_time_hours": float(training_time_hours),
                "total_steps": float(self.total_steps),
            }
            self.logger.log_training(self.total_steps, log_payload)
            reward_payload = self.reward_window.means()
            if reward_payload:
                self.logger.log_reward_components(self.total_steps, reward_payload)

            self._print_rollout_summary(log_payload)

            while self.total_steps >= next_eval_step:
                eval_metrics = evaluate_policy(
                    self.model,
                    self.env_cfg,
                    self.reward_config,
                    self.curriculum_config,
                    int(self.evaluation_cfg["num_matches"]),
                    self.device,
                )
                self.logger.log_evaluation(self.total_steps, eval_metrics)
                print(
                    f"[eval] steps={self.total_steps:,} goal_rate={eval_metrics['goal_rate']:.3f} "
                    f"reward={eval_metrics['average_reward']:.3f} touches={eval_metrics['touches_per_game']:.3f}"
                )
                self.last_eval_step = next_eval_step
                next_eval_step += int(self.training_cfg["eval_interval_steps"])

            while self.total_steps >= next_checkpoint_step:
                self._save_checkpoint(next_checkpoint_step)
                print(f"[checkpoint] saved at step {next_checkpoint_step:,} -> {self.directories.checkpoint_dir}")
                self.last_checkpoint_step = next_checkpoint_step
                next_checkpoint_step += int(self.training_cfg["checkpoint_interval_steps"])

        self._save_checkpoint(self.total_steps, suffix="final")
        print(f"[done] final checkpoint saved in {self.directories.checkpoint_dir}")
        self.logger.close()
        self.env.close()

    def _ppo_update(self, rollout: RolloutBuffer, advantages: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        observations = rollout.observations[: rollout.pointer].reshape(-1, self.obs_dim)
        actions = rollout.actions[: rollout.pointer].reshape(-1)
        old_log_probs = rollout.log_probs[: rollout.pointer].reshape(-1)
        old_values = rollout.values[: rollout.pointer].reshape(-1)
        flat_advantages = advantages.reshape(-1)
        flat_returns = returns.reshape(-1)

        flat_advantages = (flat_advantages - flat_advantages.mean()) / (flat_advantages.std() + 1e-8)
        total_samples = observations.shape[0]
        batch_size = min(int(self.training_cfg["batch_size"]), total_samples)
        minibatch_size = int(self.training_cfg["minibatch_size"])
        epochs = int(self.training_cfg["epochs"])

        observations_t = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.int64, device=self.device)
        old_log_probs_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(flat_advantages, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(flat_returns, dtype=torch.float32, device=self.device)

        policy_losses = []
        value_losses = []
        entropies = []
        kls = []

        for _ in range(epochs):
            if batch_size < total_samples:
                batch_indices = np.random.choice(total_samples, size=batch_size, replace=False)
            else:
                batch_indices = np.arange(total_samples)
            np.random.shuffle(batch_indices)

            for start in range(0, len(batch_indices), minibatch_size):
                indices = batch_indices[start : start + minibatch_size]
                idx = torch.as_tensor(indices, dtype=torch.int64, device=self.device)
                new_log_probs, entropy, values = self.model.evaluate_actions(
                    observations_t[idx],
                    actions_t[idx],
                )
                log_ratio = new_log_probs - old_log_probs_t[idx]
                ratio = torch.exp(log_ratio)
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - float(self.training_cfg["clip_range"]),
                    1.0 + float(self.training_cfg["clip_range"]),
                )
                surrogate_1 = ratio * advantages_t[idx]
                surrogate_2 = clipped_ratio * advantages_t[idx]
                policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
                value_loss = nn.functional.mse_loss(values, returns_t[idx])
                entropy_loss = entropy.mean()

                loss = (
                    policy_loss
                    + float(self.training_cfg["value_coef"]) * value_loss
                    - float(self.training_cfg["entropy_coef"]) * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), float(self.training_cfg["max_grad_norm"]))
                self.optimizer.step()

                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropies.append(float(entropy_loss.item()))
                kls.append(float((0.5 * torch.square(log_ratio)).mean().item()))

        predicted_values = old_values
        exp_var = explained_variance(predicted_values, flat_returns)
        return {
            "policy_loss": float(np.mean(policy_losses)),
            "value_loss": float(np.mean(value_losses)),
            "training_loss": float(np.mean(policy_losses) + np.mean(value_losses)),
            "entropy": float(np.mean(entropies)),
            "kl_divergence": float(np.mean(kls)),
            "explained_variance": float(exp_var),
        }

    def _save_checkpoint(self, step: int, suffix: str | None = None) -> None:
        filename = f"step_{int(step):012d}.pt" if suffix is None else f"{suffix}.pt"
        path = self.directories.checkpoint_dir / filename
        save_checkpoint(
            path,
            self.model,
            self.optimizer,
            training_state={
                "total_steps": int(self.total_steps),
                "episodes_completed": int(self.episodes_completed),
                "last_eval_step": int(self.last_eval_step),
                "last_checkpoint_step": int(self.last_checkpoint_step),
            },
            config=self.resolved_config,
            seed=self.seed,
        )

    def _load_resume_state(self, checkpoint_path: Path) -> None:
        checkpoint = load_checkpoint(checkpoint_path, self.model, self.optimizer, map_location=self.device)
        training_state = checkpoint.get("training_state", {})
        self.total_steps = int(training_state.get("total_steps", 0))
        self.episodes_completed = int(training_state.get("episodes_completed", 0))
        self.last_eval_step = int(training_state.get("last_eval_step", 0))
        self.last_checkpoint_step = int(training_state.get("last_checkpoint_step", 0))

    def _find_latest_global_checkpoint(self) -> Optional[Path]:
        checkpoints_root = self.project_root / self.config["paths"]["checkpoints_dir"]
        run_dirs = sorted([path for path in checkpoints_root.iterdir() if path.is_dir()]) if checkpoints_root.exists() else []
        latest = None
        for run_dir in run_dirs:
            candidate = find_latest_checkpoint(run_dir)
            if candidate is not None:
                latest = candidate
        return latest

    def _ingest_infos(self, infos) -> None:
        for info in infos:
            if "reward_components" in info:
                self.reward_window.add({
                    key: float(value)
                    for key, value in info["reward_components"].items()
                })
            if info.get("episode"):
                episode = info["episode"]
                self.episodes_completed += 1
                self.episode_window.add(
                    {
                        "episode_reward": float(episode["episode_reward"]),
                        "goal_rate": float(episode["goal_rate"]),
                        "concede_rate": float(episode["concede_rate"]),
                        "ball_touches": float(episode["touches"]),
                        "average_speed": float(episode["average_speed"]),
                        "ball_distance_to_goal": float(episode["ball_distance_to_goal"]),
                        "time_of_possession": float(episode["time_of_possession"]),
                    }
                )

    def _print_startup_summary(self) -> None:
        expected_steps_per_rollout = int(self.training_cfg["rollout_steps"]) * int(self.num_actors)
        print(
            f"[startup] run={self.directories.run_name} device={self.device} seed={self.seed} "
            f"envs={self.env_cfg['num_envs']} actors={self.num_actors} obs_dim={self.obs_dim} actions={self.num_actions}"
        )
        print(
            f"[startup] rollout_steps={self.training_cfg['rollout_steps']} "
            f"steps_per_update={expected_steps_per_rollout:,} total_target={int(self.training_cfg['total_steps']):,}"
        )
        print(f"[startup] logs={self.directories.log_dir} checkpoints={self.directories.checkpoint_dir}")

    def _print_rollout_summary(self, metrics: Dict[str, float]) -> None:
        episode_reward = metrics.get("episode_reward", float("nan"))
        goal_rate = metrics.get("goal_rate", float("nan"))
        touches = metrics.get("ball_touches", float("nan"))
        print(
            f"[train] steps={self.total_steps:,} reward={episode_reward:.3f} goal_rate={goal_rate:.3f} "
            f"touches={touches:.3f} loss={metrics['training_loss']:.4f} entropy={metrics['entropy']:.4f} "
            f"sps={metrics['steps_per_second']:.1f}"
        )

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "cuda":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
