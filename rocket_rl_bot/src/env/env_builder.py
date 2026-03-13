from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.env.actions import OptimizedDiscreteAction
from src.env.observations import CompactObservationBuilder
from src.env.rewards import MinimalRewardFunction
from src.env.state_setters import WeightedSampleSetter
from src.env.terminal_conditions import build_terminal_conditions


def _import_rlgym_backend():
    try:
        import rlgym_sim as backend

        return backend, "sim"
    except ImportError:  # pragma: no cover
        import rlgym as backend

        return backend, "classic"


def _coerce_obs(raw_obs: Any) -> np.ndarray:
    if isinstance(raw_obs, tuple) and len(raw_obs) == 2:
        raw_obs = raw_obs[0]
    if isinstance(raw_obs, dict):
        ordered = list(raw_obs.values())
        return np.asarray(ordered, dtype=np.float32)
    if isinstance(raw_obs, (list, tuple)):
        if len(raw_obs) == 0:
            return np.zeros((0, CompactObservationBuilder.OBSERVATION_DIM), dtype=np.float32)
        first = raw_obs[0]
        if np.asarray(first).ndim == 0:
            return np.asarray(raw_obs, dtype=np.float32).reshape(1, -1)
        return np.asarray(raw_obs, dtype=np.float32)
    arr = np.asarray(raw_obs, dtype=np.float32)
    if arr.ndim == 1:
        return arr.reshape(1, -1)
    return arr


def _coerce_rewards(raw_rewards: Any, num_agents: int) -> np.ndarray:
    if isinstance(raw_rewards, dict):
        values = list(raw_rewards.values())
    elif isinstance(raw_rewards, (list, tuple, np.ndarray)):
        values = raw_rewards
    else:
        values = [raw_rewards] * num_agents
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 1 and num_agents > 1:
        arr = np.repeat(arr, num_agents)
    return arr[:num_agents]


def _coerce_dones(raw_done: Any, num_agents: int) -> np.ndarray:
    if isinstance(raw_done, dict):
        values = list(raw_done.values())
    elif isinstance(raw_done, (list, tuple, np.ndarray)):
        values = raw_done
    else:
        values = [raw_done] * num_agents
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 1 and num_agents > 1:
        arr = np.repeat(arr, num_agents)
    return arr[:num_agents]


def _coerce_infos(raw_info: Any, num_agents: int) -> List[Dict[str, Any]]:
    if isinstance(raw_info, list):
        return [dict(item) if isinstance(item, dict) else {"raw_info": item} for item in raw_info[:num_agents]]
    if isinstance(raw_info, tuple):
        return [dict(item) if isinstance(item, dict) else {"raw_info": item} for item in raw_info[:num_agents]]
    if isinstance(raw_info, dict):
        if any(isinstance(value, dict) for value in raw_info.values()):
            infos = []
            for value in raw_info.values():
                infos.append(dict(value) if isinstance(value, dict) else {"raw_info": value})
            return infos[:num_agents]
        return [dict(raw_info) for _ in range(num_agents)]
    return [{} for _ in range(num_agents)]


@dataclass
class EpisodeStats:
    returns: np.ndarray
    touches: np.ndarray
    goals_for: np.ndarray
    goals_against: np.ndarray
    speed_sum: np.ndarray
    ball_distance_sum: np.ndarray
    possession_steps: np.ndarray
    component_sums: List[Dict[str, float]]
    steps: int = 0


class RLGymMatchAdapter:
    def __init__(
        self,
        env_config: Dict[str, Any],
        reward_config: Dict[str, Any],
        curriculum_config: Dict[str, Any],
    ) -> None:
        self.env_config = env_config
        self.reward_function = MinimalRewardFunction(reward_config["weights"])
        self.state_setter = WeightedSampleSetter(curriculum_config)
        self.action_parser = OptimizedDiscreteAction()
        self.obs_builder = CompactObservationBuilder()
        self.backend, self.backend_kind = _import_rlgym_backend()
        self.raw_env = self._build_raw_env()
        self.num_agents = int(env_config["team_size"] * (2 if env_config["spawn_opponents"] else 1))
        self.agent_teams = [0] * int(env_config["team_size"]) + ([1] * int(env_config["team_size"]) if env_config["spawn_opponents"] else [])
        self.last_touch_team = None
        self.episode_stats = self._empty_episode_stats()
        self.action_space_n = self.action_parser.summary.size
        self._last_reset_obs = None
        self.latest_render_state: Optional[Dict[str, Any]] = None

    def reset(self) -> np.ndarray:
        obs = _coerce_obs(self.raw_env.reset())
        self.num_agents = obs.shape[0]
        if len(self.agent_teams) != self.num_agents:
            blue_agents = min(self.env_config["team_size"], self.num_agents)
            orange_agents = max(0, self.num_agents - blue_agents)
            self.agent_teams = [0] * blue_agents + [1] * orange_agents
        self.last_touch_team = None
        self.episode_stats = self._empty_episode_stats()
        self._last_reset_obs = obs
        self.latest_render_state = self._extract_render_state(
            getattr(self.raw_env, "_prev_state", None),
            np.zeros((self.num_agents, 8), dtype=np.float32),
        )
        return obs

    def step(self, actions: Sequence[int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
        discrete_actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        parsed_actions = self.action_parser.parse_actions(discrete_actions)
        step_result = self.raw_env.step(discrete_actions)
        if len(step_result) == 5:
            raw_obs, raw_rewards, terminated, truncated, raw_infos = step_result
            raw_done = np.logical_or(terminated, truncated)
        else:
            raw_obs, raw_rewards, raw_done, raw_infos = step_result

        obs = _coerce_obs(raw_obs)
        rewards = _coerce_rewards(raw_rewards, self.num_agents)
        dones = _coerce_dones(raw_done, self.num_agents)
        infos = _coerce_infos(raw_infos, self.num_agents)
        raw_state = raw_infos.get("state") if isinstance(raw_infos, dict) else None
        self.latest_render_state = self._extract_render_state(raw_state, parsed_actions)
        component_logs = self.reward_function.consume_step_components(self.num_agents)

        touched_indices = [index for index, log in enumerate(component_logs) if log.get("ball_touched", 0.0) > 0.5]
        if touched_indices:
            self.last_touch_team = self.agent_teams[touched_indices[0]]

        self.episode_stats.steps += 1
        for index in range(self.num_agents):
            components = component_logs[index]
            infos[index]["reward_components"] = {
                name: float(components.get(name, 0.0))
                for name in self.reward_function.COMPONENT_NAMES
            }
            infos[index]["ball_touched"] = bool(components.get("ball_touched", 0.0) > 0.5)
            infos[index]["speed"] = float(components.get("speed", 0.0))
            infos[index]["ball_distance_to_goal"] = float(components.get("ball_distance_to_goal", 0.0))
            infos[index]["goals_for"] = float(components.get("goals_for", 0.0))
            infos[index]["goals_against"] = float(components.get("goals_against", 0.0))
            infos[index]["curriculum_stage"] = self.state_setter.describe()["name"]
            if self.latest_render_state is not None:
                infos[index]["render_state"] = self.latest_render_state

            self.episode_stats.returns[index] += float(rewards[index])
            self.episode_stats.touches[index] += float(components.get("ball_touched", 0.0))
            self.episode_stats.goals_for[index] += float(components.get("goals_for", 0.0))
            self.episode_stats.goals_against[index] += float(components.get("goals_against", 0.0))
            self.episode_stats.speed_sum[index] += float(components.get("speed", 0.0))
            self.episode_stats.ball_distance_sum[index] += float(components.get("ball_distance_to_goal", 0.0))
            if self.last_touch_team is not None and self.agent_teams[index] == self.last_touch_team:
                self.episode_stats.possession_steps[index] += 1.0
            for name in self.reward_function.COMPONENT_NAMES:
                self.episode_stats.component_sums[index][name] += float(components.get(name, 0.0))

        if bool(np.any(dones)):
            final_infos = self._finalize_episode()
            infos = [dict(info, episode=final_infos[index]) for index, info in enumerate(infos)]
            obs = self.reset()
            dones = np.ones(self.num_agents, dtype=np.float32)

        return obs, rewards, dones, infos

    def set_training_step(self, total_steps: int) -> None:
        self.state_setter.set_training_step(total_steps)

    def close(self) -> None:
        close_fn = getattr(self.raw_env, "close", None)
        if callable(close_fn):
            close_fn()

    def get_render_state(self) -> Optional[Dict[str, Any]]:
        return self.latest_render_state

    def _build_raw_env(self):
        common_kwargs = dict(
            tick_skip=int(self.env_config["tick_skip"]),
            team_size=int(self.env_config["team_size"]),
            spawn_opponents=bool(self.env_config["spawn_opponents"]),
            terminal_conditions=build_terminal_conditions(self.env_config),
            obs_builder=self.obs_builder,
            action_parser=self.action_parser,
            state_setter=self.state_setter,
            copy_gamestate_every_step=bool(self.env_config.get("copy_gamestate_every_step", False)),
        )
        if self.backend_kind == "classic":
            common_kwargs["game_speed"] = int(self.env_config.get("simulation_speed_multiplier", 100))

        make_fn = getattr(self.backend, "make", None)
        if callable(make_fn):
            try:
                return make_fn(reward_fn=self.reward_function, **common_kwargs)
            except TypeError:
                return make_fn(reward_function=self.reward_function, **common_kwargs)

        from rlgym.envs import Match  # pragma: no cover
        from rlgym.gym import Gym  # pragma: no cover

        match = Match(reward_function=self.reward_function, **common_kwargs)
        return Gym(match)

    def _empty_episode_stats(self) -> EpisodeStats:
        return EpisodeStats(
            returns=np.zeros(self.num_agents, dtype=np.float32),
            touches=np.zeros(self.num_agents, dtype=np.float32),
            goals_for=np.zeros(self.num_agents, dtype=np.float32),
            goals_against=np.zeros(self.num_agents, dtype=np.float32),
            speed_sum=np.zeros(self.num_agents, dtype=np.float32),
            ball_distance_sum=np.zeros(self.num_agents, dtype=np.float32),
            possession_steps=np.zeros(self.num_agents, dtype=np.float32),
            component_sums=[
                {name: 0.0 for name in self.reward_function.COMPONENT_NAMES}
                for _ in range(self.num_agents)
            ],
            steps=0,
        )

    def _finalize_episode(self) -> List[Dict[str, Any]]:
        episode_length = max(1, self.episode_stats.steps)
        final_infos: List[Dict[str, Any]] = []
        for index in range(self.num_agents):
            final_infos.append(
                {
                    "episode_reward": float(self.episode_stats.returns[index]),
                    "touches": float(self.episode_stats.touches[index]),
                    "goal_rate": float(self.episode_stats.goals_for[index]),
                    "concede_rate": float(self.episode_stats.goals_against[index]),
                    "average_speed": float(self.episode_stats.speed_sum[index] / episode_length),
                    "ball_distance_to_goal": float(
                        self.episode_stats.ball_distance_sum[index] / episode_length
                    ),
                    "time_of_possession": float(self.episode_stats.possession_steps[index] / episode_length),
                    "episode_length": int(episode_length),
                    "reward_components": {
                        name: float(self.episode_stats.component_sums[index][name])
                        for name in self.reward_function.COMPONENT_NAMES
                    },
                }
            )
        return final_infos

    def _extract_render_state(
        self,
        state: Any,
        parsed_actions: np.ndarray,
    ) -> Optional[Dict[str, Any]]:
        if state is None or not hasattr(state, "players") or not hasattr(state, "ball"):
            return None

        try:
            from rlgym_sim.utils import common_values as sim_common_values
        except ImportError:  # pragma: no cover
            from rlgym.utils import common_values as sim_common_values

        boost_locations = getattr(sim_common_values, "BOOST_LOCATIONS", [])
        boost_states = getattr(state, "boost_pads", [])

        cars = []
        for index, player in enumerate(state.players):
            car = player.car_data
            forward = np.asarray(car.forward(), dtype=np.float32)
            yaw = float(np.arctan2(forward[1], forward[0]))
            action = parsed_actions[index] if index < len(parsed_actions) else np.zeros(8, dtype=np.float32)
            cars.append(
                {
                    "car_id": int(player.car_id),
                    "team_num": int(player.team_num),
                    "position": [float(v) for v in car.position],
                    "velocity": [float(v) for v in car.linear_velocity],
                    "forward": [float(v) for v in forward],
                    "yaw": yaw,
                    "boost_amount": float(player.boost_amount),
                    "on_ground": bool(player.on_ground),
                    "has_flip": bool(player.has_flip),
                    "is_demoed": bool(player.is_demoed),
                    "height": float(car.position[2]),
                    "speed": float(np.linalg.norm(car.linear_velocity)),
                    "inputs": {
                        "throttle": float(action[0]),
                        "steer": float(action[1]),
                        "pitch": float(action[2]),
                        "yaw": float(action[3]),
                        "roll": float(action[4]),
                        "jump": bool(action[5] > 0.5),
                        "boost": bool(action[6] > 0.5),
                        "handbrake": bool(action[7] > 0.5),
                    },
                }
            )

        pads = []
        for index, location in enumerate(boost_locations):
            active = bool(boost_states[index]) if index < len(boost_states) else False
            pads.append(
                {
                    "position": [float(location[0]), float(location[1]), float(location[2])],
                    "is_large": bool(float(location[2]) > 71.0),
                    "active": active,
                }
            )

        return {
            "tick_skip": int(self.env_config["tick_skip"]),
            "score": {
                "blue": int(getattr(state, "blue_score", 0)),
                "orange": int(getattr(state, "orange_score", 0)),
            },
            "last_touch": int(getattr(state, "last_touch", 0) or 0),
            "ball": {
                "position": [float(v) for v in state.ball.position],
                "velocity": [float(v) for v in state.ball.linear_velocity],
                "height": float(state.ball.position[2]),
                "speed": float(np.linalg.norm(state.ball.linear_velocity)),
            },
            "cars": cars,
            "boost_pads": pads,
        }


def _reset_env(env: RLGymMatchAdapter) -> np.ndarray:
    return env.reset()


def _step_env(args: Tuple[RLGymMatchAdapter, Sequence[int]]):
    env, actions = args
    return env.step(actions)


class ThreadedVectorEnv:
    def __init__(self, env_fns: Iterable[Callable[[], RLGymMatchAdapter]]) -> None:
        self.envs = [fn() for fn in env_fns]
        self.executor = ThreadPoolExecutor(max_workers=len(self.envs))
        self.agents_per_env = [env.num_agents for env in self.envs]
        self.num_envs = len(self.envs)
        self.action_space_n = self.envs[0].action_space_n

    @property
    def num_actors(self) -> int:
        return int(sum(self.agents_per_env))

    def reset(self) -> np.ndarray:
        observations = list(self.executor.map(_reset_env, self.envs))
        self.agents_per_env = [obs.shape[0] for obs in observations]
        return np.concatenate(observations, axis=0)

    def step(self, flat_actions: Sequence[int]):
        actions = np.asarray(flat_actions, dtype=np.int64).reshape(-1)
        splits: List[np.ndarray] = []
        cursor = 0
        for agents in self.agents_per_env:
            splits.append(actions[cursor : cursor + agents])
            cursor += agents
        results = list(self.executor.map(_step_env, list(zip(self.envs, splits))))
        observations, rewards, dones, infos = zip(*results)
        return (
            np.concatenate(observations, axis=0),
            np.concatenate(rewards, axis=0),
            np.concatenate(dones, axis=0),
            [item for chunk in infos for item in chunk],
        )

    def update_curriculum(self, total_steps: int) -> None:
        for env in self.envs:
            env.set_training_step(total_steps)

    def close(self) -> None:
        for env in self.envs:
            env.close()
        self.executor.shutdown(wait=True)

    def get_render_state(self, env_index: int = 0) -> Optional[Dict[str, Any]]:
        return self.envs[env_index].get_render_state()


def build_env_factory(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    curriculum_config: Dict[str, Any],
) -> Callable[[], RLGymMatchAdapter]:
    def _factory() -> RLGymMatchAdapter:
        return RLGymMatchAdapter(env_config, reward_config, curriculum_config)

    return _factory


def build_vector_env(
    env_config: Dict[str, Any],
    reward_config: Dict[str, Any],
    curriculum_config: Dict[str, Any],
    num_envs: int | None = None,
) -> ThreadedVectorEnv:
    environment_count = int(num_envs or env_config["num_envs"])
    factory = build_env_factory(env_config, reward_config, curriculum_config)
    return ThreadedVectorEnv(factory for _ in range(environment_count))
