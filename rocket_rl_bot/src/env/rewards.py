from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

try:
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.common_values import BALL_MAX_SPEED, CAR_MAX_SPEED
    from rlgym_sim.utils.gamestates import GameState, PlayerData
    from rlgym_sim.utils.reward_functions import RewardFunction
except ImportError:  # pragma: no cover
    from rlgym.utils import common_values
    from rlgym.utils.common_values import BALL_MAX_SPEED, CAR_MAX_SPEED
    from rlgym.utils.gamestates import GameState, PlayerData
    from rlgym.utils.reward_functions import RewardFunction


class MinimalRewardFunction(RewardFunction):
    COMPONENT_NAMES = (
        "goal_reward",
        "ball_goal_progress",
        "velocity_to_ball",
        "touch_reward",
        "defense_position",
        "boost_efficiency",
    )

    def __init__(self, weights: Dict[str, float]) -> None:
        super().__init__()
        self.weights = {name: float(weights[name]) for name in self.COMPONENT_NAMES}
        self._prev_blue_score = 0
        self._prev_orange_score = 0
        self._prev_boost: Dict[int, float] = {}
        self._prev_touch: Dict[int, bool] = {}
        self._step_signature: Optional[tuple] = None
        self._shared_step: Dict[str, float] = {"blue_delta": 0.0, "orange_delta": 0.0}
        self._current_step_components: List[Dict[str, float]] = []

    def reset(self, initial_state: GameState) -> None:
        self._prev_blue_score = int(initial_state.blue_score)
        self._prev_orange_score = int(initial_state.orange_score)
        self._prev_boost.clear()
        self._prev_touch.clear()
        self._step_signature = None
        self._shared_step = {"blue_delta": 0.0, "orange_delta": 0.0}
        self._current_step_components = []

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        self._prepare_step(state)

        if player.team_num == common_values.BLUE_TEAM:
            opponent_goal = np.asarray(common_values.ORANGE_GOAL_BACK, dtype=np.float32)
            own_goal = np.asarray(common_values.BLUE_GOAL_BACK, dtype=np.float32)
            attack_sign = 1.0
            goal_component = self._shared_step["blue_delta"] - self._shared_step["orange_delta"]
        else:
            opponent_goal = np.asarray(common_values.BLUE_GOAL_BACK, dtype=np.float32)
            own_goal = np.asarray(common_values.ORANGE_GOAL_BACK, dtype=np.float32)
            attack_sign = -1.0
            goal_component = self._shared_step["orange_delta"] - self._shared_step["blue_delta"]

        car = player.car_data
        ball = state.ball
        car_velocity = np.asarray(car.linear_velocity, dtype=np.float32)
        ball_velocity = np.asarray(ball.linear_velocity, dtype=np.float32)
        rel_ball = np.asarray(ball.position - car.position, dtype=np.float32)
        rel_ball_norm = float(np.linalg.norm(rel_ball))

        direction_to_goal = opponent_goal - np.asarray(ball.position, dtype=np.float32)
        goal_direction_norm = np.linalg.norm(direction_to_goal) + 1e-6
        ball_goal_progress = float(np.dot(ball_velocity, direction_to_goal / goal_direction_norm) / BALL_MAX_SPEED)
        ball_goal_progress = float(np.clip(ball_goal_progress, -1.0, 1.0))

        direction_to_ball = rel_ball / (rel_ball_norm + 1e-6)
        velocity_to_ball = float(np.dot(car_velocity, direction_to_ball) / CAR_MAX_SPEED)
        velocity_to_ball = float(np.clip(max(0.0, velocity_to_ball), 0.0, 1.0))

        current_touch = bool(getattr(player, "ball_touched", False))
        previous_touch = self._prev_touch.get(player.car_id, False)
        touch_reward = 0.0
        if current_touch and not previous_touch:
            ball_speed = float(np.linalg.norm(ball_velocity) / BALL_MAX_SPEED)
            touch_reward = 1.0 + 0.5 * max(0.0, ball_goal_progress) + 0.25 * max(0.0, ball_speed - 0.2)
            touch_reward = float(np.clip(touch_reward, 0.0, 1.6))

        ball_on_own_half = attack_sign * float(ball.position[1]) < 0.0
        own_goal_to_ball = float(np.linalg.norm(own_goal - np.asarray(ball.position, dtype=np.float32)))
        own_goal_to_player = float(np.linalg.norm(own_goal - np.asarray(car.position, dtype=np.float32)))
        lateral_error = min(abs(float(car.position[0] - ball.position[0])), 3000.0) / 3000.0
        distance_factor = min(rel_ball_norm, 6000.0) / 6000.0
        defense_position = 0.0
        if ball_on_own_half:
            goal_side = 1.0 if own_goal_to_player < own_goal_to_ball else 0.0
            defense_position = goal_side * (1.0 - lateral_error) * (1.0 - distance_factor)

        previous_boost = self._prev_boost.get(player.car_id, float(player.boost_amount))
        boost_delta = max(0.0, previous_boost - float(player.boost_amount))
        speed_norm = float(np.linalg.norm(car_velocity) / CAR_MAX_SPEED)
        boost_pressed = float(previous_action[6]) if previous_action is not None and len(previous_action) >= 7 else 0.0
        boost_efficiency = speed_norm * (1.0 - min(boost_delta * 2.5, 1.0))
        if boost_pressed > 0.5 and speed_norm > 0.95:
            boost_efficiency -= 0.5
        boost_efficiency = float(np.clip(boost_efficiency, -1.0, 1.0))

        components = {
            "goal_reward": float(goal_component),
            "ball_goal_progress": float(ball_goal_progress),
            "velocity_to_ball": float(velocity_to_ball),
            "touch_reward": float(touch_reward),
            "defense_position": float(defense_position),
            "boost_efficiency": float(boost_efficiency),
            "ball_touched": float(current_touch and not previous_touch),
            "speed": float(np.linalg.norm(car_velocity)),
            "ball_distance_to_goal": float(np.linalg.norm(direction_to_goal)),
            "goals_for": float(max(0.0, goal_component)),
            "goals_against": float(max(0.0, -goal_component)),
        }
        self._current_step_components.append(components)

        self._prev_boost[player.car_id] = float(player.boost_amount)
        self._prev_touch[player.car_id] = current_touch

        total_reward = sum(self.weights[name] * components[name] for name in self.COMPONENT_NAMES)
        return float(total_reward)

    def consume_step_components(self, expected_agents: int) -> List[Dict[str, float]]:
        if len(self._current_step_components) < expected_agents:
            missing = expected_agents - len(self._current_step_components)
            self._current_step_components.extend({name: 0.0 for name in self.COMPONENT_NAMES} for _ in range(missing))
        components = self._current_step_components[:expected_agents]
        self._current_step_components = self._current_step_components[expected_agents:]
        return components

    def reward_weights(self) -> Dict[str, float]:
        return dict(self.weights)

    def _prepare_step(self, state: GameState) -> None:
        signature = (
            int(state.blue_score),
            int(state.orange_score),
            round(float(state.ball.position[0]), 2),
            round(float(state.ball.position[1]), 2),
            round(float(state.ball.position[2]), 2),
            round(float(state.ball.linear_velocity[0]), 2),
            round(float(state.ball.linear_velocity[1]), 2),
            round(float(state.ball.linear_velocity[2]), 2),
        )
        if signature == self._step_signature:
            return

        self._step_signature = signature
        blue_delta = int(state.blue_score) - self._prev_blue_score
        orange_delta = int(state.orange_score) - self._prev_orange_score
        self._shared_step = {"blue_delta": float(blue_delta), "orange_delta": float(orange_delta)}
        self._prev_blue_score = int(state.blue_score)
        self._prev_orange_score = int(state.orange_score)
        self._current_step_components = []
