from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

try:
    from rlgym_sim.utils import common_values
    from rlgym_sim.utils.common_values import BALL_MAX_SPEED, CAR_MAX_ANG_VEL, CAR_MAX_SPEED
    from rlgym_sim.utils.gamestates import GameState, PhysicsObject, PlayerData
    from rlgym_sim.utils.obs_builders import ObsBuilder
except ImportError:  # pragma: no cover
    from rlgym.utils import common_values
    from rlgym.utils.common_values import BALL_MAX_SPEED, CAR_MAX_ANG_VEL, CAR_MAX_SPEED
    from rlgym.utils.gamestates import GameState, PhysicsObject, PlayerData
    from rlgym.utils.obs_builders import ObsBuilder


POS_SCALE = np.asarray([4096.0, 5120.0, 2044.0], dtype=np.float32)
FIELD_DIAGONAL = float(np.linalg.norm(POS_SCALE))


class ObservationSlices:
    ball_position = slice(0, 3)
    ball_velocity = slice(3, 6)
    ball_angular_velocity = slice(6, 9)
    self_position = slice(9, 12)
    self_velocity = slice(12, 15)
    self_angular_velocity = slice(15, 18)
    self_forward = slice(18, 21)
    self_up = slice(21, 24)
    self_flags = slice(24, 28)
    relative_ball_position = slice(28, 31)
    relative_ball_velocity = slice(31, 34)
    relative_ball_distance = slice(34, 35)
    attack_goal_vector = slice(35, 38)
    attack_goal_distance = slice(38, 39)
    defend_goal_vector = slice(39, 42)
    defend_goal_distance = slice(42, 43)
    wall_distances = slice(43, 47)
    boost_pads = slice(47, 81)
    opponent_position = slice(81, 84)
    opponent_velocity = slice(84, 87)
    opponent_angular_velocity = slice(87, 90)
    opponent_forward = slice(90, 93)
    opponent_up = slice(93, 96)
    opponent_flags = slice(96, 100)
    self_to_opponent_position = slice(100, 103)
    self_to_opponent_velocity = slice(103, 106)
    self_to_opponent_distance = slice(106, 107)
    opponent_to_ball_position = slice(107, 110)
    opponent_to_ball_velocity = slice(110, 113)
    opponent_to_ball_distance = slice(113, 114)
    ball_to_attack_goal = slice(114, 117)
    ball_to_attack_goal_distance = slice(117, 118)
    previous_action = slice(118, 126)


class CompactObservationBuilder(ObsBuilder):
    OBSERVATION_DIM = 126
    SLICES = ObservationSlices()

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def observation_spec() -> Dict[str, slice]:
        return {
            key: value
            for key, value in ObservationSlices.__dict__.items()
            if not key.startswith("_") and isinstance(value, slice)
        }

    def reset(self, initial_state: GameState) -> None:
        return None

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        attack_goal = np.asarray(common_values.ORANGE_GOAL_BACK, dtype=np.float32)
        defend_goal = np.asarray(common_values.BLUE_GOAL_BACK, dtype=np.float32)

        if player.team_num == common_values.ORANGE_TEAM:
            ball = state.inverted_ball
            boost_pads = getattr(state, "inverted_boost_pads", state.boost_pads)
            player_car = player.inverted_car_data
            candidate_pairs = [
                (
                    other,
                    other.inverted_car_data if hasattr(other, "inverted_car_data") else other.car_data,
                )
                for other in state.players
                if other.car_id != player.car_id
            ]
        else:
            ball = state.ball
            boost_pads = state.boost_pads
            player_car = player.car_data
            candidate_pairs = [
                (other, other.car_data)
                for other in state.players
                if other.car_id != player.car_id
            ]

        opponent_candidates = [
            (other, car_data)
            for other, car_data in candidate_pairs
            if other.team_num != player.team_num
        ]
        selected_candidates = opponent_candidates or candidate_pairs
        opponent_car: Optional[PhysicsObject] = None
        opponent_player: Optional[PlayerData] = None
        if selected_candidates:
            opponent_player, opponent_car = min(
                selected_candidates,
                key=lambda item: float(np.linalg.norm(item[1].position - player_car.position)),
            )

        rel_ball_position = ball.position - player_car.position
        rel_ball_velocity = ball.linear_velocity - player_car.linear_velocity
        attack_goal_vec = attack_goal - player_car.position
        defend_goal_vec = defend_goal - player_car.position
        wall_distances = self._wall_distances(player_car.position)
        ball_to_attack_goal = attack_goal - ball.position

        features: List[np.ndarray] = [
            self._norm_pos(ball.position),
            self._norm_vel(ball.linear_velocity, BALL_MAX_SPEED),
            self._norm_vel(ball.angular_velocity, CAR_MAX_ANG_VEL),
            self._norm_pos(player_car.position),
            self._norm_vel(player_car.linear_velocity, CAR_MAX_SPEED),
            self._norm_vel(player_car.angular_velocity, CAR_MAX_ANG_VEL),
            np.asarray(player_car.forward(), dtype=np.float32),
            np.asarray(player_car.up(), dtype=np.float32),
            np.asarray(
                [
                    float(player.boost_amount),
                    float(player.on_ground),
                    float(player.has_flip),
                    float(player.is_demoed),
                ],
                dtype=np.float32,
            ),
            self._norm_pos(rel_ball_position),
            self._norm_vel(rel_ball_velocity, CAR_MAX_SPEED),
            np.asarray([self._norm_distance(rel_ball_position)], dtype=np.float32),
            self._norm_pos(attack_goal_vec),
            np.asarray([self._norm_distance(attack_goal_vec)], dtype=np.float32),
            self._norm_pos(defend_goal_vec),
            np.asarray([self._norm_distance(defend_goal_vec)], dtype=np.float32),
            wall_distances,
            np.asarray(boost_pads, dtype=np.float32),
        ]

        if opponent_car is None or opponent_player is None:
            features.extend(
                [
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(4, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(3, dtype=np.float32),
                    np.zeros(1, dtype=np.float32),
                ]
            )
        else:
            self_to_opponent_pos = opponent_car.position - player_car.position
            self_to_opponent_vel = opponent_car.linear_velocity - player_car.linear_velocity
            opponent_to_ball_pos = ball.position - opponent_car.position
            opponent_to_ball_vel = ball.linear_velocity - opponent_car.linear_velocity
            features.extend(
                [
                    self._norm_pos(opponent_car.position),
                    self._norm_vel(opponent_car.linear_velocity, CAR_MAX_SPEED),
                    self._norm_vel(opponent_car.angular_velocity, CAR_MAX_ANG_VEL),
                    np.asarray(opponent_car.forward(), dtype=np.float32),
                    np.asarray(opponent_car.up(), dtype=np.float32),
                    np.asarray(
                        [
                            float(opponent_player.boost_amount),
                            float(opponent_player.on_ground),
                            float(opponent_player.has_flip),
                            float(opponent_player.is_demoed),
                        ],
                        dtype=np.float32,
                    ),
                    self._norm_pos(self_to_opponent_pos),
                    self._norm_vel(self_to_opponent_vel, CAR_MAX_SPEED),
                    np.asarray([self._norm_distance(self_to_opponent_pos)], dtype=np.float32),
                    self._norm_pos(opponent_to_ball_pos),
                    self._norm_vel(opponent_to_ball_vel, CAR_MAX_SPEED),
                    np.asarray([self._norm_distance(opponent_to_ball_pos)], dtype=np.float32),
                ]
            )

        features.extend(
            [
                self._norm_pos(ball_to_attack_goal),
                np.asarray([self._norm_distance(ball_to_attack_goal)], dtype=np.float32),
                np.asarray(previous_action, dtype=np.float32).reshape(-1),
            ]
        )

        observation = np.concatenate(features, dtype=np.float32)
        if observation.shape[0] != self.OBSERVATION_DIM:
            raise ValueError(
                f"Expected observation dim {self.OBSERVATION_DIM}, got {observation.shape[0]}"
            )
        return observation

    @staticmethod
    def _norm_pos(vector: np.ndarray) -> np.ndarray:
        return np.asarray(vector, dtype=np.float32) / POS_SCALE

    @staticmethod
    def _norm_vel(vector: np.ndarray, max_norm: float) -> np.ndarray:
        return np.asarray(vector, dtype=np.float32) / float(max_norm)

    @staticmethod
    def _norm_distance(vector: np.ndarray) -> float:
        return float(np.linalg.norm(vector) / FIELD_DIAGONAL)

    @staticmethod
    def _wall_distances(position: np.ndarray) -> np.ndarray:
        x, y, z = [float(v) for v in position]
        return np.asarray(
            [
                (4096.0 - abs(x)) / 4096.0,
                (5120.0 + y) / 10240.0,
                (5120.0 - y) / 10240.0,
                (2044.0 - z) / 2044.0,
            ],
            dtype=np.float32,
        )
