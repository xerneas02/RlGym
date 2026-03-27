from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

import common_values

BALL_MAX_SPEED = float(common_values.BALL_MAX_SPEED)
CAR_MAX_SPEED = float(common_values.CAR_MAX_SPEED)
CAR_MAX_ANG_VEL = float(common_values.CAR_MAX_ANG_VEL)

POS_SCALE = np.asarray([4096.0, 5120.0, 2044.0], dtype=np.float32)
FIELD_DIAGONAL = float(np.linalg.norm(POS_SCALE))


class CompactObservationBuilder:
    OBSERVATION_DIM = 126

    def reset(self, initial_state: Any) -> None:
        return None

    def build_obs(self, player: Any, state: Any, previous_action: np.ndarray) -> np.ndarray:
        attack_goal = np.asarray(common_values.ORANGE_GOAL_BACK, dtype=np.float32)
        defend_goal = np.asarray(common_values.BLUE_GOAL_BACK, dtype=np.float32)

        if int(player.team_num) == int(common_values.ORANGE_TEAM):
            ball = state.inverted_ball
            boost_pads = getattr(state, "inverted_boost_pads", state.boost_pads)
            player_car = player.inverted_car_data
            other_cars = [
                other.inverted_car_data if hasattr(other, "inverted_car_data") else other.car_data
                for other in state.players
                if other.car_id != player.car_id
            ]
            other_players = [other for other in state.players if other.car_id != player.car_id]
        else:
            ball = state.ball
            boost_pads = state.boost_pads
            player_car = player.car_data
            other_cars = [other.car_data for other in state.players if other.car_id != player.car_id]
            other_players = [other for other in state.players if other.car_id != player.car_id]

        opponent_car: Optional[Any] = other_cars[0] if other_cars else None
        opponent_player: Optional[Any] = other_players[0] if other_players else None

        rel_ball_position = np.asarray(ball.position, dtype=np.float32) - np.asarray(player_car.position, dtype=np.float32)
        rel_ball_velocity = np.asarray(ball.linear_velocity, dtype=np.float32) - np.asarray(player_car.linear_velocity, dtype=np.float32)
        attack_goal_vec = attack_goal - np.asarray(player_car.position, dtype=np.float32)
        defend_goal_vec = defend_goal - np.asarray(player_car.position, dtype=np.float32)
        wall_distances = self._wall_distances(np.asarray(player_car.position, dtype=np.float32))
        ball_to_attack_goal = attack_goal - np.asarray(ball.position, dtype=np.float32)

        features: List[np.ndarray] = [
            self._norm_pos(np.asarray(ball.position, dtype=np.float32)),
            self._norm_vel(np.asarray(ball.linear_velocity, dtype=np.float32), BALL_MAX_SPEED),
            self._norm_vel(np.asarray(ball.angular_velocity, dtype=np.float32), CAR_MAX_ANG_VEL),
            self._norm_pos(np.asarray(player_car.position, dtype=np.float32)),
            self._norm_vel(np.asarray(player_car.linear_velocity, dtype=np.float32), CAR_MAX_SPEED),
            self._norm_vel(np.asarray(player_car.angular_velocity, dtype=np.float32), CAR_MAX_ANG_VEL),
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
            opponent_pos = np.asarray(opponent_car.position, dtype=np.float32)
            opponent_vel = np.asarray(opponent_car.linear_velocity, dtype=np.float32)
            opponent_ang = np.asarray(opponent_car.angular_velocity, dtype=np.float32)
            self_pos = np.asarray(player_car.position, dtype=np.float32)
            self_vel = np.asarray(player_car.linear_velocity, dtype=np.float32)
            ball_pos = np.asarray(ball.position, dtype=np.float32)
            ball_vel = np.asarray(ball.linear_velocity, dtype=np.float32)

            self_to_opponent_pos = opponent_pos - self_pos
            self_to_opponent_vel = opponent_vel - self_vel
            opponent_to_ball_pos = ball_pos - opponent_pos
            opponent_to_ball_vel = ball_vel - opponent_vel

            features.extend(
                [
                    self._norm_pos(opponent_pos),
                    self._norm_vel(opponent_vel, CAR_MAX_SPEED),
                    self._norm_vel(opponent_ang, CAR_MAX_ANG_VEL),
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
            raise ValueError(f"Expected observation dim {self.OBSERVATION_DIM}, got {observation.shape[0]}")
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
