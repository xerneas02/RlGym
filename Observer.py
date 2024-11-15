import math
import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState, PhysicsObject
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils.common_values import BALL_MAX_SPEED, CAR_MAX_SPEED, CAR_MAX_ANG_VEL

class ZeerObservations(ObsBuilder):
    # Normalization distances
    POS_MAX = np.linalg.norm([4096,5120,2044])

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        
        obs = [ball.position / self.POS_MAX,
               ball.linear_velocity / BALL_MAX_SPEED,
               ball.angular_velocity / CAR_MAX_ANG_VEL,
               [np.linalg.norm(ball.linear_velocity / self.POS_MAX)],
               previous_action,
               pads]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_car.position - player_car.position) / self.POS_MAX,
                (other_car.linear_velocity - player_car.linear_velocity) / CAR_MAX_SPEED,
                [
                    np.linalg.norm((other_car.position - player_car.position) / self.POS_MAX),
                    np.linalg.norm((other_car.linear_velocity - player_car.linear_velocity) / CAR_MAX_SPEED)
                ]
                
            ])

        obs.extend(allies)
        obs.extend(enemies)
        
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        # Calculate car position relative to the ball, and car's velocity relative to the ball
        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        # Calculate the relative position of the car to the back of the goal it is attacking
        attack_goal = common_values.ORANGE_GOAL_BACK
        rel_attack = attack_goal - player_car.position
        defend_goal = common_values.BLUE_GOAL_BACK
        rel_defend = defend_goal - player_car.position
        super_sonic = np.linalg.norm(player_car.linear_velocity) >= common_values.SUPERSONIC_THRESHOLD

        obs.extend([
            rel_pos / self.POS_MAX,
            rel_vel / CAR_MAX_SPEED,
            rel_attack / self.POS_MAX,
            rel_defend / self.POS_MAX,
            player_car.position / self.POS_MAX,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / CAR_MAX_SPEED,
            player_car.angular_velocity / CAR_MAX_ANG_VEL,
            [player.boost_amount,
             int(super_sonic),
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car





class OldZeerObservations(ObsBuilder):
    # Normalization distances
    POS_STD = 2300 
    ANG_STD = math.pi

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:

        if player.team_num == common_values.ORANGE_TEAM:
            inverted = True
            ball = state.inverted_ball
            pads = state.inverted_boost_pads
        else:
            inverted = False
            ball = state.ball
            pads = state.boost_pads

        obs = [ball.position / self.POS_STD,
               ball.linear_velocity / self.POS_STD,
               ball.angular_velocity / self.ANG_STD,
               previous_action,
               pads]

        player_car = self._add_player_to_obs(obs, player, ball, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            other_car = self._add_player_to_obs(team_obs, other, ball, inverted)

            # Extra info
            team_obs.extend([
                (other_car.position - player_car.position) / self.POS_STD,
                (other_car.linear_velocity - player_car.linear_velocity) / self.POS_STD
            ])

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, ball: PhysicsObject, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        # Calculate car position relative to the ball, and car's velocity relative to the ball
        rel_pos = ball.position - player_car.position
        rel_vel = ball.linear_velocity - player_car.linear_velocity

        # Calculate the relative position of the car to the back of the goal it is attacking
        attack_goal = common_values.ORANGE_GOAL_BACK
        rel_attack = attack_goal - player_car.position
        defend_goal = common_values.BLUE_GOAL_BACK
        rel_defend = defend_goal - player_car.position

        obs.extend([
            rel_pos / self.POS_STD,
            rel_vel / self.POS_STD,
            rel_attack / self.POS_STD,
            rel_defend / self.POS_STD,
            player_car.position / self.POS_STD,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity / self.POS_STD,
            player_car.angular_velocity / self.ANG_STD,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car