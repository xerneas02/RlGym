import math
from rlgym.utils.obs_builders import ObsBuilder
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from typing import Any, List
import numpy as np

class ObsBuilderBluePerspective(ObsBuilder):   
    def __init__(self, pos_coef=1/2300, ang_coef=1/math.pi, lin_vel_coef=1/2300, ang_vel_coef=1/math.pi):
        """
        :param pos_coef: Position normalization coefficient
        :param ang_coef: Rotation angle normalization coefficient
        :param lin_vel_coef: Linear velocity normalization coefficient
        :param ang_vel_coef: Angular velocity normalization coefficient
        """
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef
        
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

        obs = [ball.position * self.POS_COEF,
               ball.linear_velocity * self.LIN_VEL_COEF,
               ball.angular_velocity * self.ANG_VEL_COEF,
               previous_action,
               pads]

        self._add_player_to_obs(obs, player, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            self._add_player_to_obs(team_obs, other, inverted)

        obs.extend(allies)
        obs.extend(enemies)
        return np.concatenate(obs)
      
    def _add_player_to_obs(self, obs: List, player: PlayerData, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs.extend([
            player_car.position * self.POS_COEF,
            player_car.forward(),
            player_car.up(),
            player_car.linear_velocity * self.LIN_VEL_COEF,
            player_car.angular_velocity * self.ANG_VEL_COEF,
            [player.boost_amount,
             int(player.on_ground),
             int(player.has_flip),
             int(player.is_demoed)]])

        return player_car
    
import math
import numpy as np
from typing import Any, List
from rlgym.utils import common_values
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.obs_builders import ObsBuilder


class CustomObsBuilder(ObsBuilder):
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

        obs = [ball.position[0] / 2300,  # Normalize ball x position
               ball.position[1] / 2300,  # Normalize ball y position
               ball.position[2] / 2300,  # Normalize ball z position
               ball.linear_velocity[0] / 2300,  # Normalize ball x linear velocity
               ball.linear_velocity[1] / 2300,  # Normalize ball y linear velocity
               ball.linear_velocity[2] / 2300,  # Normalize ball z linear velocity
               ball.angular_velocity[0] / math.pi,  # Normalize ball x angular velocity
               ball.angular_velocity[1] / math.pi,  # Normalize ball y angular velocity
               ball.angular_velocity[2] / math.pi,  # Normalize ball z angular velocity
               previous_action[0],  # Assuming previous_action is a scalar
               previous_action[1],  # Assuming previous_action is a scalar
               player.boost_amount / 100,  # Normalize player's boost amount
               int(player.on_ground),  # Convert boolean to integer
               int(player.has_flip),  # Convert boolean to integer
               int(player.is_demoed)]  # Convert boolean to integer

        self._add_player_to_obs(obs, player, inverted)

        allies = []
        enemies = []

        for other in state.players:
            if other.car_id == player.car_id:
                continue

            if other.team_num == player.team_num:
                team_obs = allies
            else:
                team_obs = enemies

            self._add_player_to_obs(team_obs, other, inverted)

        # Pad to match the desired shape (total 70 elements)
        obs.extend([0.0] * (70 - len(obs)))

        return np.array(obs)

    def _add_player_to_obs(self, obs: List, player: PlayerData, inverted: bool):
        if inverted:
            player_car = player.inverted_car_data
        else:
            player_car = player.car_data

        obs.extend([
            player_car.position[0] / 2300,  # Normalize player x position
            player_car.position[1] / 2300,  # Normalize player y position
            player_car.position[2] / 2300,  # Normalize player z position
            player_car.forward()[0],  # Assuming forward() returns a vector
            player_car.forward()[1],  # Assuming forward() returns a vector
            player_car.forward()[2],  # Assuming forward() returns a vector
            player_car.up()[0],  # Assuming up() returns a vector
            player_car.up()[1],  # Assuming up() returns a vector
            player_car.up()[2],  # Assuming up() returns a vector
            player_car.linear_velocity[0] / 2300,  # Normalize player x linear velocity
            player_car.linear_velocity[1] / 2300,  # Normalize player y linear velocity
            player_car.linear_velocity[2] / 2300,  # Normalize player z linear velocity
            player_car.angular_velocity[0] / math.pi,  # Normalize player x angular velocity
            player_car.angular_velocity[1] / math.pi,  # Normalize player y angular velocity
            player_car.angular_velocity[2] / math.pi,  # Normalize player z angular velocity
            player.boost_amount / 100,  # Normalize player's boost amount
            int(player.on_ground),  # Convert boolean to integer
            int(player.has_flip),  # Convert boolean to integer
            int(player.is_demoed)])  # Convert boolean to integer

        return player_car
