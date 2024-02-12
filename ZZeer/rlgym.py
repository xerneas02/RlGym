"""
The action parser.
"""
import mathRlgym as math
from abc import ABC, abstractmethod
import gym.spaces
import numpy as np
from typing import List, Union, Tuple, Dict, Any

import numpy as np
from typing import List, Optional

from abc import ABC, abstractmethod
import gym
import numpy as np
from typing import Any
import numpy as np
from typing import Optional

import common_values
from rlgym_compat import PlayerData, GameState




"""
A class to represent the state of a physics object from the game.
"""

class PhysicsObject(object):
    def __init__(self, position=None, quaternion=None, linear_velocity=None, angular_velocity=None):
        self.position: np.ndarray = position if position is not None else np.zeros(3)
        self.quaternion: np.ndarray = quaternion if quaternion is not None else np.array([1, 0, 0, 0], dtype=np.float32)
        self.linear_velocity: np.ndarray = linear_velocity if linear_velocity is not None else np.zeros(3)
        self.angular_velocity: np.ndarray = angular_velocity if angular_velocity is not None else np.zeros(3)
        self._euler_angles: Optional[np.ndarray] = None
        self._rotation_mtx: Optional[np.ndarray] = None
        self._has_computed_rot_mtx = False
        self._has_computed_euler_angles = False

    def decode_car_data(self, car_data: np.ndarray):
        """
        Function to decode the physics state of a car from the game state array.
        :param car_data: Slice of game state array containing the car data to decode.
        """
        self.position = car_data[:3]
        self.quaternion = car_data[3:7]
        self.linear_velocity = car_data[7:10]
        self.angular_velocity = car_data[10:]

    def decode_ball_data(self, ball_data: np.ndarray):
        """
        Function to decode the physics state of the ball from the game state array.
        :param ball_data: Slice of game state array containing the ball data to decode.
        """
        self.position = ball_data[:3]
        self.linear_velocity = ball_data[3:6]
        self.angular_velocity = ball_data[6:9]

    def forward(self) -> np.ndarray:
        return self.rotation_mtx()[:, 0]

    def right(self) -> np.ndarray:
        return self.rotation_mtx()[:, 1]

    def left(self) -> np.ndarray:
        return self.rotation_mtx()[:, 1] * -1

    def up(self) -> np.ndarray:
        return self.rotation_mtx()[:, 2]

    def pitch(self) -> float:
        return self.euler_angles()[0]

    def yaw(self) -> float:
        return self.euler_angles()[1]

    def roll(self) -> float:
        return self.euler_angles()[2]

    # pitch, yaw, roll
    def euler_angles(self) -> np.ndarray:
        if not self._has_computed_euler_angles:
            self._euler_angles = math.quat_to_euler(self.quaternion)
            self._has_computed_euler_angles = True

        return self._euler_angles

    def rotation_mtx(self) -> np.ndarray:
        if not self._has_computed_rot_mtx:
            self._rotation_mtx = math.quat_to_rot_mtx(self.quaternion)
            self._has_computed_rot_mtx = True

        return self._rotation_mtx

    def serialize(self):
        """
        Function to serialize all the values contained by this physics object into a single 1D list. This can be useful
        when constructing observations for a policy.
        :return: List containing the serialized data.
        """
        repr = []

        if self.position is not None:
            for arg in self.position:
                repr.append(arg)
                
        if self.quaternion is not None:
            for arg in self.quaternion:
                repr.append(arg)
                
        if self.linear_velocity is not None:
            for arg in self.linear_velocity:
                repr.append(arg)
                
        if self.angular_velocity is not None:
            for arg in self.angular_velocity:
                repr.append(arg)

        if self._euler_angles is not None:
            for arg in self._euler_angles:
                repr.append(arg)

        if self._rotation_mtx is not None:
            for arg in self._rotation_mtx.ravel():
                repr.append(arg)

        return repr


class ActionParser(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def get_action_space(self) -> gym.spaces.Space:
        """
        Function that returns the action space type. It will be called during the initialization of the environment.
        
        :return: The type of the action space
        """
        raise NotImplementedError

    @abstractmethod
    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        """
        Function that parses actions from the action space into a format that rlgym understands.
        The expected return value is a numpy float array of size (n, 8) where n is the number of agents.
        The second dimension is indexed as follows: throttle, steer, yaw, pitch, roll, jump, boost, handbrake.
        The first five values are expected to be in the range [-1, 1], while the last three values should be either 0 or 1.

        :param actions: An object of actions, as passed to the `env.step` function.
        :param state: The GameState object of the current state that were used to generate the actions.

        :return: the parsed actions in the rlgym format.
        """
        raise NotImplementedError


"""
The observation builder.
"""




class ObsBuilder(ABC):

    def __init__(self):
        pass

    def get_obs_space(self) -> gym.spaces.Space:
        """
        Function that returns the observation space type. It will be called during the initialization of the environment.

        :return: The type of the observation space
        """
        pass

    @abstractmethod
    def reset(self, initial_state: GameState):
        """
        Function to be called each time the environment is reset. Note that this does not need to return anything,
        the environment will call `build_obs` automatically after reset, so the initial observation for a policy will be
        constructed in the same way as every other observation.

        :param initial_state: The initial game state of the reset environment.
        """
        raise NotImplementedError

    def pre_step(self, state: GameState):
        """
        Function to pre-compute values each step. This function is called only once each step, before build_obs is
        called for each player.

        :param state: The current state of the game.
        """
        pass

    @abstractmethod
    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        """
        Function to build observations for a policy. This is where all observations will be constructed every step and
        every reset. This function is given a player argument, and it is expected that the observation returned by this
        function will contain information from the perspective of that player. This function is called once for each
        agent automatically at every step.

        :param player: The player to build an observation for. The observation returned should be from the perspective of
        this player.

        :param state: The current state of the game.
        :param previous_action: The action taken at the previous environment step.

        :return: An observation for the player provided.
        """
        raise NotImplementedError




