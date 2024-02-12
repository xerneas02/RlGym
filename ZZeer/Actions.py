from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket

import numpy as np


from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete

from typing import Any, List
from rlgym import *

def int_to_binary_array(num):
    binary_string = bin(num)[2:].zfill(8)  # Convert integer to binary string
    binary_array = np.array([int(bit) for bit in binary_string])  # Convert binary string to array of integers
    return binary_array

class ZeerLookupAction(ActionParser):
    def __init__(self, bins=None):
        super().__init__()
    
        self.bins = [
            (-1, 0, 1),            #throttle
            (-1, -0.5, 0, 0.5, 1), #steer
            (-1, -0.5, 0, 0.5, 1), #pitch
            (-1, -0.5, 0, 0.5, 1), #yaw
            (-1, 0, 1)             #roll
            ] 
        
        self._lookup_table = self.make_lookup_table(self.bins)

    @staticmethod
    def make_lookup_table(bins):
        actions = []
        # Ground
        for throttle in bins[0]:
            for steer in bins[1]:
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in bins[2]:
            for yaw in bins[3]:
                for roll in bins[4]:
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        return self._lookup_table[actions]
    
