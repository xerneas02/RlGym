import torch
import torch.nn as nn
import numpy as np
import rlgym
from stable_baselines3 import PPO
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions import common_conditions
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED, BALL_MAX_SPEED
from numpy.linalg import norm

class CustomReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_touch = None
        self.start_goal = 0
        self.ticks = 0
        self.has_touched_ball = False

    def reset(self, initial_state: GameState):
        self.last_touch = initial_state.last_touch
        self.start_goal = initial_state.blue_score
        self.ticks = 0
        self.has_touched_ball = False

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist_ball = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        dist_reward = np.exp(-0.5 * dist_ball / CAR_MAX_SPEED)
        ball_speed = norm(state.ball.linear_velocity)
        car_speed = norm(state.players[0].car_data.linear_velocity)
        
        self.ticks += 1

        total_reward = car_speed/1000
        
        if(player.ball_touched):
            self.has_touched_ball = True
    
        return total_reward + (state.blue_score - self.start_goal)*5 - (self.ticks * (not self.has_touched_ball) * 0.01)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)