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


class BallTouchCondition(TerminalCondition):
    
    def __init__(self):
        super().__init__()
        self.last_touch = None
        
    def reset(self, initial_state: GameState):
        self.last_touch = initial_state.last_touch

    def is_terminal(self, current_state: GameState) -> bool:
        return current_state.last_touch != self.last_touch
    
#TimeOut condition mais reset la première fois qu'il touche la balle et quand un but est marqué
class NoTouchOrGoalTimeoutCondition(common_conditions.TimeoutCondition):
    def __init__(self, time):
        super().__init__(time)
        self.goal_score = 0
        self.first = True
    
    def reset(self, initial_state: GameState):
        super(NoTouchOrGoalTimeoutCondition, self).reset(initial_state)
        self.goal = initial_state.orange_score + initial_state.blue_score
        
    def is_terminal(self, current_state: GameState):
        if self.goal != current_state.orange_score + current_state.blue_score:
            self.steps = 0
            self.goal = current_state.orange_score + current_state.blue_score
            return False
        elif any(p.ball_touched for p in current_state.players) and self.first:
            self.first = False
            self.steps = 0
            self.max_steps = 300
            return False
        else:
            return super(NoTouchOrGoalTimeoutCondition, self).is_terminal(current_state)