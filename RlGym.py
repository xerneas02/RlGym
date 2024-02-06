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
from Observer import ObsBuilderBluePerspective
from Reward import CustomReward
from Terminal import BallTouchCondition
from State import StateSetterInit

env = rlgym.make(
    game_speed=100, 
    tick_skip= 8, 
    terminal_conditions=(common_conditions.TimeoutCondition(225), 
                         common_conditions.GoalScoredCondition(), 
                         BallTouchCondition()), 
    reward_fn=CustomReward(), 
    state_setter=StateSetterInit(),
    obs_builder=ObsBuilderBluePerspective())

model = PPO.load("rl_model")
model.set_env(env)
model.learn(total_timesteps=int(1e6))
model.save("rl_model")
