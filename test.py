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
from RlGym import CustomReward

model = PPO.load("rl_model")

# Create the environment separately
env = rlgym.make(game_speed=100, terminal_conditions=(common_conditions.TimeoutCondition(225), common_conditions.GoalScoredCondition()), reward_fn=CustomReward())

# Set the environment for the loaded model
model.set_env(env)


torch_model = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], 64),
    nn.Tanh(),
    nn.Linear(64, env.action_space.shape[0])
)

# Load the weights from the stable_baselines3 model
stable_baselines_weights = torch.load("rl_model.zip")
torch_model.load_state_dict(stable_baselines_weights)

# Save the PyTorch model
torch.save(torch_model.state_dict(), "pytorch_model.pth")