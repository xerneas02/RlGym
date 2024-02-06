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


class CustomReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_touch = None

    def reset(self, initial_state: GameState):
        self.last_touch = initial_state.last_touch

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist_ball = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        dist_reward = np.exp(-0.5 * dist_ball / CAR_MAX_SPEED)
        ball_speed = norm(state.ball.linear_velocity)

        total_reward = dist_reward + player.on_ground*0.3 + 2*ball_speed/BALL_MAX_SPEED + player.match_goals*10

        return total_reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)



env = rlgym.make(game_speed=100, terminal_conditions=(common_conditions.TimeoutCondition(225), common_conditions.GoalScoredCondition()), reward_fn=CustomReward())

class CustomPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(CustomPolicy, self).__init__()
        self.mlp_extractor = model.policy.mlp_extractor
        self.action_net = model.policy.action_net
        self.value_net = model.policy.value_net

    def forward(self, x):
        features = self.mlp_extractor(x)
        action_out = self.action_net(features)
        value_out = self.value_net(features)
        return action_out, value_out

model = PPO.load("rl_model")
model.set_env(env)
model.learn(total_timesteps=int(1e6))
model.save("rl_model")
