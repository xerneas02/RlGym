import torch
import torch.nn as nn
import numpy as np
import rlgym
from rlgym.envs import Match
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions import common_conditions
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED
from numpy.linalg import norm
from rlgym.utils.reward_functions import DefaultReward
from rlgym.utils.obs_builders import DefaultObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.action_parsers import DefaultAction
from rlgym.gym import Gym
from rlgym.gamelaunch import LaunchPreference
import os

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



def get_match():

    match = Match(
        game_speed=100,
        reward_function=CustomReward(),
        terminal_conditions=(common_conditions.TimeoutCondition(225), common_conditions.NoTouchTimeoutCondition(50)),
        obs_builder=DefaultObs(),
        state_setter=DefaultState(),
        action_parser = DefaultAction()
    )
    
    return match

def get_gym():
    return Gym(get_match(), pipe_id=os.getpid(), launch_preference=LaunchPreference.EPIC, use_injector=False, force_paging=False, raise_on_crash=False, auto_minimize=False)
    

if __name__ == "__main__":
    #env = SB3MultipleInstanceEnv(match_func_or_matches=get_match, num_instances=1, wait_time=40, force_paging=True)
    env = get_gym()

    for _ in range(10):
        model = PPO.load("ZZeer/rl_model", env=env, verbose=1)
        #model = PPO("MlpPolicy", env=env, verbose=1)
        model.learn(total_timesteps=int(1e5))
        model.save("ZZeer/rl_model")
