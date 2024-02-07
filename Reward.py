import numpy as np
import rlgym
from stable_baselines3 import PPO
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions import common_conditions
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED, BALL_MAX_SPEED, ORANGE_GOAL_CENTER, BACK_WALL_Y
from numpy.linalg import norm
from abc import ABC, abstractmethod
import numpy as np
import numpy as np
from scipy.spatial.distance import cosine


class CustomReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.last_touch = None
        self.start_goal_blue = 0
        self.start_goal_orange = 0
        self.ticks = 0
        self.has_touched_ball = False

    def reset(self, initial_state: GameState):
        self.last_touch = initial_state.last_touch
        self.start_goal_blue = initial_state.blue_score
        self.start_goal_orange = initial_state.orange_score
        self.ticks = 0
        self.has_touched_ball = False

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        dist_ball = np.linalg.norm(player.car_data.position - state.ball.position) - BALL_RADIUS
        dist_reward = np.exp(-0.5 * dist_ball / CAR_MAX_SPEED)
        ball_speed = norm(state.ball.linear_velocity)
        car_speed = norm(state.players[0].car_data.linear_velocity)
        
        self.ticks += 1

        total_reward = car_speed/500 + ((state.blue_score - self.start_goal_blue)*5 - (state.orange_score - self.start_goal_orange))*5 - (self.ticks * (not self.has_touched_ball) * 0.01) + (self.ticks * (self.has_touched_ball) * 0.01)
        
        if(player.ball_touched):
            self.has_touched_ball = True
    
        return total_reward

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action) + (not self.has_touched_ball)*-100


class GoalScoredReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_speed = np.linalg.norm(state.ball.linear_velocity, 2)**2
        return 1.0 + 0.5 * ball_speed / (BALL_MAX_SPEED)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class BoostDifferenceReward(RewardFunction):
    def __init__(self):
        self.previous_boost = None

    def reset(self, initial_state):
        self.previous_boost = initial_state.players[0].boost_amount

    def get_reward(self, player, state, previous_action):
        current_boost = player.boost_amount
        reward = np.sqrt(current_boost/100) - np.sqrt(self.previous_boost/100)
        self.previous_boost = current_boost
        return reward

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class BallTouchReward(RewardFunction):
    def __init__(self):
        self.last_touch = False
        self.lamb = 0

    def reset(self, initial_state):
        self.last_touch = False
        self.lamb = 0

    def get_reward(self, player, state, previous_action):
        if player.ball_touched:
            self.lamb = max(0.1, self.lamb * 0.95)
        else:
            self.lamb = min(1.0, self.lamb + 0.013)
            
        pos_ball_z = state.ball.position[2]
        reward = self.lamb * ((pos_ball_z + BALL_RADIUS)/(2*BALL_RADIUS)) ** 0.2836
        
        return reward

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class DemoReward(RewardFunction):
    def __init__(self):
        self.last_demo_count = 0
        
    def reset(self, initial_state):
        self.last_demo_count = 0

    def get_reward(self, player, state, previous_action):
        if player.match_demolishes != self.last_demo_count:
            self.last_demo_count = player.match_demolishes
            return 1
        
        return 0

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class DistancePlayerBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_position = player.car_data.position
        ball_position = state.ball.position
        
        distance = np.linalg.norm(car_position - ball_position, 2)**2 - BALL_RADIUS
        
        return np.exp(-0.5 * distance / (CAR_MAX_SPEED))

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class DistanceBallGoalReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        net_position = ORANGE_GOAL_CENTER

        c = net_position[1] - BACK_WALL_Y + BALL_RADIUS
        
        distance = np.linalg.norm(ball_position - net_position, 2)**2 - c
        
        return np.exp(-0.5 * distance / (CAR_MAX_SPEED))

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class FacingBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_orientation = player.car_data.forward()
        
        ball_position = state.ball.position
        car_position = player.car_data.position
        
        direction_to_ball = ball_position - car_position
        direction_to_ball /= np.linalg.norm(direction_to_ball, 2)**2
        
        reward = np.dot(car_orientation, direction_to_ball)
        
        return reward

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class AlignBallGoalReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_position = player.car_data.position
        ball_position = state.ball.position
        opponent_goal_position = ORANGE_GOAL_CENTER
        
        direction_to_ball = ball_position - car_position
        direction_to_goal = opponent_goal_position - car_position
        
        direction_to_ball /= np.linalg.norm(direction_to_ball)
        direction_to_goal /= np.linalg.norm(direction_to_goal)
        
        return 0.5 * np.dot(direction_to_ball, direction_to_goal) + 0.5 * np.dot(direction_to_goal, direction_to_ball)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class ClosestToBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        car_position = player.car_data.position
        distance = np.linalg.norm(ball_position - car_position)
        
        for p in state.players:
            if distance > np.linalg.norm(ball_position - p.car_data.position):
                return 0
        
        return 1

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class TouchedLastReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        return 1 if state.last_touch == player.car_id else 0

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class BehindBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_position = player.car_data.position
        ball_position = state.ball.position
        net_position = ORANGE_GOAL_CENTER
        
        direction_to_ball = ball_position - car_position
        direction_to_net = net_position - car_position
        
        return 1 if np.dot(direction_to_ball, direction_to_net) > 0 else 0

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class VelocityPlayerBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_velocity = player.car_data.linear_velocity
        
        ball_position = state.ball.position
        car_position = player.car_data.position
        
        direction_to_ball = ball_position - car_position
        direction_to_ball /= np.linalg.norm(direction_to_ball)
        
        car_velocity_direction = np.dot(car_velocity, direction_to_ball)
        
        return car_velocity_direction / np.linalg.norm(car_velocity, 2)**2

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class VelocityReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_velocity = player.car_data.linear_velocity

        return np.linalg.norm(car_velocity, 2) / CAR_MAX_SPEED

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class BoostAmountReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        return np.sqrt(player.boost_amount / 100)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

class ForwardVelocityReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_velocity = player.car_data.linear_velocity
        car_orientation = player.car_data.forward()
        
        car_forward_velocity = np.dot(car_velocity, car_orientation)

        return car_forward_velocity * np.linalg.norm(car_velocity) / CAR_MAX_SPEED

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

