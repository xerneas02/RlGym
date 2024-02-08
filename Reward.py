import numpy as np
import rlgym
from stable_baselines3 import PPO
from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions import common_conditions
from rlgym.utils.gamestates import PlayerData, GameState
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.common_values import BALL_RADIUS, CAR_MAX_SPEED, BALL_MAX_SPEED, ORANGE_GOAL_CENTER, BACK_WALL_Y, BLUE_GOAL_CENTER
from numpy.linalg import norm
from abc import ABC, abstractmethod
import numpy as np
import numpy as np
from scipy.spatial.distance import cosine


from typing import Any, Optional, Tuple, overload, Union

import numpy as np
from rlgym.utils.reward_functions import RewardFunction
from rlgym.utils.gamestates import GameState, PlayerData

from RlGym import GAME_SPEED


class CombinedReward(RewardFunction):

    def __init__(
            self,
            reward_functions: Tuple[RewardFunction, ...],
            reward_weights: Optional[Tuple[float, ...]] = None
    ):
        super().__init__()

        self.count = 0
        self.reward_functions = reward_functions
        self.reward_weights = reward_weights or np.ones_like(reward_functions)

        if len(self.reward_functions) != len(self.reward_weights):
            raise ValueError(
                ("Reward functions list length ({0}) and reward weights " \
                 "length ({1}) must be equal").format(
                    len(self.reward_functions), len(self.reward_weights)
                )
            )

    @classmethod
    def from_zipped(cls, *rewards_and_weights: Union[RewardFunction, Tuple[RewardFunction, float]]) -> "CombinedReward":
        """
        Alternate constructor which takes any number of either rewards, or (reward, weight) tuples.

        :param rewards_and_weights: a sequence of RewardFunction or (RewardFunction, weight) tuples
        """
        rewards = []
        weights = []
        for value in rewards_and_weights:
            if isinstance(value, tuple):
                r, w = value
            else:
                r, w = value, 1.
            rewards.append(r)
            weights.append(w)
        return cls(tuple(rewards), tuple(weights))

    def reset(self, initial_state: GameState) -> None:
        self.count = 0
        for func in self.reward_functions:
            func.reset(initial_state)

    def get_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ) -> float:
        rewards = [
            func.get_reward(player, state, previous_action)
            for func in self.reward_functions
        ]
        
        total = float(np.dot(self.reward_weights, rewards))
        self.count += 1

        if GAME_SPEED == 1:
            for i in range(len(rewards)):
                if rewards[i] != 0 and player.team_num == 0:
                    print(f"reward {str(self.reward_functions[i]).split('.')[1].split(' ')[0]}: {rewards[i]*self.reward_weights[i]}")
            
            
            if total != 0 and player.team_num == 0:
                print(f"Total : {total}\n-----------------------")

        return total

    def get_final_reward(
            self,
            player: PlayerData,
            state: GameState,
            previous_action: np.ndarray
    ) -> float:
        if GAME_SPEED == 1 and player.team_num == 0:
            print(f"---  Time = {self.count}  ---")
        
        rewards = [
            func.get_final_reward(player, state, previous_action)
            for func in self.reward_functions
        ]

        return float(np.dot(self.reward_weights, rewards))



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

#Si le bot marque un but
class GoalScoredReward(RewardFunction):
    def __init__(self):
        self.previous_blue_score   = 0
        self.previous_orange_score = 0
        
    def reset(self, initial_state):
        self.previous_blue_score   = initial_state.blue_score
        self.previous_orange_score = initial_state.orange_score

    def get_reward(self, player, state, previous_action):
        blue_scored   = False
        orange_scored = False
        
        if self.previous_blue_score != state.blue_score:
            blue_scored = True
            self.previous_blue_score = state.blue_score
        
        if self.previous_orange_score != state.orange_score:
            orange_scored = True
            self.previous_orange_score = state.orange_score
            
        if(player.team_num == 0 and not blue_scored  ) : return 0 
        if(player.team_num == 1 and not orange_scored) : return 0
        
        
        ball_speed = np.linalg.norm(state.ball.linear_velocity, 2)**2
        return 1.0 + 0.5 * ball_speed / (BALL_MAX_SPEED)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot collect ou utilise du boost
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
    
#Si le bot touche la balle (reward varie en fonction de la hauteur de la balle)
class BallTouchReward(RewardFunction):
    def __init__(self):
        self.last_touch = False
        self.lamb = 0

    def reset(self, initial_state):
        self.last_touch = False
        self.lamb = 0

    def get_reward(self, player, state, previous_action):
        
        if not player.ball_touched : 
            self.last_touch = False
            return 0
        
        if self.last_touch:
            self.lamb = max(0.1, self.lamb * 0.95)
        else:
            self.lamb = min(1.0, self.lamb + 0.013)
            
        self.last_touch = True
            
        pos_ball_z = state.ball.position[2]
        reward = self.lamb * ((pos_ball_z + BALL_RADIUS)/(2*BALL_RADIUS)) ** 0.2836
        
        return reward

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
#Si le bot démo
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

#Si le bot est proche de la balle
class DistancePlayerBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_position = player.car_data.position
        ball_position = state.ball.position
        
        distance = np.linalg.norm(car_position - ball_position, 2) - BALL_RADIUS
        
        return np.exp(-0.5 * distance / (CAR_MAX_SPEED))

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si la balle est proche du but adverse
class DistanceBallGoalReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        net_position = ORANGE_GOAL_CENTER

        c = net_position[1] - BACK_WALL_Y + BALL_RADIUS
        
        distance = np.linalg.norm(ball_position - net_position, 2) - c
        
        return np.exp(-0.5 * distance / (CAR_MAX_SPEED))

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
#Si le bot fait face à la balle
class FacingBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_orientation = player.car_data.forward()
        
        ball_position = state.ball.position
        car_position = player.car_data.position
        
        direction_to_ball = ball_position - car_position
        direction_to_ball /= np.linalg.norm(direction_to_ball, 2)
        
        reward = np.dot(car_orientation, direction_to_ball)
        
        return reward

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot est entre ses buts et la balle [mais il y a une ligne qui relie le bot, la balle et le but]
class AlignBallGoalReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_position  = player.car_data.position
        ball_position = state.ball.position
        
        opponent_goal_position = ORANGE_GOAL_CENTER
        self_goal_position     = BLUE_GOAL_CENTER
        
        direction_car_to_ball         = ball_position          - car_position
        direction_ball_to_car         = car_position           - ball_position
        direction_self_goal_to_car    = car_position           - self_goal_position
        direction_car_to_oponent_goal = opponent_goal_position - car_position
        
        direction_car_to_ball        /= np.linalg.norm(direction_car_to_ball)
        direction_ball_to_car        /= np.linalg.norm(direction_ball_to_car)
        direction_self_goal_to_car   /= np.linalg.norm(direction_self_goal_to_car)
        direction_car_to_oponent_goal/= np.linalg.norm(direction_car_to_oponent_goal)
        
        
        return -0.5 * np.dot(direction_ball_to_car, direction_car_to_oponent_goal) + 0.5 * np.dot(direction_car_to_ball, direction_self_goal_to_car) 

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si plus proche de la balle par rapport aux adversaires
class ClosestToBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        car_position = player.car_data.position
        distance = np.linalg.norm(ball_position - car_position, 2)
        
        for p in state.players:
            if distance > np.linalg.norm(ball_position - p.car_data.position, 2):
                return 0
        
        return 1

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot est le dernier à avoir touché la balle
class TouchedLastReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        return 1 if state.last_touch == player.car_id else 0

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot est entre la balle et son but
class BehindBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_position  = player.car_data.position
        ball_position = state.ball.position
        net_position  = ORANGE_GOAL_CENTER
        
        direction_to_ball = ball_position - car_position
        direction_to_net  = net_position - car_position
        
        return 1 if np.dot(direction_to_ball, direction_to_net) > 0 else 0

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot va dans la même direction de la balle
class VelocityPlayerBallReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_velocity = player.car_data.linear_velocity
        
        if np.linalg.norm(car_velocity, 2) < 0.01 :
            return 0
        
        ball_position = state.ball.position
        car_position = player.car_data.position
        
        direction_to_ball = ball_position - car_position
        direction_to_ball /= np.linalg.norm(direction_to_ball)
        
        car_velocity_direction = np.dot(car_velocity, direction_to_ball)
        
        return car_velocity_direction / np.linalg.norm(car_velocity, 2)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot gagne le kickoff
class KickoffReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        
        if ball_position[0] != 0 or ball_position[1] != 0:
            return 0
        
        car_velocity = player.car_data.linear_velocity
        
        if np.linalg.norm(car_velocity, 2) < 0.01 :
            return 0
        
        car_position = player.car_data.position
        
        direction_to_ball = ball_position - car_position
        direction_to_ball /= np.linalg.norm(direction_to_ball)
        
        car_velocity_direction = np.dot(car_velocity, direction_to_ball)
        
        return car_velocity_direction / np.linalg.norm(car_velocity, 2)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot bouge
class VelocityReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        car_velocity = player.car_data.linear_velocity

        return np.linalg.norm(car_velocity, 2) / CAR_MAX_SPEED

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot à du boost
class BoostAmountReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        return np.sqrt(player.boost_amount / 100)

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)

#Si le bot bouge dans la direction de la balle (dans la bonne direction), penalise la marche arrière
class ForwardVelocityReward(RewardFunction):
    def reset(self, initial_state):
        pass

    def get_reward(self, player, state, previous_action):
        
        car_velocity = player.car_data.linear_velocity
        car_orientation = player.car_data.forward()
        
        car_forward_velocity = np.dot(car_velocity, car_orientation)

        return car_forward_velocity / CAR_MAX_SPEED

    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
    
class FirstTouchReward(RewardFunction):
    def __init__(self):
        self.kickoff = True
        
    def reset(self, initial_state):
        self.kickoff = True

    def get_reward(self, player, state, previous_action):
        ball_position = state.ball.position
        if ball_position[0] != 0 or ball_position[1] != 0 and self.kickoff:
            self.kickoff = False
            return player.car_id == state.last_touch
        
        return 0

        
        
    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
    
class AirPenalityReward(RewardFunction):
    def reset(self, initial_state):
        pass
    
    def get_reward(self, player, state, previous_action):
        return (not player.on_ground)*-1
    
    def get_final_reward(self, player, state, previous_action):
        return self.get_reward(player, state, previous_action)
    
class DontTouchPenalityReward(RewardFunction):
    def __init__(self):
        super().__init__()
        self.ticks = 0

    def reset(self, initial_state: GameState):
        self.ticks = 0

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:    
        self.ticks += 1
        
        if(player.ball_touched):
            self.tick = 0
    
        return - (self.ticks * 0.001)

    def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        return self.get_reward(player, state, previous_action)

