from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, CAR_MAX_ANG_VEL, BALL_MAX_SPEED
from rlgym.utils.math import rand_vec3
from Reward import CombinedReward

import numpy as np
import random
import math

import re
import os

from Constante import *

from typing import Any, Optional, Tuple, overload, Union


            
            
def movement_ball(ball, z_axe = False):
    ball_x_velo = random.randint(-BALL_SPEED, BALL_SPEED)
    ball_y_velo = random.randint(-BALL_SPEED, BALL_SPEED)
    
    ball_z_velo = 0
    if(z_axe):
        ball_z_velo = random.randint(0, BALL_SPEED*2)
    
    ball.set_lin_vel(ball_x_velo, ball_y_velo, ball_z_velo)
            
            
class CombinedState(StateSetter):
    current_state = 0 #Attribut de classe initialisé à 0
    def __init__(
            self,
            rewards: CombinedReward,
            state_setters: Tuple[Tuple[StateSetter, Tuple[float, ...]], ...],
            state_probas : Optional[Tuple[float, ...]] = None
    ):
        super().__init__()
        self.rewards = rewards
        self.state_setters = state_setters
        self.state_probas = state_probas or np.ones_like(state_setters)
        
        
        

        if len(self.state_setters) != len(self.state_probas):
            raise ValueError(
                ("Terminal conditions list length ({0}) and condition probabilities " \
                 "length ({1}) must be equal").format(
                    len(self.state_setters), len(self.state_probas)
                )
            )
        
        sums = math.fsum(state_probas)
        if sums < 0.98 or sums > 1.2:
            raise ValueError(
                (
                    "Probas don't add up to 1"
                )
            )

    def reset(self, state_wrapper: StateWrapper) -> None:
        r = random.random()
        sum = 0
        
        for i in range(len(self.state_setters)):
            sum += self.state_probas[i]
            
            default_rewards_weights = self.rewards.get_default_reward_weights()
            size_rewards = self.rewards.get_rewards_num()
            
            if r < sum:
                #print(self.state_setters[i][0])
                CombinedState.current_state = i
                self.state_setters[i][0].reset(state_wrapper)
                if self.state_setters[i][1] == None or len(self.state_setters[i][1]) != size_rewards:
                    self.rewards.set_rewards_weights(default_rewards_weights)
                else:
                    self.rewards.set_rewards_weights(
                        [
                            self.state_setters[i][1][j] if self.state_setters[i][1][j] != 42 else default_rewards_weights[j]
                            for j in range(size_rewards)
                        ]
                    )    
                return 
        
        self.state_setters[0].reset(state_wrapper)
        self.rewards.set_rewards_weights(default_rewards_weights)

    @staticmethod
    def get_current_state() -> int :
        return CombinedState.current_state
               


class BetterRandom(StateSetter):  # Random state with some triangular distributions
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(
            x=np.random.uniform(-LIM_X, LIM_X),
            y=np.random.uniform(-LIM_Y, LIM_Y),
            z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
        )

        # 99.9% chance of below ball max speed
        ball_speed = np.random.exponential(-BALL_MAX_SPEED / np.log(1 - 0.999))
        vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED))
        state_wrapper.ball.set_lin_vel(*vel)

        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
        state_wrapper.ball.set_ang_vel(*ang_vel)

        for car in state_wrapper.cars:
            # On average 1 second at max speed away from ball
            ball_dist = np.random.exponential(BALL_MAX_SPEED)
            ball_car = rand_vec3(ball_dist)
            car_pos = state_wrapper.ball.position + ball_car
            if abs(car_pos[0]) < LIM_X \
                    and abs(car_pos[1]) < LIM_Y \
                    and 0 < car_pos[2] < LIM_Z:
                car.set_pos(*car_pos)
            else:  # Fallback on fully random
                car.set_pos(
                    x=np.random.uniform(-LIM_X, LIM_X),
                    y=np.random.uniform(-LIM_Y, LIM_Y),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
                )

            vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
            car.set_lin_vel(*vel)

            car.set_rot(
                pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
            )

            ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
            car.set_ang_vel(*ang_vel)
            car.boost = np.random.uniform(0, 1)
        



class TrainingStateSetter(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Randomly chooses 
        """
        
        car_x = random.randint(-500, 500)
        CAR_Z = 0
        yaw = 0
        state_num = random.randint(0,12)

        # Attack
        if state_num % 3 == 0:
            for car in state_wrapper.cars:
                # team_num = 0 = blue team
                if car.team_num == 0:
                    # select a unique spawn state from pre-determined values
                    CAR_Y = 0
                    yaw = 0.5 * np.pi
                    car.set_pos(car_x, CAR_Y, CAR_Z)

                # team_num = 1 = orange team
                elif car.team_num == 1:
                    # select a unique spawn state from pre-determined values
                    CAR_Y = 4260
                    yaw = -0.5 * np.pi
                    car.set_pos(0, CAR_Y, CAR_Z)


                # set car state values
                
                car.set_rot(yaw=yaw)
                car.boost = 0.25
            
            state_wrapper.ball.set_pos(x=car_x, y=2816.0, z=70.0) # random x value so agent doesn't become acclimated to one ball pos
            state_wrapper.ball.set_lin_vel(0, 0, 0)
        
        # Defend
        elif state_num % 3 == 1:
            for car in state_wrapper.cars:
                # team_num = 0 = blue team
                if car.team_num == 0:
                    # select a unique spawn state from pre-determined valuesl
                    car_y2 = -5120
                    yaw = 0.5 * np.pi
                    car.set_pos(0, car_y2, CAR_Z)

                # team_num = 1 = orange team
                elif car.team_num == 1:
                    # select a unique spawn state from pre-determined values
                    CAR_Y = -2500
                    yaw = -0.5 * np.pi
                    car.set_pos(0, CAR_Y, CAR_Z)


                # set car state values
                
                car.set_rot(yaw=yaw)
                car.boost = 0.5

            ball_x_velo = random.randint(-200, 200)
            ball_y_velo = random.randint(100,1500)
            state_wrapper.ball.set_pos(0.0, -2816.0, 70.0)
            state_wrapper.ball.set_lin_vel(ball_x_velo, ball_y_velo, 0)
        
        # Center spawn on top of agent's car
        elif state_num % 3 == 2:
            for car in state_wrapper.cars:
                # team_num = 0 = blue team
                if car.team_num == 0:
                    # select a unique spawn state from pre-determined valuesl
                    yaw = 0.5 * np.pi
                    car.set_pos(0.0, -1024.0, 30.0)

                # team_num = 1 = orange team
                elif car.team_num == 1:
                    # select a unique spawn state from pre-determined values
                    CAR_Y = -2500
                    yaw = -0.5 * np.pi
                    car.set_pos(0.0,  1024.0, 30.0)


                # set car state values
                
                car.set_rot(yaw=yaw)
                car.boost = 0.5

            state_wrapper.ball.set_pos(0.0, -960.0, 70.0)
            state_wrapper.ball.set_lin_vel(0, 0, 0)
            
            


class DefaultStateClose(StateSetter):


    def __init__(self, number_of_state = 2):
        super().__init__()
        self.number_of_state = number_of_state

    def reset(self, state_wrapper: StateWrapper):
        
        coef = 1/random.randint(1, self.number_of_state)
        
        SPAWN_BLUE_POS = [[-2048*coef, -2560*coef, 17], [2048*coef, -2560*coef, 17],
                      [-256*coef, -3840*coef, 17], [256*coef, -3840*coef, 17], [0, -4608*coef, 17]]
        SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                        0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        SPAWN_ORANGE_POS = [[2048*coef, 2560*coef, 17], [-2048*coef, 2560*coef, 17],
                            [256*coef, 3840*coef, 17], [-256*coef, 3840*coef, 17], [0, 4608*coef, 17]]
        SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                            np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4]
        random.shuffle(spawn_inds)

        blue_count = 0
        orange_count = 0
        
        if MOVE_BALL : movement_ball(state_wrapper.ball)
        
        for car in state_wrapper.cars:
            pos = [0,0,0]
            yaw = 0
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33
            

class RandomState(StateSetter):
    

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
              
        SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
        SPAWN_BLUE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                        0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                            [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
        SPAWN_ORANGE_YAW = [-0.75 * np.pi, -0.25 *
                            np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4]
        random.shuffle(spawn_inds)

        blue_count = 0
        orange_count = 0
        
        wall_x = SIDE_WALL_X - BALL_RADIUS*2
        wall_y = BACK_WALL_Y - BALL_RADIUS*2
        
        ball_x = random.randint(-int(wall_x), int(wall_x))
        ball_y = random.randint(-int(wall_y), int(wall_y))
        
        if MOVE_BALL : movement_ball(state_wrapper.ball)
        
        min_distance = 200

        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        ball_x = random.randint(-int(wall_x), int(wall_x))
        ball_y = random.randint(-int(wall_y), int(wall_y))

        while any(distance([ball_x, ball_y, 0], car_pos) < min_distance for car_pos in SPAWN_BLUE_POS + SPAWN_ORANGE_POS):
            ball_x = random.randint(-int(wall_x), int(wall_x))
            ball_y = random.randint(-int(wall_y), int(wall_y))

        #print("Position de la balle (x, y):", ball_x, ball_y)        
        
        state_wrapper.ball.set_pos(ball_x, ball_y, BALL_RADIUS)
        
        for car in state_wrapper.cars:
            pos = [0,0,0]
            yaw = 0
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33
            
            
class InvertedState(StateSetter):
    

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
              
        SPAWN_BLUE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
        SPAWN_BLUE_YAW = [-0.25 * np.pi, -0.75 * np.pi,
                        -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]
        SPAWN_ORANGE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                            [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
        SPAWN_ORANGE_YAW = [0.75 * np.pi, 0.25 *
                            np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4]
        random.shuffle(spawn_inds)

        blue_count = 0
        orange_count = 0
        
        if MOVE_BALL : movement_ball(state_wrapper.ball)
        
        for car in state_wrapper.cars:
            pos = [0,0,0]
            yaw = 0
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33
            
            
class DefaultStateCloseOrange(StateSetter):
    

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected default kickoff.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        
        coef = 1/random.randint(1, 2)
        
        SPAWN_ORANGE_POS = [[-2048*coef, -2560*coef, 17], [2048*coef, -2560*coef, 17],
                      [-256*coef, -3840*coef, 17], [256*coef, -3840*coef, 17], [0, -4608*coef, 17]]
        SPAWN_ORANGE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                        0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        SPAWN_BLUE_POS = [[2048*coef, 2560*coef, 17], [-2048*coef, 2560*coef, 17],
                            [256*coef, 3840*coef, 17], [-256*coef, 3840*coef, 17], [0, 4608*coef, 17]]
        SPAWN_BLUE_YAW = [-0.75 * np.pi, -0.25 *
                            np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4]
        random.shuffle(spawn_inds)

        blue_count = 0
        orange_count = 0
        
        if MOVE_BALL : movement_ball(state_wrapper.ball)
        
        for car in state_wrapper.cars:
            pos = [0,0,0]
            yaw = 0
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33
            

class RandomStateOrange(StateSetter):
    

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
              
        SPAWN_ORANGE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
        SPAWN_ORANGE_YAW = [0.25 * np.pi, 0.75 * np.pi,
                        0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        SPAWN_BLUE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                            [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
        SPAWN_BLUE_YAW = [-0.75 * np.pi, -0.25 *
                            np.pi, -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4]
        random.shuffle(spawn_inds)

        blue_count = 0
        orange_count = 0
        
        wall_x = SIDE_WALL_X - BALL_RADIUS*2
        wall_y = BACK_WALL_Y - BALL_RADIUS*2
        
        ball_x = random.randint(-int(wall_x), int(wall_x))
        ball_y = random.randint(-int(wall_y), int(wall_y))
        
        min_distance = 200
        
        if MOVE_BALL : movement_ball(state_wrapper.ball)

        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        ball_x = random.randint(-int(wall_x), int(wall_x))
        ball_y = random.randint(-int(wall_y), int(wall_y))

        while any(distance([ball_x, ball_y, 0], car_pos) < min_distance for car_pos in SPAWN_BLUE_POS + SPAWN_ORANGE_POS):
            ball_x = random.randint(-int(wall_x), int(wall_x))
            ball_y = random.randint(-int(wall_y), int(wall_y))

        #print("Position de la balle (x, y):", ball_x, ball_y)        
        
        state_wrapper.ball.set_pos(ball_x, ball_y, BALL_RADIUS)
        
        for car in state_wrapper.cars:
            pos = [0,0,0]
            yaw = 0
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33
            
            
class InvertedStateOrange(StateSetter):
    

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
              
        SPAWN_ORANGE_POS = [[-2048, -2560, 17], [2048, -2560, 17],
                      [-256, -3840, 17], [256, -3840, 17], [0, -4608, 17]]
        SPAWN_ORANGE_YAW = [-0.25 * np.pi, -0.75 * np.pi,
                        -0.5 * np.pi, -0.5 * np.pi, -0.5 * np.pi]
        SPAWN_BLUE_POS = [[2048, 2560, 17], [-2048, 2560, 17],
                            [256, 3840, 17], [-256, 3840, 17], [0, 4608, 17]]
        SPAWN_BLUE_YAW = [0.75 * np.pi, 0.25 *
                            np.pi, 0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi]
        # possible kickoff indices are shuffled
        spawn_inds = [0, 1, 2, 3, 4]
        random.shuffle(spawn_inds)

        blue_count = 0
        orange_count = 0
        
        if MOVE_BALL : movement_ball(state_wrapper.ball)
        
        for car in state_wrapper.cars:
            pos = [0,0,0]
            yaw = 0
            # team_num = 0 = blue team
            if car.team_num == 0:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_BLUE_POS[spawn_inds[blue_count]]
                yaw = SPAWN_BLUE_YAW[spawn_inds[blue_count]]
                blue_count += 1
            # team_num = 1 = orange team
            elif car.team_num == 1:
                # select a unique spawn state from pre-determined values
                pos = SPAWN_ORANGE_POS[spawn_inds[orange_count]]
                yaw = SPAWN_ORANGE_YAW[spawn_inds[orange_count]]
                orange_count += 1
            # set car state values
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.33            
        
class LineState(StateSetter):
    
    def __init__(self, largeur):
        super().__init__()
        self.largeur = largeur

    def reset(self, state_wrapper: StateWrapper):
        
        count = 0
        wall_x = SIDE_WALL_X - BALL_RADIUS
        #wall_y = BACK_WALL_Y - BALL_RADIUS
        ceiling = CEILING_Z - BALL_RADIUS
        #----SPAWN BOUBOULE--------------------------------
        ball_x = random.randint(-int(wall_x), int(wall_x))
        #ball_y = random.randint(-int(wall_y), int(wall_y))
        ball_y = 0 + random.randint(-int(self.largeur), int(self.largeur))
        ball_z = random.randint(int(BALL_RADIUS)+1, int(ceiling/2))
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
        if MOVE_BALL : movement_ball(state_wrapper.ball)
        #---------------------------------------------------
        #----SPAWN CARS-------------------------------------
        for car in state_wrapper.cars:
            car_x = random.randint(-int(wall_x), int(wall_x))
            #ball_y = random.randint(-int(wall_y), int(wall_y))
            car_y = 2300 + random.randint(-int(self.largeur), int(self.largeur)) if count == 1  else -2300 + random.randint(-int(self.largeur), int(self.largeur))
            yaw = -0.5 * np.pi if count == 1 else 0.5 * np.pi
            car_z = 17
            car.set_pos(car_x, car_y, car_z)
            car.boost = 0.33
            car.set_rot(yaw=yaw)
            count = count + 1
        #---------------------------------------------------    
            
            
class Alea(StateSetter):
    
    def __init__(self, ball_moove, z_axis):
        super().__init__()
        self.ball_moove = ball_moove
        self.z_axis = z_axis

    def reset(self, state_wrapper: StateWrapper):
        
        #----SPAWN BOUBOULE--------------------------------
        ball_x = random.randint(int(-3200), int(3200))
        ball_y = random.randint(int(-4250), int(4250))
        ball_z = random.randint(int(BALL_RADIUS)+1, int(LIM_Z / 2))
        state_wrapper.ball.set_pos(ball_x, ball_y, 17)
        x_velo = 0
        y_velo = 0
        z_velo = 0
        if self.ball_moove:
            x_velo = random.randint(-2000, 2000)
            y_velo = random.randint(-2000, 2000)
            if self.z_axis:
                z_velo = random.randint(0, 2000)
        state_wrapper.ball.set_lin_vel(x_velo, y_velo, z_velo)
        #---------------------------------------------------
        #----SPAWN CARS-------------------------------------
        for car in state_wrapper.cars:
            car_x = random.randint(int(-3200), int(3200))
            car_y = random.randint(int(-4250), int(4250))
            yaw = random.random() * 2 * np.pi
            car_z = 17
            car.set_pos(car_x, car_y, car_z)
            car.boost = random.random()
            car.set_rot(yaw=yaw)
        #---------------------------------------------------
        
class Attaque(StateSetter):
    
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        
        count = 0
        inverse = random.choice([-1, 1])
        inverse2 = random.choice([-1, 1])
        angle = 0 if inverse == 1 else np.pi
        #----SPAWN BOUBOULE--------------------------------
        ball_x = 2000 * inverse
        ball_y = 4000 * inverse2
        ball_z = int(BALL_RADIUS)+1
        ball_x_velo = -1000 * inverse
        ball_y_velo = 0
        ball_z_velo = 0
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
        state_wrapper.ball.set_lin_vel(ball_x_velo, ball_y_velo, ball_z_velo)
        #---------------------------------------------------
        #----SPAWN CARS-------------------------------------
        for car in state_wrapper.cars:
            car_x = -3500 * inverse if count == 1 else 3200 * inverse
            car_y = 4100 * inverse2 if count == 1 else 3000 * inverse2
            yaw = ((np.pi)/6) + angle if count == 1 else ((np.pi) + angle)
            #print(f"--->{count} {(-np.pi * inverse)}")
            car_z = 17
            car.set_pos(car_x, car_y, car_z)
            car.boost = 0.33
            car.set_rot(yaw=yaw)
            count = count + 1
        #---------------------------------------------------
        
class Defense(StateSetter):
    
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        count = 0
        inverse = random.choice([-1, 1])
        inverse2 = random.choice([-1, 1])
        angle = 0 if inverse2 == 1 else np.pi
        position1x = 851 if inverse2 == 1 else -278
        position1y = -4346 if inverse2 == 1 else -241
        position2x = -278 if inverse2 == 1 else 851
        position2y = -241 if inverse2 == 1 else -4346
        #----SPAWN BOUBOULE--------------------------------
        ball_x = -284 * inverse
        ball_y = -1115 * inverse2
        ball_z = int(BALL_RADIUS)+1
        ball_x_velo = 10
        ball_y_velo = -500 * inverse2
        ball_z_velo = 0
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
        state_wrapper.ball.set_lin_vel(ball_x_velo, ball_y_velo, ball_z_velo)
        #---------------------------------------------------
        #----SPAWN CARS-------------------------------------
        for car in state_wrapper.cars:
            car_x = position2x * inverse if count == 1 else position1x * inverse
            car_y = position2y * inverse2 if count == 1 else position1y* inverse2
            if count == 1 and inverse == 1:
                yaw = (3*np.pi)/2 if inverse2 == 1 else (5*np.pi)/4
            elif count == 1 and inverse == -1: 
                yaw = (3*np.pi)/2 if inverse2 == 1 else (7*np.pi)/4
            elif count == 0 and inverse == 1:
                yaw = (3*np.pi)/4 if inverse2 == 1 else (3*np.pi)/2 + angle
            elif count == 0 and inverse == -1:
                yaw = (np.pi)/4 if inverse2 == 1 else (3*np.pi)/2 + angle
            #print(f"--->{count} {(-np.pi * inverse)}")
            car_z = 17
            car.set_pos(car_x, car_y, car_z)
            car.boost = 0.40
            car.set_rot(yaw=yaw)
            if count == 1 and inverse2 == 1:
                car.set_lin_vel(ball_x_velo, ball_y_velo, ball_z_velo)
            elif count == 0 and inverse2 == -1:
                car.set_lin_vel(ball_x_velo, ball_y_velo, ball_z_velo)
            
            count = count + 1
        #---------------------------------------------------
        
        
class AirBallAD(StateSetter):
    
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        count = 0
        inverse = random.choice([-1, 1])
        position1x = -5.71 if inverse == 1 else 7.13
        position1y = -4676 if inverse == 1 else 3718
        position2x = -7.13 if inverse == 1 else 5.71
        position2y = -3718 if inverse == 1 else 4676
        
        #----SPAWN BOUBOULE--------------------------------
        ball_x = -7.13 * inverse
        ball_y = -4211 * inverse
        ball_z = int(BALL_RADIUS)+1
        ball_x_velo = 0
        ball_y_velo = 0
        ball_z_velo = random.randint(500, 1000)
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
        state_wrapper.ball.set_lin_vel(ball_x_velo, ball_y_velo, ball_z_velo)
        #---------------------------------------------------
        #----SPAWN CARS-------------------------------------
        for car in state_wrapper.cars:
            car_x = position2x  if count == 1 else position1x 
            car_y = position2y  if count == 1 else position1y
            yaw = (3*np.pi) / 2  if count == 1 else np.pi/2 
            car_z = 17
            car.set_pos(car_x, car_y, car_z)
            car.boost = 1.0
            car.set_rot(yaw=yaw)
            count = count + 1
        #---------------------------------------------------
        
class DefenseRapide(StateSetter):
    
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        count = 0
        inverse = random.choice([-1, 1])
        inverse2 = random.choice([-1, 1])
        position1x = -848 * inverse2 if inverse == 1 else 0
        position1y = 36 if inverse == 1 else -670
        position2x = 0 if inverse == 1 else -848 * inverse2
        position2y = 670 if inverse == 1 else -36
        angle = np.pi if inverse == -1 else 0
        #----SPAWN BOUBOULE--------------------------------
        ball_x = 0 * inverse
        ball_y = -483 * inverse
        ball_z = int(BALL_RADIUS)+1
        ball_x_velo = 0
        ball_y_velo = random.randint(int(800), int(1050))
        ball_y_velo = ball_y_velo * inverse * -1
        ball_z_velo = 0
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
        state_wrapper.ball.set_lin_vel(ball_x_velo, ball_y_velo, ball_z_velo)
        #---------------------------------------------------
        #----SPAWN CARS-------------------------------------
        for car in state_wrapper.cars:
            car_x = position2x  if count == 1 else position1x 
            car_y = position2y  if count == 1 else position1y
            yaw = (3*np.pi) / 2 + angle if count == 1 else (3*np.pi) / 2 + angle
            car_z = 17
            car.set_pos(car_x, car_y, car_z)
            car.boost = 1.0
            car.set_rot(yaw=yaw)
            car.set_lin_vel(ball_x_velo, ball_y_velo, ball_z_velo)
            count = count + 1
        #---------------------------------------------------
        
class Mur(StateSetter):
    
    def __init__(self, angles):
        super().__init__()
        self.angles = angles

    def reset(self, state_wrapper: StateWrapper):
        count = 0
        inverse = random.choice([-1, 1])
        position1x = -4078.99 * inverse
        position1y = -1990
        position2x = -1360 * inverse
        position2y = 3477 if inverse == 1 else 3477
        angle = (5 * np.pi) / 3 if inverse == 1 else (4 * np.pi) / 3
        #----SPAWN BOUBOULE--------------------------------
        ball_x = -3590 * inverse
        ball_y = 0
        ball_z = int(BALL_RADIUS)+1
        ball_x_velo = random.randint(int(1800), int(3500)) * -1 * inverse
        ball_y_velo = random.randint(int(-self.angles), int(self.angles))
        ball_z_velo = 0
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
        state_wrapper.ball.set_lin_vel(ball_x_velo, ball_y_velo, ball_z_velo)
        #---------------------------------------------------
        #----SPAWN CARS-------------------------------------
        for car in state_wrapper.cars:
            car_x = position2x  if count == 1 else position1x 
            car_y = position2y  if count == 1 else position1y
            yaw = angle if count == 1 else np.pi / 2
            roll = 0 if count == 1 else 1884 * inverse
            car_z = 650 if count == 0 else 17.1
            car.set_pos(car_x, car_y, car_z)
            car.boost = 0.70
            car.set_rot(0, yaw=yaw, roll=roll)
            if count == 0:
                car.set_lin_vel(0, 300, 0)
            count = count + 1
        #---------------------------------------------------
        
class OpenGoal(StateSetter):
    
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        count = 0
        position1x = random.randint(-2500, 2500);
        position1y = random.randint(-2000, 1000);
        position2x = random.randint(-3500, 3500);
        position2y = random.randint(-4000, 4000);
        angle = np.pi / 2
        angle2 = random.random() * np.pi * 2
        #----SPAWN BOUBOULE--------------------------------
        ball_x = random.randint(-2000, 2000);
        ball_y = random.randint(2000, 4000);
        ball_z = int(BALL_RADIUS)+1
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
        #---------------------------------------------------
        #----SPAWN CARS-------------------------------------
        for car in state_wrapper.cars:
            car_x = position2x  if count == 1 else position1x 
            car_y = position2y  if count == 1 else position1y
            yaw = angle2 if count == 1 else angle
            car_z = 17.1
            car.set_pos(car_x, car_y, car_z)
            car.boost = random.random()
            car.set_rot(0, yaw=yaw)
            count = count + 1
        #---------------------------------------------------

class ChaosState(StateSetter):
    

    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):        
        min_distance = 400
        
        wall_x = SIDE_WALL_X - min_distance
        wall_y = BACK_WALL_Y - min_distance
        
        ball_x = random.randint(-int(wall_x), int(wall_x))
        ball_y = random.randint(-int(wall_y), int(wall_y))
        
        movement_ball(state_wrapper.ball)
        
        

        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        ball_x = random.randint(-int(wall_x), int(wall_x))
        ball_y = random.randint(-int(wall_y), int(wall_y))     
        
        state_wrapper.ball.set_pos(ball_x, ball_y, BALL_RADIUS)
        objPos = [state_wrapper.ball.position]
        
        for car in state_wrapper.cars:
            car_x = random.randint(-int(wall_x), int(wall_x))
            car_y = random.randint(-int(wall_y), int(wall_y))
            
            while any(distance([car_x, car_y, 0], obj_pos) < min_distance for obj_pos in objPos):
                car_x = random.randint(-int(wall_x), int(wall_x))
                car_y = random.randint(-int(wall_y), int(wall_y))
                
            
            velocity = np.random.uniform(-1, 1, 2)
    
            velocity_magnitude = np.linalg.norm(velocity)
            normalized_velocity = velocity / velocity_magnitude if velocity_magnitude > 0 else velocity
            
            scaled_velocity = normalized_velocity * CAR_MAX_SPEED
                
            yaw = random.random()*2*np.pi
            
            car.set_pos(car_x, car_y, 0)
            car.set_rot(yaw=yaw)
            car.boost = random.random()
            car.set_lin_vel(scaled_velocity[0], scaled_velocity[1], 0)
            
            objPos.append(car.position)

class ReplayState(StateSetter):
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):     
        error = True
        while(error):
            try:
                error = False
                data_directory = REPLAY_FOLDER
                random_csv_file = random.choice(os.listdir(data_directory))
                random_csv_path = os.path.join(data_directory, random_csv_file)

                random_data_file = open(random_csv_path, 'r')
                random_data      = random_data_file.read()
                random_data_file.close()

                random_data = random_data.split("\n")
                for i in range(len(random_data)):
                    random_data[i] = random_data[i].split(",")


                random_data[0] = [re.sub(r'^(?<=\.)\.|\.(\d+)$', '', s) for s in random_data[0]]

                column = [(random_data[0][i], random_data[1][i]) for i in range(len(random_data[1]))]


                player1 = None
                player2 = None
                for name, _ in column:
                    if name != "ball" and name != "game":
                        if player1 == None:
                            player1 = name
                        elif name != player1:
                            player2 = name 


                row = random.randint(2, len(random_data))

                info_player1 = {}
                info_player2 = {}
                info_ball    = {}

                for i in range(len(random_data[row])):
                    if   column[i][0] == player1:
                        info_player1[column[i][1]] = random_data[row][i]
                    elif column[i][0] == player2:
                        info_player2[column[i][1]] = random_data[row][i]
                    elif column[i][0] == 'ball' :
                        info_ball[column[i][1]]    = random_data[row][i]  
                        
                state_wrapper.cars[0].set_pos(float(info_player1['pos_x']), float(info_player1['pos_y']), float(info_player1['pos_z']))
                state_wrapper.cars[1].set_pos(float(info_player2['pos_x']), float(info_player2['pos_y']), float(info_player2['pos_z']))
                state_wrapper.ball.set_pos(float(info_ball['pos_x']), float(info_ball['pos_y']), float(info_ball['pos_z']))
                
                state_wrapper.cars[0].set_lin_vel(float(info_player1['vel_x']), float(info_player1['vel_y']), float(info_player1['vel_z']))
                state_wrapper.cars[1].set_lin_vel(float(info_player2['vel_x']), float(info_player2['vel_y']), float(info_player2['vel_z']))
                state_wrapper.ball.set_lin_vel(float(info_ball['vel_x']), float(info_ball['vel_y']), float(info_ball['vel_z']))
                
                state_wrapper.cars[0].set_ang_vel(float(info_player1['ang_vel_x']), float(info_player1['ang_vel_y']), float(info_player1['ang_vel_z']))
                state_wrapper.cars[1].set_ang_vel(float(info_player2['ang_vel_x']), float(info_player2['ang_vel_y']), float(info_player2['ang_vel_z']))
                state_wrapper.ball.set_ang_vel(float(info_ball['ang_vel_x']), float(info_ball['ang_vel_y']), float(info_ball['ang_vel_z']))
                
                state_wrapper.cars[0].set_rot(float(info_player1['rot_x']), float(info_player1['rot_y']), float(info_player1['rot_z']))
                state_wrapper.cars[1].set_rot(float(info_player2['rot_x']), float(info_player2['rot_y']), float(info_player2['rot_z']))
                
                state_wrapper.cars[0].boost = float(info_player1['boost'])/255
                state_wrapper.cars[1].boost = float(info_player2['boost'])/255
                
            except Exception as e:
                error = True
        
        

    