from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, CAR_MAX_ANG_VEL, BALL_MAX_SPEED
from rlgym.utils.math import rand_vec3
from Reward import CombinedReward

import numpy as np
import random
import math
from Constante import *

from typing import Any, Optional, Tuple, overload, Union



            
            
class CombinedState(StateSetter):
    
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
                ("Reward functions list length ({0}) and reward weights " \
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

class StateSetterInit(StateSetter):
    def reset(self, state_wrapper: StateWrapper):
    
        # Set up our desired spawn location and orientation. Here, we will only change the yaw, leaving the remaining orientation values unchanged.
        desired_car_pos = [100,100,17] #x, y, z
        desired_yaw = np.pi/2
        
        # Loop over every car in the game.
        for car in state_wrapper.cars:
            if car.team_num == BLUE_TEAM:
                pos = desired_car_pos
                yaw = desired_yaw
                
            elif car.team_num == ORANGE_TEAM:
                # We will invert values for the orange team so our state setter treats both teams in the same way.
                pos = [-1*coord for coord in desired_car_pos]
                yaw = -1*desired_yaw
                
            # Now we just use the provided setters in the CarWrapper we are manipulating to set its state. Note that here we are unpacking the pos array to set the position of 
            # the car. This is merely for convenience, and we will set the x,y,z coordinates directly when we set the state of the ball in a moment.
            car.set_pos(*pos)
            car.set_rot(yaw=yaw)
            car.boost = 0.72
            
        # Now we will spawn the ball in the center of the field, floating in the air.
        state_wrapper.ball.set_pos(x=0, y=0, z=CEILING_Z/2)
        



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


    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies state_wrapper values to emulate a randomly selected default kickoff.

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        
        coef = 1/random.randint(1, 2)
        
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
        
        wall_x = SIDE_WALL_X - BALL_RADIUS
        wall_y = BACK_WALL_Y - BALL_RADIUS
        
        ball_x = random.randint(-int(wall_x), int(wall_x))
        ball_y = random.randint(-int(wall_y), int(wall_y))
        
        min_distance = 200

        def distance(p1, p2):
            return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        ball_x = random.randint(-int(wall_x), int(wall_x))
        ball_y = random.randint(-int(wall_y), int(wall_y))

        while any(distance([ball_x, ball_y, 0], car_pos) < min_distance for car_pos in SPAWN_BLUE_POS + SPAWN_ORANGE_POS):
            ball_x = random.randint(-int(wall_x), int(wall_x))
            ball_y = random.randint(-int(wall_y), int(wall_y))

        print("Position de la balle (x, y):", ball_x, ball_y)        
        
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
    
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        
        count = 0
        wall_x = SIDE_WALL_X - BALL_RADIUS
        #wall_y = BACK_WALL_Y - BALL_RADIUS
        ceiling = CEILING_Z - BALL_RADIUS
        #----SPAWN BOUBOULE--------------------------------
        ball_x = random.randint(-int(wall_x), int(wall_x))
        #ball_y = random.randint(-int(wall_y), int(wall_y))
        ball_y = 0
        ball_z = random.randint(int(BALL_RADIUS)+1, int(ceiling/2))
        state_wrapper.ball.set_pos(ball_x, ball_y, ball_z)
        #---------------------------------------------------
        #----SPAWN CARS-------------------------------------
        for car in state_wrapper.cars:
            car_x = random.randint(-int(wall_x), int(wall_x))
            #ball_y = random.randint(-int(wall_y), int(wall_y))
            car_y = 3000 if count == 1  else -3000
            yaw = -0.5 * np.pi if count == 1 else 0.5 * np.pi
            car_z = 17
            car.set_pos(car_x, car_y, car_z)
            car.boost = 0.33
            car.set_rot(yaw=yaw)
            count = count + 1
        #---------------------------------------------------