from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
from rlgym.utils.common_values import BLUE_TEAM, ORANGE_TEAM, CEILING_Z
import numpy as np

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
        
from rlgym.utils.state_setters import StateSetter
from rlgym.utils.state_setters import StateWrapper
import numpy as np
import random


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