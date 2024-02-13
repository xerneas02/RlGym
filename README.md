# ZZeer: Rocket League Deep Learning Bot

## Introduction

This project aims to develop a bot for Rocket League utilizing the capabilities of deep reinforcement learning. By harnessing deep learning techniques for complex data analysis and decision-making, along with reinforcement learning to optimize the bot's actions based on rewards received within the game environment, this initiative seeks to create an autonomous bot that continuously improves its performance. The project is implemented in Python using the RLGym environment.

## Module python

Python version 3.8.10

```bash
pip install rlgym stable-baselines3==1.7.0 torch
```

## RlBot

For RlBot you only need the ZZeer folder and there should be an rl_model.zip in the folder this should correspond to the last version of the model if you add this folder into your bot folder on RlBot you should see the ZZeer bot.

## Launching Training

If you want to launch a training you should first check **Constante.py** the useful constants are **GAME_SPEED** set to 1 for real-time and 100 for accelerated time, and **NUM_INSTANCE** which dictates the number of instances that will run simultaneously.
To launch the training run **RlGym.py**.

## Reward

All reward classes need to implement at least the methods **get_reward** and **get_final_reward**. The **get_reward** method is called at each step to get the reward of the bot and the **get_final_reward** method is called once at the end of the episode.
Some of this reward functions come from [Seer: Reinforcement Learning in Rocket League pdf](https://nevillewalo.ch/assets/docs/MA_Neville_Walo_Seer_RLRL.pdf)

### CombinedReward
This reward is a bit special; it's a reward that already exists in **rlgym.utils.reward_functions**. This version just adds some log messages when the game speed is at 1 to help debug the code.
The objective of this special reward is to combine multiple other rewards and give them different weights.

### GoalScoredReward
If the bot scores a goal, it receives a reward proportional to the speed of the ball.

### BoostDifferenceReward
This rewards the bot for collecting or using boost.

### BallTouchReward
Rewards the bot for touching the ball, with the reward varying based on the height of the ball.

### DemoReward
Rewards the bot for demolishing opponents.

### DistancePlayerBallReward
Rewards the bot for being close to the ball.

### DistanceBallGoalReward
Rewards the bot for having the ball close to the opponent's goal.

### FacingBallReward
Rewards the bot for facing the ball.

### AlignBallGoalReward
Rewards the bot for being aligned with the ball and the opponent's goal.

### ClosestToBallReward
Rewards the bot for being closest to the ball compared to opponents.

### TouchedLastReward
Rewards the bot for being the last to touch the ball.

### BehindBallReward
Rewards the bot for being between the ball and its own goal.

### VelocityPlayerBallReward
Rewards the bot for moving in the direction of the ball.

### KickoffReward
Rewards the bot during kickoff.

### VelocityReward
Rewards the bot for its velocity.

### BoostAmountReward
Rewards the bot based on its boost amount.

### ForwardVelocityReward
Rewards the bot for moving forward and penalizes backward movement.

### FirstTouchReward
Rewards the bot for being the first to touch the ball after kickoff.

### AirPenalityReward
Penalizes the bot for being in the air.

### DontTouchPenalityReward
Penalizes the bot for not touching the ball.

### VelocityBallOwnGoalReward
Rewards the bot based on the velocity of the ball towards its own goal.

### VelocityBallOpponentGoalReward
Rewards the bot based on the velocity of the ball towards the opponent's goal.

### SaveReward
Rewards the bot for saving a shot towards its own goal.

### DontGoalPenalityReward
Penalizes the bot for conceding a goal.

### BehindTheBallPenalityReward
Penalizes the bot for being behind the ball.

## StateSetter

State setters allow setting starting states with different car positions, car velocities, ball positions, ball velocities, boost amounts, etc. A StateSetter needs to implement the **reset** method.

### CombinedState
This state setter is inspired by the **CombinedReward** class. It allows combining multiple other state setters and assigning them different probabilities of appearing, as well as different weights for the rewards. The way it works is by providing it a tuple of tuples, with each inner tuple containing a StateSetter class and another tuple. If this inner tuple is left empty, it will keep the default weights of the rewards. If you fill it with as many floats as there are rewards, it will override the default reward weights. (if you put 42 as a reward weight, it will keep the default value)

### BetterRandom
This state setter is one of the state use by [Necto](https://github.com/Rolv-Arild/Necto) it generates random starting states with some constraints and distributions:

- **Ball Position**: Randomly selected within the game boundaries (`LIM_X`, `LIM_Y`) for the x and y coordinates and follows a triangular distribution for the z-coordinate between `BALL_RADIUS` and `LIM_Z`.
  
- **Ball Linear Velocity**: Generates a random velocity with a 99.9% chance of being below the maximum ball speed (`BALL_MAX_SPEED`). It does so by sampling from an exponential distribution to ensure that most velocities are below the maximum speed.
  
- **Ball Angular Velocity**: Randomly generates an angular velocity with a triangular distribution between 0 and the maximum angular velocity (`CAR_MAX_ANG_VEL`) plus a small offset.
  
- **Car Position**: Calculates a position for each car based on the position of the ball, ensuring that the car is, on average, 1 second away from the ball at maximum speed. If the calculated position is within the game boundaries (`LIM_X`, `LIM_Y`) and above the ground but still close to the ball, it uses that position. Otherwise, it falls back to generating a fully random position.
  
- **Car Linear Velocity**: Generates a random linear velocity with a triangular distribution between 0 and the maximum car speed (`CAR_MAX_SPEED`).
  
- **Car Orientation (Rotation)**: Randomly sets the car's pitch, yaw, and roll within specified limits (`PITCH_LIM`, `YAW_LIM`, `ROLL_LIM`), following triangular distributions.
  
- **Car Angular Velocity**: Randomly generates an angular velocity with a triangular distribution between 0 and the maximum car angular velocity (`CAR_MAX_ANG_VEL`).
  
- **Car Boost**: Assigns a random boost amount between 0 and 1 to each car.

### TrainingStateSetter
This state setter was found on the github [RL-in-RL](https://github.com/calebjjor/RL-in-RL/tree/main?tab=readme-ov-file) and is designed for training scenarios where different initial conditions are required to teach the agent specific skills or behaviors. It randomly selects from several predefined spawn states to provide a variety of training scenarios.

- **Attack**: In this scenario (selected approximately one-third of the time), the cars are positioned for an attacking play. Blue team cars are positioned near the center of the field, while orange team cars are positioned near the opponent's goal. Car orientations are adjusted accordingly, and a moderate boost amount is set for each car. The ball is placed at a random x-coordinate within the field and at a fixed y-coordinate (2816.0) with a zero linear velocity.

- **Defend**: This scenario (also selected approximately one-third of the time) sets up defensive play. Blue team cars are positioned near their own goal, while orange team cars are positioned near the center of the field. Car orientations and boost amounts are adjusted accordingly. The ball is placed near the center of the field at a specific location (-2816.0, 70.0) with a randomized linear velocity.

- **Center Spawn**: The remaining scenario (selected approximately one-third of the time) spawns cars centered on the agent's car. Blue team cars are positioned near the bottom of the field, while orange team cars are positioned near the top. Car orientations and boost amounts are set accordingly. The ball is placed above the center of the field with zero linear velocity.


### DefaultStateClose
This state setter is a variation of the default state setter where the initial positions of the cars are adjusted to create training scenarios with different distances from the ball at kickoff. It randomly selects a subset of predefined kickoff positions for both blue and orange team cars and adjusts their positions accordingly.

- **Initialization**:
  - The constructor takes an optional parameter `number_of_state`, which determines the number of different kickoff positions to choose from.
  - By default, it initializes with 2 different kickoff positions.

- **Reset**:
  - The `reset` method sets the initial state of the environment.
  - It first calculates a coefficient based on the number of state variations, inversely proportional to a randomly chosen integer between 1 and the provided `number_of_state`.
  - Predefined kickoff positions and orientations for both blue and orange teams are defined, with variations in distance from the ball.
  - The possible kickoff indices are shuffled to randomize the selection.
  - Blue and orange team cars are then positioned according to the selected kickoff positions, and their orientations are adjusted accordingly.
  - Each car is assigned a boost value of 0.33, representing a moderate boost amount for kickoff scenarios.
  - If `MOVE_BALL` is True, it invokes the `movement_ball` function to handle ball movement.


### RandomState
This state setter randomly distributes ball on the field while keeping the car on the default kickoff positions.

### InvertedState
This is a normal kickoff state but the car are facing away from the ball.

### DefaultStateCloseOrange / RandomStateOrange / InvertedStateOrange
Samething as **DefaultStateClose / RandomState / InvertedState** but the orange car are on the blue side and the blue are on the orange side.






