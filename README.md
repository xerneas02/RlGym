# ZZeer: Rocket League Deep Learning Bot

## Introduction

This project aims to develop a bot for Rocket League utilizing the capabilities of deep reinforcement learning. By harnessing deep learning techniques for complex data analysis and decision-making, along with reinforcement learning to optimize the bot's actions based on rewards received within the game environment, this initiative seeks to create an autonomous bot that continuously improves its performance. The project is implemented in Python using the RLGym environment.

## Module python

Python version 3.8.10

```bash
pip install rlgym stable-baselines3==1.7.0 torch sb3-contrib rlgym_tools scipy shimmy>=0.2.1 stable-baselines3[extra]
```

## RlBot

For RlBot you only need the ZZeer folder and there should be an rl_model.zip in the folder this should correspond to the last version of the model if you add this folder into your bot folder on RlBot you should see the ZZeer bot.

## Launching Training

If you want to launch a training you should first check **Constante.py** the useful constants are **GAME_SPEED** set to 1 for real-time and 100 for accelerated time, and **NUM_INSTANCE** which dictates the number of instances that will run simultaneously.
To launch the training run **RlGym.py**.

## Log

During training, ZZeer generates log files to track the progress of the simulation and provide insights into the bot's performance.

### Tensorboard Logs

Tensorboard log files are generated during training and stored in the `logs` folder. These logs are organized into folders named `PPO_...`, corresponding to the specific training session. Tensorboard logs provide visualizations and metrics for monitoring the training process, including rewards, policy loss, value loss, and more.

### Reward Log

Every approximately 100,000 steps, the `log_rew.txt` file records the total amount of each reward gained by the bot during that period. This log provides a detailed breakdown of the rewards accumulated by the bot, helping to analyze its behavior and performance over time.

### Error Log

The `log_error.txt` file indicates if there was a crash during the training process. If a crash occurs, this log file will contain the crash message, enabling developers to diagnose and address any issues that may have occurred.

### Bot Restart Log

The `log.txt` file records each time the bot restarts during training. Bot restarts may occur if the bot fails to make significant progress over time or if the simulation finishes. This log provides information on the total number of steps performed by the bot, giving insights into its training progress and performance.

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
This state setter randomly distributes the ball on the field while keeping the cars in their default kickoff positions.

### InvertedState
This is a normal kickoff state, but the cars are facing away from the ball.

### DefaultStateCloseOrange / RandomStateOrange / InvertedStateOrange
Similar to **DefaultStateClose / RandomState / InvertedState**, but the orange cars are positioned on the blue side, and the blue cars are on the orange side.

### LineState
This state setter randomly places the ball and the cars in their respective zones on the field.

## Terminal Conditions
Terminal conditions are events that signal the end of an episode. They need to implement both a `reset` method and an `is_terminal` method.

#### BallTouchCondition
This terminal condition checks if there has been a touch on the ball since the last reset. It resets its state whenever the game state is reset.

#### NoTouchOrGoalTimeoutCondition
This terminal condition acts as a timeout condition but resets the timeout timer when the ball is touched for the first time or when a goal is scored. It initializes with a specified timeout duration.

#### NoTouchFirstTimeoutCondition
Similar to `NoTouchOrGoalTimeoutCondition`, this terminal condition resets the timeout timer when the ball is touched for the first time. It initializes with a specified timeout duration.

#### NoGoalTimeoutCondition
This terminal condition resets the timeout timer when a goal is scored. It initializes with a specified timeout duration and can be further modified by a coefficient.

#### AfterTouchTimeoutCondition
This terminal condition starts a timeout timer when the ball is touched for the first time by any player.

## ActionParser

An ActionParser is a component responsible for parsing action indices into concrete actions suitable for the environment. It defines how the action space is represented and how actions are translated from indices to their corresponding values.

### Description
- **Initialization**: Upon initialization, an ActionParser typically defines the bins or discrete values for each action dimension.
- **Action Space**: Provides a method to obtain the action space representation.
- **Action Parsing**: Parses action indices into concrete action values.

### ZeerLookupAction

This ActionParser implementation generates a lookup table for Rocket League actions based on predefined bins or discrete values for each action dimension (throttle, steer, pitch, yaw, roll). The lookup table covers both ground and aerial actions.

#### Behavior
- **Initialization**: Initializes with predefined bins for each action dimension.
- **Lookup Table Generation**: Generates a lookup table mapping action indices to concrete action values based on the defined bins. The lookup table includes actions for both ground and aerial maneuvers.
- **Action Space**: Defines the action space as a discrete space with the size equal to the number of entries in the lookup table.
- **Action Parsing**: Parses action indices into concrete action values using the generated lookup table.

## Observer

An Observation Builder, is responsible for constructing the observation vector provided to an agent at each timestep in a reinforcement learning environment. It determines what information the agent can perceive from the environment.

### ZeerObservations

This Observation Builder constructs observation vectors for Rocket League agents. It includes information about the ball, player's car, teammates, and opponents.

#### Behavior
- **Observation Construction**: Constructs the observation vector based on the state of the environment and the player's perspective. It includes information such as the position, velocity, and orientation of the ball and player's car, relative positions to the goal, boost amount, and other status indicators.
- **Player Perspective**: Considers the perspective of the player, determining whether to use regular or inverted data for the ball, boost pads, and player's car. The inverted perspective is used if the player is on the orange team to be sure that the perspective of the field is the same for the bot no matter what it's team is.
- **Normalization**: Normalizes position and velocity information based on predefined standard deviations.
- **Additional Information**: Includes additional information such as the relative position of teammates and opponents, relative positions to the attacking and defending goals, and various status indicators for the player.







