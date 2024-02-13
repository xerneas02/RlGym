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
This state setter is inspired by the **CombinedReward** class. It allows combining multiple other state setters and assigning them different probabilities of appearing, as well as different weights for the rewards. The way it works is by providing it a tuple of tuples, with each inner tuple containing a StateSetter class and another tuple. If this inner tuple is left empty, it will keep the default weights of the rewards. If you fill it with as many floats as there are rewards, it will override the default reward weights. For example, if you put 42 as a reward weight, it will keep the default value.




