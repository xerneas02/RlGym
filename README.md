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

