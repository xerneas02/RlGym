import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
import numpy as np
import os
    
    
class TrainedAgent:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(cur_dir, "rl_model.zip")
        self.agent = PPO.load(model_path, env=None)

    def act(self, state):
        state = state[:70]

        return self.agent.predict(state)[0]
