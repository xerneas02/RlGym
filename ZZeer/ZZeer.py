import numpy as np
from stable_baselines3 import PPO
import pathlib
from Actions import ZeerLookupAction
    

class TrainedAgent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "n_envs": 1,
        }
        
        self.actor = PPO.load(str(_path) + '/rl_model', device='cpu', custom_objects=custom_objects)
        self.parser = ZeerLookupAction()

    def act(self, state):
        action = self.actor.predict(state)
        
        #print("Type of action:", type(action))
        #print("Shape of action:", action)
        
        x = self.parser.parse_actions(action[0], state)
        #print(x)

        return x
