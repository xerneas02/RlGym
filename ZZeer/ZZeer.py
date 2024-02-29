import numpy as np
from sb3_contrib import RecurrentPPO
import pathlib
from Actions import ZeerLookupAction
from CustomPolicy import CustomNetwork, CustomActorCriticPolicy
from Extracor import CustomFeatureExtractor

class TrainedAgent:
    def __init__(self):
        _path = pathlib.Path(__file__).parent.resolve()
        custom_objects = {
            "lr_schedule": 0.000001,
            "clip_range": .02,
            "n_envs": 1,
        }
        
        self.actor = RecurrentPPO.load(str(_path) + '/rl_model')
        self.parser = ZeerLookupAction()

    def act(self, state):
        action = self.actor.predict(state)
        
        #print("Type of action:", type(action))
        #print("Shape of action:", action)
        
        x = self.parser.parse_actions(action[0], state)
        #print(x)

        return x
