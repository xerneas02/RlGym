from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Dict, List, Tuple, Type, Union

import gymnasium as gym
import torch as th
from gymnasium import spaces
from torch import nn

from stable_baselines3.common.preprocessing import get_flattened_obs_dim, is_image_space
from stable_baselines3.common.type_aliases import TensorDict
from stable_baselines3.common.utils import get_device

class CustomFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim : int = 256) -> None:
        super().__init__(observation_space, features_dim)
        self.extractor = nn.Sequential(
                nn.Flatten(),
                nn.Linear(get_flattened_obs_dim(observation_space),features_dim),
                nn.LeakyReLU()
        )
    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.extractor.forward(observations)