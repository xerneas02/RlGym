from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticCnnPolicy

class CustomNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        lstm_output_dim:int = 256,
        last_layer_dim_pi: int = 128,
        last_layer_dim_vf: int = 128,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf


        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(lstm_output_dim, 256), 
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256,last_layer_dim_vf),
            nn.LeakyReLU()
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(lstm_output_dim, 256), 
            nn.LeakyReLU(),
            nn.Linear(256,last_layer_dim_vf),
            nn.LeakyReLU()
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        lstm_output = self.lstm.forward(features)
        return self.forward_actor(lstm_output), self.forward_critic(lstm_output)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)
    

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(RecurrentActorCriticCnnPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomNetwork(self.features_dim)


