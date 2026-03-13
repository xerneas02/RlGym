from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_sizes=(256, 256), activation=nn.ReLU) -> None:
        super().__init__()
        layers = []
        last_dim = input_dim
        for hidden_dim in hidden_sizes:
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(activation())
            last_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.output_dim = last_dim

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int) -> None:
        super().__init__()
        self.policy_body = MLP(obs_dim, hidden_sizes=(256, 256), activation=nn.ReLU)
        self.value_body = MLP(obs_dim, hidden_sizes=(256, 256), activation=nn.ReLU)
        self.policy_head = nn.Linear(self.policy_body.output_dim, action_dim)
        self.value_head = nn.Linear(self.value_body.output_dim, 1)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        policy_features = self.policy_body(observations)
        value_features = self.value_body(observations)
        logits = self.policy_head(policy_features)
        values = self.value_head(value_features).squeeze(-1)
        return logits, values

    def act(self, observations: torch.Tensor, deterministic: bool = False):
        logits, values = self.forward(observations)
        distribution = Categorical(logits=logits)
        if deterministic:
            actions = torch.argmax(logits, dim=-1)
        else:
            actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        return actions, log_probs, values

    def evaluate_actions(self, observations: torch.Tensor, actions: torch.Tensor):
        logits, values = self.forward(observations)
        distribution = Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy, values
