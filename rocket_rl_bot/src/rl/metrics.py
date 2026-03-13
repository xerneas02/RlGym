from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Iterable, List

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
):
    steps, actors = rewards.shape
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(actors, dtype=np.float32)
    for step in reversed(range(steps)):
        if step == steps - 1:
            next_value = next_values
        else:
            next_value = values[step + 1]
        non_terminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * non_terminal - values[step]
        last_advantage = delta + gamma * gae_lambda * non_terminal * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages, returns


def explained_variance(predictions: np.ndarray, targets: np.ndarray) -> float:
    target_var = np.var(targets)
    if target_var <= 1e-8:
        return 0.0
    return float(1.0 - np.var(targets - predictions) / target_var)


class MetricWindow:
    def __init__(self, maxlen: int = 200) -> None:
        self.maxlen = maxlen
        self.storage: Dict[str, Deque[float]] = {}

    def add(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            if key not in self.storage:
                self.storage[key] = deque(maxlen=self.maxlen)
            self.storage[key].append(float(value))

    def means(self) -> Dict[str, float]:
        return {
            key: float(np.mean(values))
            for key, values in self.storage.items()
            if len(values) > 0
        }
