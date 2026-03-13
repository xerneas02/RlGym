from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

try:
    from gym.spaces import Discrete
except ImportError:  # pragma: no cover
    Discrete = None

try:
    from rlgym_sim.utils.action_parsers import ActionParser
except ImportError:  # pragma: no cover
    from rlgym.utils.action_parsers import ActionParser


ActionVector = Sequence[float]


def _as_action(
    throttle: float,
    steer: float,
    pitch: float,
    yaw: float,
    roll: float,
    jump: int,
    boost: int,
    handbrake: int,
) -> np.ndarray:
    return np.asarray(
        [throttle, steer, pitch, yaw, roll, float(jump), float(boost), float(handbrake)],
        dtype=np.float32,
    )


def _deduplicate(actions: Iterable[np.ndarray]) -> np.ndarray:
    seen = set()
    unique: List[np.ndarray] = []
    for action in actions:
        key = tuple(float(x) for x in action.tolist())
        if key in seen:
            continue
        seen.add(key)
        unique.append(action)
    return np.stack(unique, axis=0)


def build_lookup_table() -> np.ndarray:
    """
    96 actions, intentionally capped to keep exploration efficient.

    Restrictions:
    - No boost while reversing.
    - No jump + handbrake combinations.
    - No roll on grounded driving actions.
    - No contradictory aerial controls with full pitch and full roll simultaneously.
    - Powerslide is only kept for meaningful turn angles.
    """

    grounded = [
        _as_action(0, 0, 0, 0, 0, 0, 0, 0),
        _as_action(1, 0, 0, 0, 0, 0, 0, 0),
        _as_action(0.5, 0, 0, 0, 0, 0, 0, 0),
        _as_action(-1, 0, 0, 0, 0, 0, 0, 0),
        _as_action(1, 1, 0, 1, 0, 0, 0, 0),
        _as_action(1, 0.5, 0, 0.5, 0, 0, 0, 0),
        _as_action(0.5, 0.5, 0, 0.5, 0, 0, 0, 0),
        _as_action(1, -0.5, 0, -0.5, 0, 0, 0, 0),
        _as_action(0.5, -0.5, 0, -0.5, 0, 0, 0, 0),
        _as_action(1, -1, 0, -1, 0, 0, 0, 0),
        _as_action(0, 1, 0, 1, 0, 0, 0, 0),
        _as_action(0, -1, 0, -1, 0, 0, 0, 0),
        _as_action(-1, 1, 0, 1, 0, 0, 0, 0),
        _as_action(-1, -1, 0, -1, 0, 0, 0, 0),
        _as_action(1, 1, 0, 1, 0, 0, 0, 1),
        _as_action(1, -1, 0, -1, 0, 0, 0, 1),
        _as_action(1, 0.5, 0, 0.5, 0, 0, 0, 1),
        _as_action(1, -0.5, 0, -0.5, 0, 0, 0, 1),
        _as_action(0, 1, 0, 1, 0, 0, 0, 1),
        _as_action(0, -1, 0, -1, 0, 0, 0, 1),
        _as_action(-1, 1, 0, 1, 0, 0, 0, 1),
        _as_action(-1, -1, 0, -1, 0, 0, 0, 1),
    ]

    boosted_ground = [
        _as_action(1, 0, 0, 0, 0, 0, 1, 0),
        _as_action(1, 0.5, 0, 0.5, 0, 0, 1, 0),
        _as_action(1, -0.5, 0, -0.5, 0, 0, 1, 0),
        _as_action(1, 0.75, 0, 0.75, 0, 0, 1, 0),
        _as_action(1, -0.75, 0, -0.75, 0, 0, 1, 0),
        _as_action(1, 1, 0, 1, 0, 0, 1, 0),
        _as_action(1, -1, 0, -1, 0, 0, 1, 0),
        _as_action(1, 1, 0, 1, 0, 0, 1, 1),
        _as_action(1, -1, 0, -1, 0, 0, 1, 1),
        _as_action(1, 0.5, 0, 0.5, 0, 0, 1, 1),
        _as_action(1, -0.5, 0, -0.5, 0, 0, 1, 1),
        _as_action(1, 0, 0, 0, 0, 0, 1, 1),
    ]

    takeoffs = [
        _as_action(0, 0, 0, 0, 0, 1, 0, 0),
        _as_action(1, 0, 0, 0, 0, 1, 0, 0),
        _as_action(1, 0, -1, 0, 0, 1, 0, 0),
        _as_action(1, 0, 1, 0, 0, 1, 0, 0),
        _as_action(1, 0, 0, 1, 0, 1, 0, 0),
        _as_action(1, 0, 0, -1, 0, 1, 0, 0),
        _as_action(1, 1, 0, 1, 0, 1, 0, 0),
        _as_action(1, -1, 0, -1, 0, 1, 0, 0),
    ]

    dodges = [
        _as_action(1, 0, -1, 0, 0, 1, 0, 0),
        _as_action(1, 0, 1, 0, 0, 1, 0, 0),
        _as_action(1, 0, -1, -1, 0, 1, 0, 0),
        _as_action(1, 0, -1, 1, 0, 1, 0, 0),
        _as_action(1, 0, 1, -1, 0, 1, 0, 0),
        _as_action(1, 0, 1, 1, 0, 1, 0, 0),
        _as_action(1, 0, 0, -1, 0, 1, 0, 0),
        _as_action(1, 0, 0, 1, 0, 1, 0, 0),
        _as_action(1, 0, -0.5, -1, 0, 1, 0, 0),
        _as_action(1, 0, -0.5, 1, 0, 1, 0, 0),
        _as_action(1, 0, 0.5, -1, 0, 1, 0, 0),
        _as_action(1, 0, 0.5, 1, 0, 1, 0, 0),
        _as_action(1, 0, -1, 0, 0, 1, 1, 0),
        _as_action(1, 0, -1, -1, 0, 1, 1, 0),
        _as_action(1, 0, -1, 1, 0, 1, 1, 0),
        _as_action(1, 0, 0, -1, 0, 1, 1, 0),
        _as_action(1, 0, 0, 1, 0, 1, 1, 0),
        _as_action(1, 0, 1, 0, 0, 1, 1, 0),
    ]

    aerial = [
        _as_action(1, 0, -1, 0, 0, 0, 0, 0),
        _as_action(1, 0, 1, 0, 0, 0, 0, 0),
        _as_action(1, 0, 0, -1, 0, 0, 0, 0),
        _as_action(1, 0, 0, 1, 0, 0, 0, 0),
        _as_action(1, 0, -1, -1, 0, 0, 0, 0),
        _as_action(1, 0, -1, 1, 0, 0, 0, 0),
        _as_action(1, 0, 1, -1, 0, 0, 0, 0),
        _as_action(1, 0, 1, 1, 0, 0, 0, 0),
        _as_action(1, 0, 0, 0, -1, 0, 0, 0),
        _as_action(1, 0, 0, 0, 1, 0, 0, 0),
        _as_action(1, 0, -0.5, 0, -1, 0, 0, 0),
        _as_action(1, 0, -0.5, 0, 1, 0, 0, 0),
        _as_action(1, 0, 0.5, 0, -1, 0, 0, 0),
        _as_action(1, 0, 0.5, 0, 1, 0, 0, 0),
        _as_action(1, 0, -1, 0, -1, 0, 0, 0),
        _as_action(1, 0, -1, 0, 1, 0, 0, 0),
        _as_action(1, 0, 1, 0, -1, 0, 0, 0),
        _as_action(1, 0, 1, 0, 1, 0, 0, 0),
        _as_action(1, 0, -1, -1, -1, 0, 0, 0),
        _as_action(1, 0, -1, 1, 1, 0, 0, 0),
        _as_action(1, 0, 1, -1, -1, 0, 0, 0),
        _as_action(1, 0, 1, 1, 1, 0, 0, 0),
        _as_action(1, 0, -1, 0, 0, 0, 1, 0),
        _as_action(1, 0, 1, 0, 0, 0, 1, 0),
        _as_action(1, 0, 0, -1, 0, 0, 1, 0),
        _as_action(1, 0, 0, 1, 0, 0, 1, 0),
        _as_action(1, 0, -1, -1, 0, 0, 1, 0),
        _as_action(1, 0, -1, 1, 0, 0, 1, 0),
        _as_action(1, 0, 1, -1, 0, 0, 1, 0),
        _as_action(1, 0, 1, 1, 0, 0, 1, 0),
        _as_action(1, 0, 0, 0, -1, 0, 1, 0),
        _as_action(1, 0, 0, 0, 1, 0, 1, 0),
        _as_action(1, 0, -0.5, 0, -1, 0, 1, 0),
        _as_action(1, 0, -0.5, 0, 1, 0, 1, 0),
        _as_action(1, 0, 0.5, 0, -1, 0, 1, 0),
        _as_action(1, 0, 0.5, 0, 1, 0, 1, 0),
        _as_action(1, 0, -1, -1, -1, 0, 1, 0),
        _as_action(1, 0, -1, 1, 1, 0, 1, 0),
        _as_action(1, 0, 1, -1, -1, 0, 1, 0),
        _as_action(1, 0, 1, 1, 1, 0, 1, 0),
    ]

    lookup_table = _deduplicate([*grounded, *boosted_ground, *takeoffs, *dodges, *aerial])
    if not 90 <= len(lookup_table) <= 120:
        raise ValueError(f"Lookup table size must stay in [90, 120], got {len(lookup_table)}")
    return lookup_table


@dataclass
class ActionSpaceSummary:
    size: int
    dimensions: int = 8


class OptimizedDiscreteAction(ActionParser):
    def __init__(self) -> None:
        super().__init__()
        self._lookup_table = build_lookup_table()
        self.summary = ActionSpaceSummary(size=int(self._lookup_table.shape[0]))

    @property
    def lookup_table(self) -> np.ndarray:
        return self._lookup_table

    def get_action_space(self):  # pragma: no cover - depends on installed gym
        if Discrete is None:
            return self.summary.size
        return Discrete(self.summary.size)

    def parse_actions(self, actions: np.ndarray, state=None) -> np.ndarray:
        indices = np.asarray(actions, dtype=np.int64).reshape(-1)
        return self._lookup_table[indices]

    def get_action(self, index: int) -> np.ndarray:
        return self._lookup_table[int(index)].copy()

    def nearest_action_index(self, action: ActionVector) -> int:
        target = np.asarray(action, dtype=np.float32).reshape(1, -1)
        distances = np.square(self._lookup_table - target).sum(axis=1)
        return int(np.argmin(distances))
