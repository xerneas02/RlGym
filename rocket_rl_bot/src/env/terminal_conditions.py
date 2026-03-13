from __future__ import annotations

from typing import Dict, List

try:
    from rlgym_sim.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition,
        NoTouchTimeoutCondition,
        TimeoutCondition,
    )
except ImportError:  # pragma: no cover
    from rlgym.utils.terminal_conditions.common_conditions import (
        GoalScoredCondition,
        NoTouchTimeoutCondition,
        TimeoutCondition,
    )


def build_terminal_conditions(config: Dict) -> List[object]:
    return [
        GoalScoredCondition(),
        NoTouchTimeoutCondition(int(config["no_touch_timeout_steps"])),
        TimeoutCondition(int(config["timeout_steps"])),
    ]
