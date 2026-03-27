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
    conditions: List[object] = []
    if bool(config.get("end_on_goal", True)):
        conditions.append(GoalScoredCondition())
    conditions.append(NoTouchTimeoutCondition(int(config["no_touch_timeout_steps"])))
    conditions.append(TimeoutCondition(int(config["timeout_steps"])))
    return conditions
