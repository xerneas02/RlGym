import Terminal
import State

from State import CombinedState

from rlgym.utils.gamestates import PlayerData, GameState

from rlgym.utils.terminal_conditions import TerminalCondition
from rlgym.utils.terminal_conditions import common_conditions

from typing import Any, Optional, Tuple, overload, Union, List

class CustomTerminalCondition(TerminalCondition):
    def __init__(self, tuple_state : Tuple[List[TerminalCondition], ...]):
        super().__init__()
        self.terminalcond_per_state = tuple_state
        
    def reset(self, initial_state: GameState):
        for i in range(len(self.terminalcond_per_state[CombinedState.get_current_state()])):
            self.terminalcond_per_state[CombinedState.get_current_state()][i].reset(initial_state)
    
    def is_terminal(self, current_state: GameState) -> bool:
        term = False
        i = 0
        while i < len(self.terminalcond_per_state[CombinedState.get_current_state()]) and not term:
            term = term or self.terminalcond_per_state[CombinedState.get_current_state()][i].is_terminal(current_state)
            i += 1
        return term