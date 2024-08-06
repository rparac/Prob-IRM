"""
Responsible for handing RM state transitions.
Abstracts away the complexity in computing next state in an RM.
"""

import abc
from typing import Optional

from rm_marl.reward_machine import RewardMachine

class RMTransitioner:
    def __init__(self, rm: Optional[RewardMachine]):
        self.rm = rm

    @abc.abstractmethod
    def get_initial_state(self):
        raise NotImplementedError("get_initial_state")

    @abc.abstractmethod
    def get_next_state(self, curr_state, event):
        raise NotImplementedError("get_next_state")