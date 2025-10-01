from typing import Optional

import numpy as np

from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_transition.rm_transitioner import RMTransitioner


class DeterministicRMTransitioner(RMTransitioner):
    # None can be provided if we are learning the RM
    def __init__(self, rm: Optional[RewardMachine]):
        super().__init__(rm)

    def get_initial_state(self):
        assert isinstance(self.rm.u0, (str, int))
        return self._to_one_hot(self.rm.u0)

    def get_next_state(self, curr_state, event):
        curr_state_idx = np.argmax(curr_state)
        curr_state = self.rm.states[curr_state_idx]
        assert isinstance(curr_state, (str, int))

        if not isinstance(event, (list, tuple)):
            event = (event,)

        u = curr_state
        if u in self.rm.transitions:
            for condition in self.rm.transitions[u]:
                if not isinstance(condition, (list, tuple)):
                    condition = (condition,)
                if all(self._is_event_satisfied(c, event) for c in condition):
                    u = self.rm.transitions[u][condition]

        return self._to_one_hot(u)

    def _to_one_hot(self, u):
        idx = self.rm.to_idx(u)
        u_one_hot = np.zeros(len(self.rm.states), dtype=np.float32)
        u_one_hot[idx] = 1
        return u_one_hot

    @staticmethod
    def _is_event_satisfied(condition, observations):
        if not condition:  # empty conditions = unconditional transition (always taken)
            return True

        # check if some condition in the array does not hold (conditions are AND)
        if condition.startswith("~"):
            fluent = condition[1:]  # take literal without the tilde
            if fluent in observations:
                return False
        else:
            if condition not in observations:
                return False
        return True