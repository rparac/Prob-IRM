import math
from typing import Dict, Iterable, Optional

import numpy as np

from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_transition.rm_transitioner import RMTransitioner


class ProbRMTransitioner(RMTransitioner):
    # rm should be provided unless it is learned.
    # In that case, the rm parameter will be set by the learning agent
    def __init__(self, rm: Optional[RewardMachine]):
        super().__init__(rm)

    def get_initial_state(self):
        assert isinstance(self.rm.u0, (str, int))
        u0 = np.zeros(len(self.rm.states))
        curr_true_state = self.rm.states.index(self.rm.u0)
        u0[curr_true_state] = 1

        return u0

    def get_next_state(self, curr_state: np.ndarray, event: Dict[str, float]) -> np.ndarray:
        assert isinstance(curr_state, np.ndarray)
        assert isinstance(event, dict)

        # return self.get_next_state_simple(curr_state, event.copy())

        belief_out = np.zeros(curr_state.shape)

        label_probs = event.copy()
        # Adds absorbing state labels
        label_probs["True"] = 1
        label_probs["False"] = 1

        for u_from_idx in range(belief_out.size):
            u_from = self.rm.states[u_from_idx]
            transition_prob_sum = 0
            for transition_labels, u_out in self.rm.transitions[u_from].items():
                u_out_idx = self.rm.to_idx(u_out)
                transition_prob = self.compute_transition_probability(transition_labels, label_probs)
                belief_out[u_out_idx] += transition_prob * curr_state[u_from_idx]
                transition_prob_sum += transition_prob
            # The transitions to the same state are not captured with the transitions variable.
            # So, 1 - transition_prob_sum -> the probability of transitioning to the same state
            belief_out[u_from_idx] += (1 - transition_prob_sum) * curr_state[u_from_idx]

        return belief_out

    @staticmethod
    def compute_transition_probability(transition_labels: Iterable[str], label_probs: Dict[str, float]) -> float:
        transition_label_probs = [1 - label_probs[label[1:]] if label.startswith("~") else label_probs[label]
                                  for label in transition_labels]
        return math.prod(transition_label_probs)

    # Simplified version
    # TODO: remove
    def get_next_state_simple(self, curr_state, label_probs):
        assert np.isclose(sum(curr_state), 1, rtol=1e-5, atol=1e-5)
        belief_out = curr_state.copy()
        if label_probs['by'] > 0.5:
            belief_out[0] = 0
            belief_out[1] = 1
        if label_probs['g'] > 0.5:
            if belief_out[0] == 1:
                belief_out[0] = 0
                belief_out[2] = 1
            else:
                belief_out[1] = 0
                belief_out[3] = 1

        assert np.isclose(sum(belief_out), 1, rtol=1e-5, atol=1e-5)
        return belief_out
