import random
from typing import List, Dict, Set

import numpy as np

from rm_marl.rm_learning.ilasp.ilasp_example_representation import ObservablePredicate, LastPredicate, \
    ISAILASPExample
from rm_marl.rm_learning.trace_tracker import NoisyTraceTracker


# Computes the classification examples for the logic-based learning method
# TODO: Merge this with ProbFFNSLLearner
class NoisyILASPExampleGenerator:
    def __init__(self):
        # Number of samples
        self.I = 100
        # Large number to reduce the error of rounding to int
        self.K = 1000
        # The probability that an example is incorrect
        self.epsilon = 0.1  # 0.0000001

        # The number of rules we expect with length 1
        self.r_1 = 100
        # The multiplier from the number of rules we expect of length n compared to n-1
        self.r_mult = 2 / 3
        self.max_rule_len = 10

        self.ex_counter = 0

        random.seed(0)

    def create_examples_from(self, trace: NoisyTraceTracker) -> List[ISAILASPExample]:
        if trace.is_complete:
            if trace.is_positive:
                ex_type = ISAILASPExample.ExType.GOAL
            else:
                ex_type = ISAILASPExample.ExType.DEND
        else:
            ex_type = ISAILASPExample.ExType.INCOMPLETE

        sol = []
        for i in range(self.I):
            ex_id = f"ex_{self.ex_counter}"
            context = self.create_example_context(trace)
            penalty = np.round(-self.K * np.log(self.epsilon / (1 - self.epsilon)) / self.I).astype(int)
            last_predicate = LastPredicate(len(trace.trace) - 1)
            sol.append(ISAILASPExample(ex_id, penalty, ex_type, context, last_predicate))
            self.ex_counter += 1
        return sol

    def create_example_context(self, trace: NoisyTraceTracker) -> Set[ObservablePredicate]:
        assert len(trace.trace) > 0
        # Create context
        sol = []
        for time_step, labels in enumerate(trace.trace):
            true_labels = self._sample_dict(labels)
            predicates = [ObservablePredicate(label, time_step) for label in true_labels]
            sol.extend(predicates)
        return set(sol)

    # labels - dictionary of labels paired with their probability
    # returns: keys which are considered as true
    def _sample_dict(self, labels: Dict[str, float]) -> List[str]:
        true_elems = []
        for label, prob in labels:
            if random.random() <= prob:
                true_elems.append(label)
        return true_elems
