import itertools
import os
from collections import defaultdict

from ...reward_machine import RewardMachine
from ...utils.logging import getLogger
from ..learner import RMLearner
from .samp2symb.algo.infer_specific import find_specific_dfa
from .samp2symb.base.dfa import DFA
from .samp2symb.base.trace import AlphaTrace, Sample

try:
    from itertools import pairwise
except ImportError:
    def pairwise(iterable):
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

LOGGER = getLogger(__name__)

class S2SLearner(RMLearner):

    def __init__(self, agent_id, rm_learning_freq: int = 50, max_same_rm: int = 5):
        super().__init__(agent_id)

        self.automaton = None
        self.rm_learning_freq = rm_learning_freq
        self.max_same_rm = max_same_rm
        self.same_rm = 0

        self._num_prev_examples = 0

    def process_examples(self, examples):
        alpha_traces = []
        for t in examples:
            trace = AlphaTrace.loads("".join(str(tuple(sorted(i))) for i in t))
            trace.intendedEvaluation = True
            alpha_traces.append(trace)
        return sorted(alpha_traces, key=lambda i: len(i.vector))

    def learn(self, observables, rm, orig_positive_examples, _negative_examples, _incomplete_examples):
        positive_examples = self.process_examples(orig_positive_examples)

        if self.same_rm > self.max_same_rm or not positive_examples or len(positive_examples) % self.rm_learning_freq != 0:
            LOGGER.debug(f"[{self.agent_id}] No positive examples: {len(orig_positive_examples)}")
            return
            
        if self._num_prev_examples == len(positive_examples):
            return
        
        self._num_prev_examples = len(positive_examples)
        self.rm_learning_counter += 1

        self.automaton = self._build_automaton(positive_examples, observables)
        candidate_rm = self._generate_rm(self.automaton)

        if candidate_rm.states:

            if candidate_rm != rm:
                self.same_rm = 0
                LOGGER.debug(f"[{self.agent_id}] New RM found.")
                rm_plot_filename = os.path.join(
                    self.log_folder, f"plot_{self.rm_learning_counter}"
                )
                candidate_rm.plot(rm_plot_filename)
                return candidate_rm
            else:
                self.same_rm += 1

    def _build_automaton(self, positive_examples, observables):
        sample = Sample(positive_examples, [])
        try:
            automaton = find_specific_dfa(
                sample,
                dfa=self.automaton,
                start_size=1,
                max_size=len(observables),
                force_nsup=True, 
                force_sub=True,
                timeout=1000,
            )
        except:
            automaton = find_specific_dfa(
                sample,
                start_size=1,
                max_size=len(observables),
                force_nsup=True, 
                force_sub=True,
                timeout=1000,
            )

        automaton.export_dot(os.path.join(
            self.log_folder, f"plot_{self.rm_learning_counter}_{self.agent_id}_original.dot"
        ))

        def powerset(iterable):
            from itertools import chain, combinations
            s = list(iterable)  # allows duplicate elements
            return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

        def has_out_transisitons(s, a):
            return bool([v for v in a.transitions.get(s, {}).values() if v != s])

        def is_dead_end(s, a):
            return s not in a.final_states and not has_out_transisitons(s, a)

        automaton = automaton.translate(
            alphabet=lambda i: tuple(sorted(i.split("_"))), 
            states=lambda i: None if is_dead_end(i, automaton) else i
        )

        for s in automaton.states:
            labels = [l for l, v in automaton.transitions.get(s, {}).items() if v == s]
            _ = [automaton.transitions[s].pop(l) for l in labels]
        
        _ = [automaton.transitions.pop(fs) for fs in automaton.final_states]

        automaton.alphabet = list(powerset(sorted({l for ls in sample.alphabet for l in ls.split("_")})))

        automaton.export_dot(os.path.join(
            self.log_folder, f"plot_{self.rm_learning_counter}_{self.agent_id}_pruned.dot"
        ))

        return automaton


    def _generate_rm(self, automaton):
        rm = RewardMachine()

        init_states = list(automaton.init_states)
        final_states = list(automaton.accepting_states)

        for u in automaton.transitions.keys():
            for l, v in automaton.transitions[u].items():
                rm.add_transition(u, v, l)

        rm.set_u0(init_states[0])
        rm.set_uacc(final_states[0])

        return rm

