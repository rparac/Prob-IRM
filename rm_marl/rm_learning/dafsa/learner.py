"""
DAFSA library
https://joss.theoj.org/papers/10.21105/joss.01986
"""
import itertools
import os

import networkx as nx
from dafsa import DAFSA

from ...reward_machine import RewardMachine
from ...utils.logging import getLogger
from ..learner import RMLearner

try:
    from itertools import pairwise
except ImportError:
    def pairwise(iterable):
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

LOGGER = getLogger(__name__)

class DAFSALearner(RMLearner):

    def __init__(self, agent_id, num_examples: int = 50):
        super().__init__(agent_id)
        self._previous_examples = []
        self.num_examples = num_examples

    def learn(self, _observables, rm, positive_examples, _negative_examples, _incomplete_examples):
        positive_examples = self.process_examples(positive_examples)

        if not positive_examples:
            LOGGER.debug(f"[{self.agent_id}] No positive examples")
            return

        selected_examples = sorted(positive_examples, key=len)[:self.num_examples]
        if selected_examples == self._previous_examples:
            return
            
        self.rm_learning_counter += 1

        automaton = DAFSA(selected_examples)
        self._previous_examples = selected_examples
        candidate_rm = self._generate_rm(automaton)

        if candidate_rm.states:
            candidate_rm.set_u0("u0")
            candidate_rm.set_uacc("u_acc")

            if candidate_rm != rm:
                LOGGER.debug(f"[{self.agent_id}] New RM found.")
                rm_plot_filename = os.path.join(
                    self.log_folder, f"plot_{self.rm_learning_counter}"
                )
                candidate_rm.plot(rm_plot_filename)
                return candidate_rm

    def process_examples(self, examples):
        return set(examples)

    def _generate_rm(self, automaton):
        rm = RewardMachine()

        final_nodes = [nid for nid, n in automaton.nodes.items() if n.final]
        graph = automaton.to_graph()
        _, shortest_path = nx.multi_source_dijkstra(graph, sources=final_nodes, target=0, weight=lambda *args: 1)
        shortest_sequence = [graph.edges[e]["label"] for e in pairwise(shortest_path)][::-1]

        for i, event in enumerate(shortest_sequence[:-1]):
            rm.add_transition(f"u{i}", f"u{i+1}", event)
        rm.add_transition(f"u{len(shortest_sequence)-1}", "u_acc", shortest_sequence[-1])

        return rm

