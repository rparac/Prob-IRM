import itertools
import os

import networkx as nx
from dafsa import DAFSA

from ...reward_machine import RewardMachine
from ...utils.logging import getLogger
from ..learner import RMLearner

LOGGER = getLogger(__name__)

class DAFSALearner(RMLearner):

    def __init__(self, agent_id):
        super().__init__(agent_id)
        self._previous_examples = []

    def learn(self, _observables, rm, positive_examples, _negative_examples, _incomplete_examples):
        
        if not positive_examples:
            LOGGER.debug(f"[{self.agent_id}] No positive examples")
            return

        selected_examples = sorted(positive_examples, key=len)[:50]
        if selected_examples == self._previous_examples:
            return
            
        self.rm_learning_counter += 1

        candidate_rm = self._generate_rm(selected_examples)
        self._previous_examples = selected_examples

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

    def _generate_rm(self, positive_examples):
        automaton = DAFSA(
            positive_examples
        )

        rm = RewardMachine()

        final_nodes = [nid for nid, n in automaton.nodes.items() if n.final]
        graph = automaton.to_graph()
        _, shortest_path = nx.multi_source_dijkstra(graph, sources=final_nodes, target=0, weight=lambda *args: 1)
        shortest_sequence = [graph.edges[e]["label"] for e in itertools.pairwise(shortest_path)][::-1]

        for i, event in enumerate(shortest_sequence[:-1]):
            rm.add_transition(f"u{i}", f"u{i+1}", event)
        rm.add_transition(f"u{len(shortest_sequence)-1}", "u_acc", shortest_sequence[-1])

        return rm

