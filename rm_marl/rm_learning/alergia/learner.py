import itertools
import os
from collections import deque

import networkx as nx
from aalpy.learning_algs import run_Alergia
from aalpy.utils.FileHandler import save_automaton_to_file
from networkx.algorithms.shortest_paths.generic import \
    _build_paths_from_predecessors

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

class AlergiaLearner(RMLearner):

    def __init__(self, agent_id, eps=0.25, transition_probability_threshold=0.2, learn_frequency=10):
        super().__init__(agent_id)

        self.eps = eps
        self.transition_probability_threshold = transition_probability_threshold
        self.learn_frequency = learn_frequency

    def learn(self, _observables, rm, positive_examples, _negative_examples, incomplete_examples):

        if not positive_examples:
            LOGGER.debug(f"[{self.agent_id}] No positive examples")
            return

        self.rm_learning_counter += 1

        if self.rm_learning_counter % self.learn_frequency != 0:
            return

        LOGGER.debug(f"[{self.agent_id}] Using {len(positive_examples)} examples")
        automaton = run_Alergia(positive_examples, automaton_type='mc', eps=self.eps, optimize_for='accuracy', print_info=False)
        candidate_rm = self._generate_rm(automaton)

        if candidate_rm.states:
            try:
                candidate_rm.set_u0("u0")
                candidate_rm.set_uacc("u_acc")
            except Exception as e:
                rm_plot_filename = os.path.join(
                    self.log_folder, f"plot_{self.rm_learning_counter}"
                )
                candidate_rm.plot(rm_plot_filename)
                save_automaton_to_file(automaton, rm_plot_filename + "_proba", "pdf")
                raise e

            if candidate_rm != rm:
                LOGGER.debug(f"[{self.agent_id}] New RM found.")
                rm_plot_filename = os.path.join(
                    self.log_folder, f"plot_{self.rm_learning_counter}"
                )
                candidate_rm.plot(rm_plot_filename)
                save_automaton_to_file(automaton, rm_plot_filename + "_proba", "pdf")
                return candidate_rm

    def _generate_rm(self, automaton):
        rm = RewardMachine()

        state_names = {s.state_id: f"u{i}" if s.transitions else "u_acc" for i, s in enumerate(automaton.states, start=1)}

        g = nx.DiGraph()
        for state in automaton.states:
            for new_state, prob in state.transitions:
                g.add_edge(state_names[state.state_id], state_names[new_state.state_id], label=new_state.output, proba=prob)
        g.add_edge("u0", state_names[automaton.initial_state.state_id], label=automaton.initial_state.output, proba=1.0)

        _, shortest_path = _most_probable_route(g.reverse(), source="u_acc", target="u0", weight="proba")
        # nx.multi_source_dijkstra(g.reverse(), sources=["u_acc"], target="u0", weight=f)
        sg = g.edge_subgraph((
            e for e, d in g.edges.items() 
            if e in pairwise(shortest_path[::-1]) or
            d["proba"] > self.transition_probability_threshold
        ))
        sg = sg.subgraph({"u0"}.union(nx.descendants(sg, "u0")))

        for (f, t), d in sg.edges.items():
            rm.add_transition(f, t, d["label"])

        return rm

def _most_probable_route(G, source="u_acc", target="u0", weight="proba"):
    
    weight_func = lambda u, v, e: e[weight]
    combine_dist_func = lambda du, w: du * w

    paths = {source: [source]}  # dictionary of paths
    dist = {v: -1 for v in [source]}
    dist = _bellman_ford_modified(G, [source], weight_func, paths=paths, dist=dist, target=target)
    return (-dist[target], paths[target])


def _bellman_ford_modified(
    G, source, weight, pred=None, paths=None, dist=None, target=None, heuristic=True, combine_dist_func=None
):
    """Relaxation loop for Bellman–Ford algorithm.

    This is an implementation of the SPFA variant.
    See https://en.wikipedia.org/wiki/Shortest_Path_Faster_Algorithm

    Parameters
    ----------
    G : NetworkX graph

    source: list
        List of source nodes. The shortest path from any of the source
        nodes will be found if multiple sources are provided.

    weight : function
        The weight of an edge is the value returned by the function. The
        function must accept exactly three positional arguments: the two
        endpoints of an edge and the dictionary of edge attributes for
        that edge. The function must return a number.

    pred: dict of lists, optional (default=None)
        dict to store a list of predecessors keyed by that node
        If None, predecessors are not stored

    paths: dict, optional (default=None)
        dict to store the path list from source to each node, keyed by node
        If None, paths are not stored

    dist: dict, optional (default=None)
        dict to store distance from source to the keyed node
        If None, returned dist dict contents default to 0 for every node in the
        source list

    target: node label, optional
        Ending node for path. Path lengths to other destinations may (and
        probably will) be incorrect.

    heuristic : bool
        Determines whether to use a heuristic to early detect negative
        cycles at a hopefully negligible cost.

    Returns
    -------
    Returns a dict keyed by node to the distance from the source.
    Dicts for paths and pred are in the mutated input dicts by those names.

    Raises
    ------
    NodeNotFound
        If any of `source` is not in `G`.

    NetworkXUnbounded
        If the (di)graph contains a negative cost (di)cycle, the
        algorithm raises an exception to indicate the presence of the
        negative cost (di)cycle.  Note: any negative weight edge in an
        undirected graph is a negative cost cycle
    """
    for s in source:
        if s not in G:
            raise ValueError(f"Source {s} not in G")

    if pred is None:
        pred = {v: [] for v in source}

    if dist is None:
        dist = {v: 0 for v in source}

    if combine_dist_func is None:
        combine_dist_func = lambda du, w: du + w

    # Heuristic Storage setup. Note: use None because nodes cannot be None
    nonexistent_edge = (None, None)
    pred_edge = {v: None for v in source}
    recent_update = {v: nonexistent_edge for v in source}

    G_succ = G.succ if G.is_directed() else G.adj
    inf = float("inf")
    n = len(G)

    count = {}
    q = deque(source)
    in_q = set(source)
    while q:
        u = q.popleft()
        in_q.remove(u)

        # Skip relaxations if any of the predecessors of u is in the queue.
        if all(pred_u not in in_q for pred_u in pred[u]):
            dist_u = dist[u]
            for v, e in G_succ[u].items():
                dist_v = combine_dist_func(dist_u, weight(u, v, e))

                if dist_v < dist.get(v, inf):
                    # In this conditional branch we are updating the path with v.
                    # If it happens that some earlier update also added node v
                    # that implies the existence of a negative cycle since
                    # after the update node v would lie on the update path twice.
                    # The update path is stored up to one of the source nodes,
                    # therefore u is always in the dict recent_update
                    if heuristic:
                        if v in recent_update[u]:
                            raise ValueError("Negative cost cycle detected.")
                        # Transfer the recent update info from u to v if the
                        # same source node is the head of the update path.
                        # If the source node is responsible for the cost update,
                        # then clear the history and use it instead.
                        if v in pred_edge and pred_edge[v] == u:
                            recent_update[v] = recent_update[u]
                        else:
                            recent_update[v] = (u, v)

                    if v not in in_q:
                        q.append(v)
                        in_q.add(v)
                        count_v = count.get(v, 0) + 1
                        if count_v == n:
                            raise ValueError("Negative cost cycle detected.")
                        count[v] = count_v
                    dist[v] = dist_v
                    pred[v] = [u]
                    pred_edge[v] = u

                elif dist.get(v) is not None and dist_v == dist.get(v):
                    pred[v].append(u)

    if paths is not None:
        sources = set(source)
        dsts = [target] if target is not None else pred
        for dst in dsts:
            gen = _build_paths_from_predecessors(sources, dst, pred)
            paths[dst] = next(gen)

    return dist