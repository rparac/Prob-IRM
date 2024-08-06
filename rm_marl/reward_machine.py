from collections import defaultdict
import copy
from typing import Type

import numpy as np


class _CustomDefaultDict(dict):
    def __getitem__(self, __key):
        return super().get(__key, None)


class RewardMachine:

    ACCEPT_CONDITION = "True"
    REJECT_CONDITION = "False"

    def __init__(self) -> None:
        self.states = []
        self.events = set()
        self.transitions = defaultdict(self._transition_constructor)

        # Associates u -> phi(u), which can be used for potential-based reward shaping
        self.state_potentials = {}

        self.u0 = None
        self.uacc = None
        self.urej = None

    @staticmethod
    def _transition_constructor():
        return _CustomDefaultDict()

    def __repr__(self):
        s = "MACHINE:\n"
        s += "u0: {}\n".format(self.u0)
        s += "uacc: {}\n".format(self.uacc)
        if self.urej:
            s += "urej: {}\n".format(self.urej)
        for trans_init_state in self.transitions:
            for event in self.transitions[trans_init_state]:
                trans_end_state = self.transitions[trans_init_state][event]
                s += "({} ---({})---> {})\n".format(
                    trans_init_state, event, trans_end_state
                )
        return s

    # def tranitions_eq(self, __o):
    #     def not_less_tup_eq(from1, from2):
    #         f_from1 = [x for x in from1 if not x.startswith('~')]
    #         f_from2 = [x for x in from2 if not x.startswith('~')]
    #
    #     if self.states != __o.states:
    #         return False
    #
    #     for s1, s2 in zip(self.states, __o.states):
    #         t1 = self.transitions[s1]
    #         t2 = __o.transitions[s2]
    #
    #         for (from1, to1), (from2, to2) in zip(t1.items(), t2.items())
    #
    #             t1.keys()
    #         t2.keys()

    def __eq__(self, __o: object) -> bool:

        if not isinstance(__o, RewardMachine):
            return False

        return (
            set(self.states) == set(__o.states)
            and self.events == __o.events  # - comparing transitions is enough
            and self.u0 == __o.u0
            and self.uacc == __o.uacc
            and self.urej == __o.urej
            and set(self.transitions.keys()) == set(__o.transitions.keys())
            and all(
                set(self.transitions[k].keys()) == set(__o.transitions[k].keys())
                for k in self.transitions.keys()
            )
            and all(
                self.transitions[k1][k2] == __o.transitions[k1][k2]
                for k1 in self.transitions.keys()
                for k2 in self.transitions[k1].keys()
            )
        )

    def set_u0(self, state) -> None:
        assert state in self.states, f"{state} is unknown"
        self.u0 = state

    def set_uacc(self, state) -> None:
        assert state in self.states, f"{state} is unknown"
        self.uacc = state
        self.transitions[state] = self._transition_constructor()

    def set_urej(self, state) -> None:
        assert state in self.states, f"{state} is unknown"
        self.urej = state
        self.transitions[state] = self._transition_constructor()

    def copy(self) -> "RewardMachine":
        return copy.deepcopy(self)

    def plot(self, file_name) -> None:
        from graphviz import Digraph

        dot = Digraph()

        # create Graphviz edges
        edges = [
            (n1, n2, ev)
            for n1 in self.transitions.keys()
            for ev, n2 in self.transitions[n1].items()
        ]
        for from_state, to_state, condition in edges:
            if condition not in (self.ACCEPT_CONDITION, self.REJECT_CONDITION):
                if from_state == self.u0:
                    from_state = f"{from_state} (u0)"
                if to_state == self.u0:
                    to_state = f"{to_state} (u0)"

                if from_state == self.uacc:
                    from_state = f"{from_state} (uacc)"
                elif from_state == self.urej:
                    from_state = f"{from_state} (urej)"
                if to_state == self.uacc:
                    to_state = f"{to_state} (uacc)"
                elif to_state == self.urej:
                    to_state = f"{to_state} (urej)"

                dot.edge(str(from_state), str(to_state), str(condition))

        dot.render(file_name)

    def to_idx(self, state_name: str) -> int:
        try:
            return self.states.index(state_name)
        except ValueError:
            return -1

    def from_idx(self, state_idx: int) -> str:
        return self.states[state_idx]

    def add_states(self, u_list):
        _ = [self.states.append(u) for u in u_list if u not in self.states]

    def add_transition(self, u1, u2, event):
        # Adding machine state
        self.add_states([u1, u2])
        # Adding event
        self.events.add(event)
        # Adding state-transition to delta_u
        if self.transitions[u1][event] and self.transitions[u1][event] != u2:
            raise Exception("Trying to make rm transition function non-deterministic.")
        else:
            self.transitions[u1][event] = u2

    def get_next_state(self, u1, event):
        if not isinstance(event, (list, tuple)):
            event = (event,)

        if u1 in self.transitions:
            for condition in self.transitions[u1]:
                if not isinstance(condition, (list, tuple)):
                    condition = (condition,)
                if all(self._is_event_satisfied(c, event) for c in condition):
                    return self.transitions[u1][condition]
        return u1

    def get_reward(self, u1, u2):
        if isinstance(u1, np.ndarray):
            state_diff = u2 - u1
            # Difference between how much "closer" we are to the accepting state
            #  compared to the rejecting state
            reward = self.accepting_state_prob(state_diff) - self.rejecting_state_prob(state_diff)
            return reward

        if u1 not in (self.uacc, self.urej):
            if u2 == self.uacc:
                return 1
            elif u2 == self.urej:
                return -1
        return 0

    def _comput_state_min_distance_matrix(self):

        distance_matrix = {}
        for u1 in self.states:

            distances = {s: float("inf") for s in self.states}
            distances[u1] = 0

            queue = [(u1, 0)]
            visited = {u1}
            while len(queue) > 0:
                current_state, current_distance = queue.pop()

                successor_states = [self.transitions[current_state][e] for e in self.transitions[current_state].keys()]

                for u2 in [u for u in successor_states if u not in visited and u is not None]:
                    queue.append((u2, current_distance + 1))
                    visited.add(u2)
                    distances[u2] = current_distance + 1

            distance_matrix[u1] = distances

        return distance_matrix

    def compute_state_pontentials(self):

        # An accepting state exist: high-potential = close to accepting state
        if self.uacc is not None:

            # First, we need to compute the distance matrix between each pair of states
            min_distances = self._comput_state_min_distance_matrix()
            self.state_potentials = {u: len(self.states) - min_distances[u][self.uacc] for u in self.states}

        # No accepting state but rejecting state exists: high-potential = far from rejecting state
        elif self.urej is not None:

            min_distances = self._comput_state_min_distance_matrix()
            self.state_potentials = {u: min_distances[u][self.urej] for u in self.states}

        # Neither accepting nor rejecting states exist: use zero potentials everywhere
        else:

            self.state_potentials = {u: 0 for u in self.states}

    def get_valid_events(self, u1=None):
        if u1 is None:
            return tuple({l for v in self.transitions.values() for e in v.keys() for l in e})
        return tuple(l for e in self.transitions[u1].keys() for l in e)

    def is_state_terminal(self, u):
        if isinstance(u, (int, str)):
            return u in (self.uacc, self.urej)

        return self.is_accepting_state(u) or self.is_rejecting_state(u)

    def is_accepting_state(self, u):
        if isinstance(u, (int, str)):
            return u == self.uacc

        acc_idx = self.to_idx(self.uacc)
        curr_state_idx = np.argmax(u)
        return acc_idx == curr_state_idx

    def accepting_state_prob(self, u):
        if isinstance(u, (int, str)):
            return 1 if u == self.uacc else 0

        acc_idx = self.to_idx(self.uacc)
        # Todo: replace this with a different indicator (remove -1)
        return u[acc_idx] if acc_idx != -1 else 0

    def is_rejecting_state(self, u):
        if isinstance(u, (int, str)):
            return u == self.urej

        rej_idx = self.to_idx(self.urej)
        curr_state_idx = np.argmax(u)
        return rej_idx == curr_state_idx

    def rejecting_state_prob(self, u):
        if isinstance(u, (int, str)):
            return 1 if u == self.urej else 0

        rej_idx = self.to_idx(self.urej)
        # Todo: replace this with a different indicator (remove -1)
        return u[rej_idx] if rej_idx != -1 else 0

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

    def is_valid(self):
        queue = set()
        seen = set()
        u = self.u0

        queue.update(self.transitions[u].values())

        if self.uacc in queue or self.urej in queue:
            return True

        while queue:
            u = queue.pop()
            queue.update((n for n in self.transitions[u].values() if n not in seen))
            seen.add(u)

            if self.uacc in queue or self.urej in queue:
                return True

        return False

    @staticmethod
    def load_from_file(file, node_cls=lambda a: a):
        with open(file) as f:
            lines = [l.rstrip() for l in f if l.rstrip() and not l.startswith("#")]

        rm = RewardMachine()

        # adding transitions
        for e in lines[1:]:
            t = eval(e)
            t = (node_cls(t[0]), node_cls(t[1]), tuple(t[2].split(",")))
            rm.add_transition(*t)
            if t[-1] == ("True",):
                rm.set_uacc(t[0])
            elif t[-1] == ("False",):
                rm.set_urej(t[0])

        u0 = node_cls(eval(lines[0]))
        rm.set_u0(u0)
        return rm

    def save_to_file(self, file):
        lines = [f"{self.u0} # initial state\n"]

        for f, et in self.transitions.items():
            for e, t in et.items():
                lines.append(f"({f}, {t}, '{e}')\n")

        lines[-1] = lines[-1].strip("\n")
        with open(file, "w") as f:
            f.writelines(lines)
