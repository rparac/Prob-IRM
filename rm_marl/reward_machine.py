from collections import defaultdict
import copy


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

    def __eq__(self, __o: object) -> bool:
        return (
            set(self.states) == set(__o.states)
            and self.events == __o.events
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

    def set_urej(self, state) -> None:
        assert state in self.states, f"{state} is unknown"
        self.urej = state

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

                if to_state == self.uacc:
                    to_state = f"{to_state} (uacc)"
                elif to_state == self.urej:
                    to_state = f"{to_state} (urej)"

                dot.edge(str(from_state), str(to_state), str(condition))

        dot.render(file_name)

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
        if u1 not in (self.uacc, self.urej):
            if u2 == self.uacc:
                return 1
            elif u2 == self.urej:
                return -1
        return 0

    def get_valid_events(self, u1=None):
        if u1 is None:
            return tuple({l for v in self.transitions.values() for e in v.keys() for l in e})
        return tuple(l for e in self.transitions[u1].keys() for l in e)

    def is_state_terminal(self, u):
        return u in (self.uacc, self.urej)

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

            
