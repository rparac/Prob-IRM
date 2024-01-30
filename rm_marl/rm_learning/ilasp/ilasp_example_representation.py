import abc
import enum
from typing import Optional, List, Set, Type, Dict


class ILASPTerm(abc.ABC):
    @abc.abstractmethod
    def as_term_str(self) -> str:
        raise NotImplementedError("Should not use ILASPTerm directly")


class ILASPPredicate(abc.ABC):
    @abc.abstractmethod
    def as_predicate_str(self) -> str:
        raise NotImplementedError("Should not use ILASPPredicateDirectly")


class LastPredicate(ILASPPredicate):
    def __init__(self, time_step: int):
        self.time_step = time_step

    def as_predicate_str(self):
        return f'last({self.time_step}).'

    def __eq__(self, other):
        return isinstance(other, LastPredicate) and self.time_step == other.time_step


class ObservablePredicate(ILASPPredicate):
    def __init__(self, label: str, time_step: int):
        self.label = label
        self.time_step = time_step

    def as_predicate_str(self):
        return f'obs("{self.label}", {self.time_step}).'

    def __eq__(self, other):
        if not isinstance(other, ObservablePredicate):
            return False
        # Intentionally defined for label only
        return self.label == other.label

    def __hash__(self):
        return hash(self.label)


class ISAILASPExample:
    class ExType(enum.Enum):
        GOAL = enum.auto()
        DEND = enum.auto()
        INCOMPLETE = enum.auto()

    ex_id: str
    penalty: Optional[int]
    example_type: ExType
    observable_context: List[ObservablePredicate]
    last_predicate: LastPredicate
    is_positive: bool

    def __init__(self, ex_id: str, ex_penalty: Optional[int], example_type: ExType,
                 observable_context: List[ObservablePredicate],
                 last_predicate: LastPredicate, is_positive: bool = True):
        self.ex_id = ex_id
        self.penalty = ex_penalty
        self.example_type = example_type
        self.observable_context = observable_context
        self.last_predicate = last_predicate
        self.is_positive = is_positive

    def __eq__(self, other):
        if not isinstance(other, ISAILASPExample):
            return False

        return (self.example_type == other.example_type and
                self.observable_context == other.observable_context and
                self.last_predicate == other.last_predicate)

    def __hash__(self):
        return sum(hash(x) for x in self.observable_context)

    def _generate_inc_exc_str(self):
        if self.example_type == ISAILASPExample.ExType.GOAL:
            inc_str = "accept"
        elif self.example_type == ISAILASPExample.ExType.DEND:
            inc_str = "reject"
        else:
            inc_str = ""

        if self.example_type == ISAILASPExample.ExType.GOAL:
            exc_str = "reject"
        elif self.example_type == ISAILASPExample.ExType.DEND:
            exc_str = "accept"
        else:
            exc_str = "accept, reject"

        return f"{{{inc_str}}}, {{{exc_str}}},"

    def __repr__(self):
        context_str = '  '.join([elem.as_predicate_str() for elem in self.observable_context])
        context_str += f'\n  {self.last_predicate.as_predicate_str()}'
        prefix = "pos" if self.is_positive else "neg"
        return f"#{prefix}({self.ex_id}{'' if self.penalty is None else f'@{self.penalty}'}, " \
               f"{self._generate_inc_exc_str()}" \
               f"{{\n" \
               f"  {context_str}\n" \
               "}).\n"

    def compact_observations(self):
        # Group observations by time-step
        time_obs_dict = {}
        for elem in self.observable_context:
            elems = time_obs_dict.get(elem.time_step, set())
            elems.add(elem)
            time_obs_dict[elem.time_step] = elems
        self.observable_context = []

        # Merge identical observations
        sol = []
        time_steps = sorted(time_obs_dict.keys())
        curr_observations = time_obs_dict[time_steps[0]]
        for t in time_steps[1:]:
            if time_obs_dict[t] != curr_observations:
                sol.append(curr_observations)
                curr_observations = time_obs_dict[t]
        sol.append(curr_observations)

        # Add correct time steps
        ret = []
        for t, observations in enumerate(sol):
            for observation in observations:
                observation.time_step = t
                self.observable_context.append(observation)

        # Add Last Predicate
        last_timestep = len(sol) - 1
        self.last_predicate = LastPredicate(last_timestep)
        return ret, last_timestep

    def generate_incomplete_examples(self) -> List['ISAILASPExample']:
        sols = []
        for i in range(1, len(self.observable_context)):
            ex_id = f"{self.ex_id}_inc_{i}"
            sols.append(
                ISAILASPExample(ex_id, self.penalty, ISAILASPExample.ExType.INCOMPLETE, self.observable_context[:i],
                                LastPredicate(i))
            )
        return sols

    @staticmethod
    def add_example(curr_examples, new_example):
        ex_updated = False
        for ex in curr_examples:
            if ex == new_example:
                ex_updated = True
                ex.penalty += new_example.penalty
        if not ex_updated:
            curr_examples.append(new_example)
        return curr_examples


class ISAExampleContainer:
    def __init__(self):
        self._storage: Dict[ISAILASPExample, int] = {}

    def add(self, ex):
        if ex in self._storage:
            curr_pen = self._storage[ex]
            ex.penalty += curr_pen
        self._storage[ex] = ex.penalty

    def as_list(self):
        return list(self._storage.keys())


# Lifts the example representation from Daniel's work to ISAILASPExample
def _lift(example: List[List[str]], ex_id: str, ex_type: ISAILASPExample.ExType) -> ISAILASPExample:
    obs_context = []
    for i in range(0, len(example)):
        for symbol in example[i]:
            obs_context.append(ObservablePredicate(symbol, i))
    last = LastPredicate(len(example) - 1)
    penalty = None
    return ISAILASPExample(ex_id, penalty, ex_type, obs_context, last)


def lift_goal_example(example: List[List[str]], ex_id: str) -> ISAILASPExample:
    return _lift(example, ex_id, ex_type=ISAILASPExample.ExType.GOAL)


def lift_dend_example(example: List[List[str]], ex_id: str) -> ISAILASPExample:
    return _lift(example, ex_id, ex_type=ISAILASPExample.ExType.DEND)


def lift_inc_example(example: List[List[str]], ex_id: str) -> ISAILASPExample:
    return _lift(example, ex_id, ex_type=ISAILASPExample.ExType.INCOMPLETE)
