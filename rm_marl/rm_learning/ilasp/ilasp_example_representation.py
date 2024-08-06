import abc
import copy
import enum
import itertools
from typing import Optional, List, Set, Type, Dict, Union


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
    # penalty is rounded in this class to avoid dealing with extremely large numbers
    penalty: Optional[float]
    example_type: ExType
    observable_context: List[ObservablePredicate]
    last_predicate: Optional[LastPredicate]
    is_positive: bool
    penalty_threshold: int

    def __init__(self, ex_id: str, ex_penalty: Optional[float], example_type: ExType,
                 observable_context: List[ObservablePredicate],
                 last_predicate: Optional[LastPredicate], is_positive: bool = True,
                 penalty_threshold: int = 1):
        self.ex_id = ex_id
        self.penalty = ex_penalty
        self.example_type = example_type
        self.observable_context = observable_context
        self.last_predicate = last_predicate
        self.is_positive = is_positive

        # Large number to reduce the rounding error to int
        self.penalty_rounding_scale = 1  # 10

        self.penalty_threshold = penalty_threshold

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

    # Example is active if it has a non-zero penalty
    # Example is active if it has a penalty larger than the threshold
    def is_active(self):
        return self.penalty is None or round(self.penalty * self.penalty_rounding_scale) >= self.penalty_threshold

    def __repr__(self):
        context_str = '  '.join([elem.as_predicate_str() for elem in self.observable_context])
        if self.last_predicate:
            context_str += f'\n  {self.last_predicate.as_predicate_str()}'
        prefix = "pos" if self.is_positive else "neg"
        penalty_rounded = round(self.penalty * self.penalty_rounding_scale) if self.penalty else None
        return f"#{prefix}({self.ex_id}{'' if penalty_rounded is None else f'@{penalty_rounded}'}, " \
               f"{self._generate_inc_exc_str()}" \
               f"{{\n" \
               f"  {context_str}\n" \
               "}).\n"

    def compact_observations(self):
        previous_last_predicate = None
        while self.last_predicate != previous_last_predicate:
            previous_last_predicate = self.last_predicate
            # Group observations by time-step
            time_obs_dict = {}
            for elem in self.observable_context:
                elems = time_obs_dict.get(elem.time_step, set())
                elems.add(elem)
                time_obs_dict[elem.time_step] = elems
            self.observable_context = []

            if len(time_obs_dict.keys()) == 0:
                # no observations detected
                self.last_predicate = None
                return

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
            for t, observations in enumerate(sol):
                for observation in observations:
                    observation.time_step = t
                    self.observable_context.append(observation)

            # Add Last Predicate
            last_timestep = len(sol) - 1
            self.last_predicate = LastPredicate(last_timestep)
        previous_last_predicate = None

    def generate_incomplete_examples(self) -> List['ISAILASPExample']:
        sols = []
        for i in range(1, len(self.observable_context)):
            ex_id = f"{self.ex_id}_inc_{i}"
            sols.append(
                ISAILASPExample(ex_id, self.penalty, ISAILASPExample.ExType.INCOMPLETE, self.observable_context[:i],
                                LastPredicate(i - 1), penalty_threshold=self.penalty_threshold)
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
    def __init__(self, ilasp_filter_threshold=None):
        self.storage: Dict[ISAILASPExample, float] = {}

        self._ilasp_filter_threshold = ilasp_filter_threshold

    def add(self, ex):
        self.storage[ex] = self.storage.get(ex, 0) + ex.penalty

        # if ex in self._storage:
        #     curr_pen = self._storage[ex]
        #     del self._storage[ex]
        #     ex.penalty += curr_pen
        # self._storage[ex] = ex.penalty

    def merge(self, ex_container):
        for ex, val in ex_container.storage.items():
            self.storage[ex] = self.storage.get(ex, 0) + val

    # Should be called before a direct call to ISAILASPExample
    def fix_penalties(self):
        for ex, pen in self.storage.items():
            ex.penalty = pen

    # Return examples such that their penalties sum to total_sum
    def as_list_reweighted(self, total_sum: Union[int, float]) -> List[ISAILASPExample]:
        self.fix_penalties()
        ex_pen_sum = sum(self.storage.values())

        # Filter examples that would have a penalty > _ilasp_penalty_threshold after
        # reweighting
        threshold = self._ilasp_filter_threshold * ex_pen_sum / total_sum

        # Need to compute the new penalty sum so the final values sum to a desired number (e.g 100)
        new_ex_penalty_sum = sum(x for x in self.storage.values() if x >= threshold)

        ret = []
        for ex, ex_val in self.storage.items():
            new_ex = copy.deepcopy(ex)
            if new_ex.penalty >= threshold:
                new_ex.penalty = (ex_val / new_ex_penalty_sum) * total_sum
                ret.append(new_ex)
        return ret

    def __len__(self):
        return len(self.storage)


class MultiISAExampleContainer:
    def __init__(self, ilasp_filter_threshold):
        self._ilasp_filter_threshold = ilasp_filter_threshold
        self._goal_examples = ISAExampleContainer(ilasp_filter_threshold)
        self._dend_examples = ISAExampleContainer(ilasp_filter_threshold)
        self._inc_examples = ISAExampleContainer(ilasp_filter_threshold)

    def merge(self, ex_container: ISAExampleContainer, ex_type: ISAILASPExample.ExType):
        if ex_type == ISAILASPExample.ExType.GOAL:
            self._goal_examples.merge(ex_container)
        elif ex_type == ISAILASPExample.ExType.DEND:
            self._dend_examples.merge(ex_container)
        else:
            self._inc_examples.merge(ex_container)

    def get_observables(self):
        out = set()
        for ex in itertools.chain(self._goal_examples.storage.keys(), self._dend_examples.storage.keys(),
                                  self._inc_examples.storage.keys()):
            out = out.union([obs.label for obs in ex.observable_context])

        # Sorting is important for reproducibility (sets are not)
        return list(sorted(out))

    def generate_incomplete_examples(self, only_positive=False):
        _examples = [self._goal_examples] if only_positive else [self._dend_examples,
                                                                 self._inc_examples]
        out = ISAExampleContainer(self._ilasp_filter_threshold)
        for ex_container in _examples:
            ex_container.fix_penalties()
            for base_ex in ex_container.storage.keys():
                for ex in base_ex.generate_incomplete_examples():
                    out.add(ex)
        return out

    def generate_goal_dend_inc(self, total_ex_sum: int) -> \
            (List[ISAILASPExample], List[ISAILASPExample], List[ISAILASPExample]):
        new_inc_pos = self.generate_incomplete_examples(only_positive=True)
        new_inc_rest = self.generate_incomplete_examples(only_positive=False)

        # new_inc = self.generate_incomplete_examples()
        new_inc_rest.merge(self._inc_examples)
        # self.merge(new_inc, ISAILASPExample.ExType.INCOMPLETE)

        gl = self._goal_examples.as_list_reweighted(total_ex_sum)
        de = self._dend_examples.as_list_reweighted(total_ex_sum)
        # inc = new_inc.as_list_reweighted(10 * total_ex_sum)
        inc = new_inc_rest.as_list_reweighted(total_ex_sum)
        pos_inc = new_inc_pos.as_list_reweighted(total_ex_sum)
        return gl, de, inc + pos_inc


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
