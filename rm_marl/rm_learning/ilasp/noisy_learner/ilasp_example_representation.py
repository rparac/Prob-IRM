import abc
from typing import Optional, List, Set, Type


class ILASPTerm(abc.ABC):
    @abc.abstractmethod
    def as_term_str(self) -> str:
        raise NotImplementedError("Should not use ILASPTerm directly")


class RejectTerm(ILASPTerm):
    def as_term_str(self) -> str:
        return "reject"


class AcceptTerm(ILASPTerm):
    def as_term_str(self) -> str:
        return "accept"


class ILASPPredicate(abc.ABC):
    @abc.abstractmethod
    def as_predicate_str(self) -> str:
        raise NotImplementedError("Should not use ILASPPredicateDirectly")


class LastPredicate(ILASPPredicate):
    def __init__(self, time_step: int):
        self.time_step = time_step

    def as_predicate_str(self):
        return f'last({self.time_step}).'


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


class ISAILASPExample:
    ex_id: str
    penalty: Optional[int]
    inclusion: Set[ILASPTerm]
    exclusion: Set[ILASPTerm]
    observable_context: Set[ObservablePredicate]
    last_predicate: LastPredicate
    is_positive: bool

    def __init__(self, ex_id: str, ex_penalty: Optional[int], inclusion: Set[ILASPTerm],
                 exclusion: Set[ILASPTerm], observable_context: Set[ObservablePredicate],
                 last_predicate: LastPredicate, is_positive: bool = True):
        self.ex_id = ex_id
        self.penalty = ex_penalty
        self.inclusion = inclusion
        self.exclusion = exclusion
        self.observable_context = observable_context
        self.last_predicate = last_predicate
        self.is_positive = is_positive

    def __eq__(self, other):
        if not isinstance(other, ISAILASPExample):
            return False

        return (self.inclusion == other.inclusion and
                self.exclusion == other.exclusion and
                self.observable_context == other.observable_context and
                self.last_predicate == other.last_predicate)

    def __repr__(self):
        context_str = '\n'.join([elem.as_predicate_str() for elem in self.observable_context])
        context_str += f'\n{self.last_predicate}'
        prefix = "pos" if self.is_positive else "neg"
        return f"#{prefix}({self.ex_id}@{self.penalty},\n" \
               f"{{ {','.join([e.as_term_str() for e in self.inclusion])} }},\n" \
               f"{{ {','.join([e.as_term_str() for e in self.exclusion])} }}, \n" \
               f"{{\n" \
               f" {context_str}" \
               "}).\n"

    def compact_observations(self):
        # Group observations by time-step
        time_obs_dict = {}
        other_predicates = []
        for elem in self.observable_context:
            elems = time_obs_dict.get(elem.time_step, set())
            elems.add(elem)
            time_obs_dict[elem.time_step] = elems
        self.observable_context = set(other_predicates)

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
                self.observable_context.add(observation)

        # Add Last Predicate
        last_timestep = len(sol) - 1
        self.last_predicate = LastPredicate(last_timestep)
        return ret, last_timestep

    def generate_incomplete_examples(self) -> List['ISAILASPExample']:
        sols = []
        for i in range(1, len(self.observable_context)):
            ex_id = f"{self.ex_id}_inc_{i}"
            sols.append(
                ISAILASPExample(ex_id, self.penalty, set(), {AcceptTerm(), RejectTerm()}, self.observable_context[:i],
                                LastPredicate(i))
            )
        return sols
