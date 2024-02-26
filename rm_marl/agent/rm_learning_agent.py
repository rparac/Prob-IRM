from typing import Callable
from typing import Optional, Type

from rm_marl.rm_learning.trace_tracker import TraceTracker
from .rm_agent import RewardMachineAgent
from ..algo import Algo
from ..algo import QRM
from ..reward_machine import RewardMachine
from ..rm_learning import ILASPLearner
from ..rm_learning import RMLearner
from ..rm_transition.rm_transitioner import RMTransitioner
from ..utils.logging import getLogger

LOGGER = getLogger(__name__)


class RewardMachineLearningAgent(RewardMachineAgent):
    def __init__(
            self,
            agent_id: str,
            rm_transitioner: RMTransitioner,
            algo_cls: Type[Algo] = QRM,
            algo_kws: dict = None,
            rm_learner_cls: Type[RMLearner] = ILASPLearner,
            rm_learner_kws: dict = None,
    ):
        rm_learner_kws = rm_learner_kws or {}
        self.rm_learner = rm_learner_cls(agent_id, **rm_learner_kws)
        self.trace = TraceTracker()
        # self.trace = NoisyTraceTracker()
        # self.incomplete_examples = []
        self.incomplete_examples = {}
        # self.positive_examples = []
        self.positive_examples = {}
        # self.negative_examples = []
        self.dend_examples = {}

        rm_transitioner.rm = self.default_rm()

        super().__init__(agent_id, rm_transitioner, algo_cls, algo_kws)

    def set_log_folder(self, folder):
        super().set_log_folder(folder)
        self.rm_learner.set_log_folder(self.log_folder)

    # TODO: Maybe we should move this elsewhere; Useful for wrapping a no-RM agent as well
    @staticmethod
    def default_rm():
        rm = RewardMachine()
        rm.add_states(["u0"])
        rm.set_u0("u0")
        return rm

    @property
    def observables(self):
        union = set((l for e in self.incomplete_examples for ls in e for l in ls)).union(
            set((l for e in self.positive_examples for ls in e for l in ls))).union(
            set((l for e in self.dend_examples for ls in e for l in ls))
        )
        return union

    def reset(self, seed: Optional[int] = None):
        self.trace.reset()
        return super().reset(seed)

    def update_agent(
            self,
            state,
            action,
            reward,
            terminated,
            truncated,
            is_positive_trace,
            next_state,
            labels,
            learning=True,
    ):
        loss, interrupt, rm_updated = super().update_agent(
            state, action, reward, terminated, truncated, is_positive_trace, next_state, labels, learning
        )

        if learning:
            self.trace.update(labels, next_state, is_positive_trace, terminated)

            if terminated or truncated:
                candidate_rm = self.rm_learner.learn(self.rm, self.u, self.trace, terminated, truncated,
                                                     is_positive_trace)
                if candidate_rm:
                    self.rm = candidate_rm
                    self.algo.reset()
                    rm_updated = True
                    # We can always interrupt if a new rm is learned
                    # TODO: check if we need the interrupt variable; looks like it is fully captured by rm_updated
                    interrupt = True

        return loss, interrupt, rm_updated

    def project_labels(self, labels):
        if isinstance(labels, list):
            return tuple(labels)
        return labels

    # Convert [a,a,a,a,b] -> [a,b] & [a,a,a,b,a,a] -> [a,b,a]
    def _deduplicate_list(self, l):
        sol = [l[0]]
        for elem1, elem2 in zip(l, l[1:]):
            if elem1 != elem2:
                sol.append(elem2)
        return sol

    def _deduplicate_trace_w_penalties(self, trace: list, penalties: list, t_norm_fn: Callable[[float, float], float]):
        elems = [trace[0]]
        pens = [penalties[0]]
        curr_pen = penalties[0]

        for elem1, elem2, pen in zip(trace, trace[1:], penalties[1:]):
            curr_pen = t_norm_fn(curr_pen, pen)
            if elem1 != elem2:
                elems.append(elem2)
                pens.append(curr_pen)
        return elems, pens

    def _noisy_update_examples(self, trace: list, penalties: list, complete: bool, positive: bool):
        if not trace:
            return False

        t_trace, red_pens = self._deduplicate_trace_w_penalties(trace, penalties, min)
        t_trace = tuple(t_trace)

        if complete:
            if t_trace in self.incomplete_examples:
                del self.incomplete_examples[t_trace]
            if positive:
                self.positive_examples[t_trace] = red_pens[-1] + self.positive_examples.get(t_trace, 0)
            else:
                self.dend_examples[t_trace] = red_pens[-1] + self.dend_examples.get(t_trace, 0)
            for i in range(1, len(trace) - 1):
                if t_trace[:i] not in self.positive_examples and t_trace[:i] not in self.dend_examples:
                    self.incomplete_examples[t_trace[:i]] = red_pens[i - 1] + self.incomplete_examples.get(t_trace, 0)
        else:
            self.incomplete_examples[t_trace] = red_pens[-1] + self.incomplete_examples.get(t_trace, 0)
