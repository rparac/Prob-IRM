from typing import TYPE_CHECKING, Optional, Callable

from ..algo import QRM
from ..reward_machine import RewardMachine
from ..rm_learning import ILASPLearner, DAFSALearner, S2SLearner
from ..rm_transition.rm_transitioner import RMTransitioner
from ..utils.logging import getLogger
from .rm_agent import RewardMachineAgent
from rm_marl.rm_learning.trace_tracker import TraceTracker, NoisyTraceTracker

if TYPE_CHECKING:
    from ..algo import Algo
    from ..rm_learning import RMLearner

LOGGER = getLogger(__name__)


class RewardMachineLearningAgent(RewardMachineAgent):
    def __init__(
            self,
            agent_id: str,
            rm_transitioner: RMTransitioner,
            algo_cls: "Algo" = QRM,
            algo_kws: dict = None,
            rm_learner_cls: "RMLearner" = ILASPLearner,
            rm_learner_kws: dict = None,
    ):
        rm_learner_kws = rm_learner_kws or {}
        self.rm_learner = rm_learner_cls(agent_id, **rm_learner_kws)
        # self.trace = TraceTracker()
        self.trace = NoisyTraceTracker()
        # self.incomplete_examples = []
        self.incomplete_examples = {}
        # self.positive_examples = []
        self.positive_examples = {}
        # self.negative_examples = []
        self.dend_examples = {}

        rm_transitioner.rm = self._default_rm()

        super().__init__(agent_id, rm_transitioner, algo_cls, algo_kws)

    def set_log_folder(self, folder):
        super().set_log_folder(folder)
        self.rm_learner.set_log_folder(self.log_folder)

    @staticmethod
    def _default_rm():
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
            self.trace.update(labels)

            if terminated or truncated:
                candidate_rm = self.rm_learner.update_rm(self.observables, self.rm, self.trace)

        return loss, interrupt, rm_updated

    # TODO: delete
    def old_update_agent(
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
            self.trace.update(labels, next_state)
            if isinstance(self.rm_learner, (ILASPLearner, DAFSALearner, S2SLearner)):
                seq = self.trace.no_dups_labels_sequence
            # elif isinstance(self.rm_learner, (DAFSALearner,)):
            #     seq = self.trace.no_dups_flatten_labels_sequence
            else:
                seq = self.trace.flatten_labels_sequence
            if terminated or truncated:
                examples_updated = self._update_examples(
                    seq, terminated, is_positive_trace
                )
                if examples_updated and self._should_relearn_rm(terminated, is_positive_trace):
                    candidate_rm = self.rm_learner.learn(
                        self.observables,
                        self.rm,
                        self.positive_examples,
                        self.dend_examples,
                        self.incomplete_examples,
                    )
                    if candidate_rm:
                        self.rm = candidate_rm
                        self.algo.reset()
                        rm_updated = True
            # TODO: discuss with Leo why we need this part
            # elif self.rm.is_state_terminal(self.u):
            #     LOGGER.debug(
            #         f"[{self.agent_id}] the RM {self.rm_learner.rm_learning_counter} is wrong "
            #         f"or the state belief is wrong.")
            #     examples_updated = self._update_examples(seq, complete=False, positive=False)
            #     if examples_updated:
            #         candidate_rm = self.rm_learner.learn(
            #             self.observables,
            #             self.rm,
            #             self.positive_examples,
            #             self.dend_examples,
            #             self.incomplete_examples
            #         )
            #         if candidate_rm:
            #             self.rm = candidate_rm
            #             self.algo.reset()
            #             rm_updated = True
            #         interrupt = True

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

    def _update_examples(self, trace: tuple, complete: bool, positive: bool):
        if not trace:
            return False

        if complete:
            if positive:
                self.positive_examples[trace] = None
            else:
                self.dend_examples[trace] = None
            for i in range(1, len(trace)):
                self.incomplete_examples.append(trace[:i])
        else:
            self.incomplete_examples.append(trace)

        self.incomplete_examples = [
            e for e in self.incomplete_examples
            if e not in self.positive_examples
        ]

        return True

    # RM should be relearned if there is a mismatch between
    #  environment and the current RM
    def _should_relearn_rm(self, terminated: bool, is_positive: bool):
        if not terminated:
            # TODO: check if this is fine
            # We can't determine if there is a mismatch
            return False

        if is_positive:
            return not self.rm.is_accepting_state(self.u)
        return not self.rm.is_rejecting_state(self.u)
