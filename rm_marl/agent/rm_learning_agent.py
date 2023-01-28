import os
from itertools import groupby
from typing import TYPE_CHECKING, Optional

from ..algo import QRM
from ..reward_machine import RewardMachine
from ..rm_learning import ILASPLearner, DAFSALearner
from ..utils.logging import getLogger
from .rm_agent import RewardMachineAgent

if TYPE_CHECKING:
    from ..algo import Algo
    from ..reward_machine import RewardMachine
    from ..rm_learning import RMLearner

LOGGER = getLogger(__name__)


class TraceTracker:
    def __init__(self) -> None:
        self.trace = []

    def reset(self):
        self.trace.clear()

    def update(self, labels):
        self.trace.append(labels or [])

    @property
    def flatten_trace(self):
        return tuple(e for es in self.trace for e in es)

    @property
    def nodups_trace(self):
        return tuple(i[0] for i in groupby(self.flatten_trace or []))


class RewardMachineLearningAgent(RewardMachineAgent):
    def __init__(
        self,
        agent_id: str,
        algo_cls: "Algo" = QRM,
        algo_kws: dict = None,
        rm_learner_cls: "RMLearner" = DAFSALearner,
        rm_learner_kws: dict = None,
    ):
        rm_learner_kws = rm_learner_kws or {}
        self.rm_learner = rm_learner_cls(agent_id, **rm_learner_kws)
        self.trace = TraceTracker()
        self.incomplete_examples = set()
        self.positive_examples = set()
        self.negative_examples = set()

        rm = self._default_rm()

        super().__init__(agent_id, rm, algo_cls, algo_kws)

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
        union = self.incomplete_examples.union(self.positive_examples).union(
            self.negative_examples
        )
        return {l for t in union for l in t}

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
        next_state,
        labels,
        learning=True,
    ):
        loss, interrupt = super().update_agent(
            state, action, reward, terminated, truncated, next_state, labels, learning
        )

        if learning:
            self.trace.update(labels)
            if terminated or truncated:
                examples_updated = self._update_examples(
                    self.trace.nodups_trace, terminated
                )
                if examples_updated:
                    candidate_rm = self.rm_learner.learn(
                        self.observables,
                        self.rm,
                        self.positive_examples,
                        self.negative_examples,
                        self.incomplete_examples,
                    )
                    if candidate_rm:
                        self.rm = candidate_rm
                        self.algo.reset()
            # elif self.rm.is_state_terminal(self.u):
            #     LOGGER.debug(f"[{self.agent_id}] the RM {self.rm_learning_counter} is wrong.")
            #     examples_updated = self._update_examples(self.trace.nodups_trace, False)
            #     if examples_updated:
            #         self.rm_num_states = 3
            #         self._update_reward_machine()
            #         interrupt = True

        return loss, interrupt

    def project_labels(self, labels):
        return tuple(labels)

    def _update_examples(self, trace: tuple, complete: bool):
        updated = False

        if complete:
            if trace and trace not in self.positive_examples:
                self.positive_examples.add(trace)
                updated = True
                # for i in range(len(trace) - 1):
                #     pre = trace[: i + 1]
                #     if pre not in self.positive_examples:
                #         self.incomplete_examples.add(pre)
                #     post = trace[-i - 1 :]
                #     if post not in self.positive_examples:
                #         self.incomplete_examples.add(post)
        else:
            if trace and trace not in self.incomplete_examples:
                self.incomplete_examples.add(trace)
                updated = True

        _ = [self.incomplete_examples.discard(e) for e in self.positive_examples]

        return updated
