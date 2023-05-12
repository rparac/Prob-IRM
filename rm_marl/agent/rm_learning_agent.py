from itertools import groupby
from typing import TYPE_CHECKING, Optional
from collections import OrderedDict

from ..algo import QRM
from ..reward_machine import RewardMachine
from ..rm_learning import ILASPLearner, DAFSALearner, AlergiaLearner # , S2SLearner
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
        self.obs = []

        self._hash_state_mapping = OrderedDict()

    def reset(self):
        self.trace.clear()
        self.obs.clear()

    def update(self, labels, obs):
        self.trace.append(self._process_label(labels))
        self.obs.append(self._process_obs(obs))

    def _process_label(self, labels):
        # TODO check or remove that
        # assert len(labels) < 2, f"Assumption that there is only one label at a time: [{labels}]"
        return labels or []

    def _process_obs(self, obs):
        state_hash = hash(str(obs))
        if state_hash not in self._hash_state_mapping: 
            self._hash_state_mapping[state_hash] = obs
        return list(self._hash_state_mapping.keys()).index(state_hash) + 1

    @property
    def labels_sequence(self):
        return tuple(tuple(es) for es in self.trace if es)

    @property
    def no_dups_labels_sequence(self):
        return tuple(i[0] for i in groupby(self.labels_sequence or tuple()))

    @property
    def flatten_labels_sequence(self):
        return tuple(e for es in self.trace for e in es)

    @property
    def no_dups_flatten_labels_sequence(self):
        return tuple(i[0] for i in groupby(self.flatten_labels_sequence or tuple()))

    @property
    def sequence(self):
        return tuple((l[0], o) for l, o in zip(self.trace, self.obs) if l)


class RewardMachineLearningAgent(RewardMachineAgent):
    def __init__(
        self,
        agent_id: str,
        algo_cls: "Algo" = QRM,
        algo_kws: dict = None,
        rm_learner_cls: "RMLearner" = ILASPLearner,
        rm_learner_kws: dict = None,
    ):
        rm_learner_kws = rm_learner_kws or {}
        self.rm_learner = rm_learner_cls(agent_id, **rm_learner_kws)
        self.trace = TraceTracker()
        self.incomplete_examples = []
        self.positive_examples = []
        self.negative_examples = []

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
        union = set((l for e in self.incomplete_examples for ls in e for l in ls)).union(
            set((l for e in self.positive_examples for ls in e for l in ls))).union(
            set((l for e in self.negative_examples for ls in e for l in ls))
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
        next_state,
        labels,
        learning=True,
    ):
        loss, interrupt = super().update_agent(
            state, action, reward, terminated, truncated, next_state, labels, learning
        )

        if learning:
            self.trace.update(labels, next_state)
            if isinstance(self.rm_learner, (ILASPLearner, DAFSALearner)):#, S2SLearner)):
                seq = self.trace.no_dups_labels_sequence
            # elif isinstance(self.rm_learner, (DAFSALearner,)):
            #     seq = self.trace.no_dups_flatten_labels_sequence
            else:
                seq = self.trace.flatten_labels_sequence
            if terminated or truncated:
                examples_updated = self._update_examples(
                    seq, terminated
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
            elif self.rm.is_state_terminal(self.u):
                LOGGER.debug(f"[{self.agent_id}] the RM {self.rm_learner.rm_learning_counter} is wrong.")
                examples_updated = self._update_examples(seq, False)
                if examples_updated:
                    candidate_rm = self.rm_learner.learn(
                        self.observables,
                        self.rm,
                        self.positive_examples,
                        self.negative_examples,
                        self.incomplete_examples
                    )
                    if candidate_rm:
                        self.rm = candidate_rm
                        self.algo.reset()
                    interrupt = True

        return loss, interrupt

    def project_labels(self, labels):
        return tuple(labels)

    def _update_examples(self, trace: tuple, complete: bool):
        if not trace:
            return False

        if complete:
            self.positive_examples.append(trace)
            for i in range(1, len(trace)):
                self.incomplete_examples.append(trace[:i])
        else:
            self.incomplete_examples.append(trace)

        self.incomplete_examples = [
            e for e in self.incomplete_examples 
            if e not in self.positive_examples
        ]

        return True
