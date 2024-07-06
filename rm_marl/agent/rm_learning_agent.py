from typing import Callable, List, Union
from typing import Optional, Type

from rm_marl.rm_learning.trace_tracker import TraceTracker
from ._base import Agent
from .rm_agent import RewardMachineAgent
from ..algo import Algo
from ..algo import QRM
from ..reward_machine import RewardMachine
from ..rm_learning import ILASPLearner
from ..rm_learning import RMLearner
from ..rm_transition.rm_transitioner import RMTransitioner
from ..utils.logging import getLogger

LOGGER = getLogger(__name__)


class RewardMachineLearningAgent(Agent):
    def __init__(
            self,
            rm_agent: Union[RewardMachineAgent, List[RewardMachineAgent]],
            rm_learner_cls: Type[RMLearner] = ILASPLearner,
            rm_learner_kws: dict = None,
    ):
        # self.incomplete_examples = []
        self.incomplete_examples = {}
        # self.positive_examples = []
        self.positive_examples = {}
        # self.negative_examples = []
        self.dend_examples = {}

        # Used joint agent name
        if isinstance(rm_agent, list):
            self.rm_agents = {ag.agent_id: ag for ag in rm_agent}
        else:
            self.rm_agents = {rm_agent.agent_id: rm_agent}

        self.rm = RewardMachineAgent.default_rm()
        for agent in self.rm_agents.values():
            agent.rm = self.rm
        agent_id = "_".join(self.rm_agents.keys())

        self.traces = {aid: TraceTracker() for aid in self.rm_agents.keys()}

        rm_learner_kws = rm_learner_kws or {}
        self.rm_learner = rm_learner_cls(agent_id, **rm_learner_kws)
        super().__init__(agent_id)

    def set_log_folder(self, folder):
        super().set_log_folder(folder)
        self.rm_learner.set_log_folder(self.log_folder)
        for agent in self.rm_agents.values():
            agent.set_log_folder(folder)

    @property
    def observables(self):
        union = set((l for e in self.incomplete_examples for ls in e for l in ls)).union(
            set((l for e in self.positive_examples for ls in e for l in ls))).union(
            set((l for e in self.dend_examples for ls in e for l in ls))
        )
        return union

    def reset(self, seed: Optional[int] = None, agent_id: str = None):
        self.traces[agent_id].reset()
        self.rm_agents[agent_id].reset(seed)

    def action(self, state, greedy: bool = False, agent_id=None, **algo_args):
        rm_agent = self.rm_agents[agent_id]
        return rm_agent.action(state, greedy, **algo_args)

    def get_current_state(self, agent_id=None):
        rm_agent = self.rm_agents[agent_id]
        return rm_agent.get_current_state()

    def learn(self, state, u, action, reward, done, next_state, next_u, agent_id=None):
        rm_agent = self.rm_agents[agent_id]
        return rm_agent.learn(state, u, action, reward, done, next_state, next_u)

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
            agent_id=None,
    ):

        rm_agent = self.rm_agents[agent_id]
        loss, agents_to_interrupt, updated_rm = rm_agent.update_agent(
            state, action, reward, terminated, truncated, is_positive_trace, next_state, labels, learning
        )

        if learning:
            self.traces[agent_id].update(labels, is_positive_trace, terminated)

            # TODO: we may need to rethink is_state_terminal
            if terminated or truncated or self.rm.is_state_terminal(rm_agent.u):

                if not (terminated or truncated):
                    LOGGER.debug(f"[{self.agent_id}] the RM {self.rm_learner.rm_learning_counter} is wrong.")
                    agents_to_interrupt = {rm_agent}

                candidate_rm = self.rm_learner.update_rm(self.rm, rm_agent.u, self.traces[agent_id], terminated,
                                                         truncated,
                                                         is_positive_trace)
                if candidate_rm:
                    self.rm = candidate_rm
                    for agent in self.rm_agents.values():
                        agent.rm = candidate_rm
                        agent.algo.on_rm_reset(self.rm)
                    updated_rm = candidate_rm
                    # We can always interrupt if a new rm is learned
                    # TODO: check if we need the interrupt variable; looks like it is fully captured by rm_updated
                    agents_to_interrupt = set(self.rm_agents.keys())

        return loss, agents_to_interrupt, updated_rm

    def project_labels(self, labels):
        if isinstance(labels, list):
            return tuple(labels)
        return labels

    def get_statistics(self):
        return self.rm_learner.get_statistics()
