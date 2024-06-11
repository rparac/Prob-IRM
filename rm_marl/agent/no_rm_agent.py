from typing import Optional, Type

from . import RewardMachineAgent
from ..algo import QRM
from ._base import Agent
from ..algo import Algo
from ..rm_transition.deterministic_rm_transitioner import DeterministicRMTransitioner


class NoRMAgent(Agent):
    def __init__(
            self, agent_id: str, algo: Algo,
    ):
        super().__init__(agent_id)

        rm_transitioner = DeterministicRMTransitioner(rm=None)
        # Default RM agent is functionally equivalent to the default rm agent (has just one state)
        self.agent = RewardMachineAgent.default_rm_agent(agent_id, rm_transitioner, algo)
        self.agent.reset()

    # Pass every call to the class to the self.agent
    def __getattr__(self, item):
        return getattr(self.agent, item)
