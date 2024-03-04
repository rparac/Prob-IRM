from typing import Optional, Type

import numpy as np

from ._base import Agent
from ..algo import QRM, Algo
from ..rm_transition.rm_transitioner import RMTransitioner


class RewardMachineAgent(Agent):
    def __init__(
            self, agent_id: str, rm_transitioner: RMTransitioner, algo_cls: Type[Algo] = QRM, algo_kws: dict = None
    ):
        super().__init__(agent_id, algo_cls, algo_kws)
        self.rm_transitioner = rm_transitioner

        self.algo.reset(rm=self.rm)
        self.reset()

    def reset(self, seed: Optional[int] = None):
        self.u = self.rm_transitioner.get_initial_state()

    def action(self, state, greedy: bool = False, **algo_args):
        return self.algo.action(state, self.u, greedy=greedy, **algo_args)

    def learn(self, state, u, action, reward, done, next_state, next_u):
        return self.algo.learn(state, u, action, reward, done, next_state, next_u)

    def update_agent(
            self, state, action, reward, terminated, truncated, is_positive_trace, next_state, labels, learning=True
    ):
        loss = None
        next_u = self.rm_transitioner.get_next_state(self.u, labels)

        if learning:
            loss = self.learn(
                state, self.u, action, reward, terminated or truncated, next_state, next_u
            )

        self.u = next_u
        return loss, False, False

    # TODO: check if this is this truly necessary given that filter labels exists
    def project_labels(self, labels):
        if isinstance(labels, dict):
            return {e: v for e, v in labels.items() if self.rm.get_valid_events()}

        return tuple(e for e in labels if e in self.rm.get_valid_events())

    @property
    def rm(self):
        return self.rm_transitioner.rm

    @rm.setter
    def rm(self, new_value):
        self.rm_transitioner.rm = new_value
