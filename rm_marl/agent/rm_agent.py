from typing import Optional, Type

import numpy as np

from ._base import Agent
from ..algo import QRM, Algo
from ..reward_machine import RewardMachine
from ..rm_transition.rm_transitioner import RMTransitioner


class RewardMachineAgent(Agent):
    def __init__(
            self, agent_id: str, rm_transitioner: RMTransitioner, algo: Algo
    ):
        super().__init__(agent_id)
        self.rm_transitioner = rm_transitioner

        self.algo = algo
        self.reset()

    @classmethod
    def default_rm_agent(cls, agent_id, rm_transitioner: RMTransitioner, algo: Algo):
        rm = RewardMachineAgent.default_rm()
        rm_transitioner.rm = rm
        return cls(agent_id, rm_transitioner, algo)

    @staticmethod
    def default_rm():
        rm = RewardMachine()
        rm.add_states(["u0"])
        rm.set_u0("u0")
        return rm

    def reset(self, seed: Optional[int] = None, **kwargs):
        self.u = self.rm_transitioner.get_initial_state()
        self.algo.on_env_reset()

    def action(self, state, greedy: bool = False, **algo_args):
        return self.algo.action(state, self.u, greedy=greedy, **algo_args)

    def get_current_state(self, **kwargs):
        return self.u

    def learn(self, state, u, action, reward, done, next_state, next_u, **kwargs):
        return self.algo.learn(state, u, action, reward, done, next_state, next_u)

    def update_agent(
            self, state, action, reward, terminated, truncated, is_positive_trace, next_state, labels, learning=True,
            **kwargs
    ):
        loss = None
        next_u = self.rm_transitioner.get_next_state(self.u, labels)

        if learning:
            loss = self.learn(
                state, self.u, action, reward, terminated or truncated, next_state, next_u
            )

        self.u = next_u
        return loss, set(), None

    def set_log_folder(self, folder):
        super().set_log_folder(folder)
        self.algo.set_save_path(folder)

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

    def get_statistics(self):
        return {}