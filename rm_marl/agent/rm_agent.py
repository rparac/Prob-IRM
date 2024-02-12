from typing import Optional, Type

from ..algo import QRM
from ._base import Agent
from ..algo import Algo
from ..reward_machine import RewardMachine


class RewardMachineAgent(Agent):
    def __init__(
        self, agent_id: str, rm: "RewardMachine", algo_cls: Type[Algo] = QRM, algo_kws: dict = None
    ):

        super().__init__(agent_id, algo_cls, algo_kws)

        self.rm = rm

        self.reset()

    def reset(self, seed: Optional[int] = None):
        self.u = self.rm.u0

    def action(self, state, greedy: bool = False, **algo_args):
        return self.algo.action(state, self.u, greedy=greedy, **algo_args)

    def learn(self, state, u, action, reward, done, next_state, next_u):
        return self.algo.learn(state, u, action, reward, done, next_state, next_u)

    def update_agent(
        self, state, action, reward, terminated, truncated, next_state, labels, learning=True, **kwargs
    ):
        loss = None
        next_u = self.rm.get_next_state(self.u, labels)

        if learning:
            loss = self.learn(
                state, self.u, action, reward, terminated or truncated, next_state, next_u
            )

        self.u = next_u
        return loss, False, False

    def project_labels(self, labels):
        return tuple(e for e in labels if e in self.rm.get_valid_events())
