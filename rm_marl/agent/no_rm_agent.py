from typing import Optional, Type

from ..algo import QRM
from ._base import Agent
from ..algo import Algo


class NoRMAgent(Agent):
    def __init__(
        self, agent_id: str, algo_cls: Type[Algo] = QRM, algo_kws: dict = None
    ):

        super().__init__(agent_id, algo_cls, algo_kws)

        self.reset()

    def reset(self, seed: Optional[int] = None):
        self.u = 0

    def action(self, state, greedy: bool = False):
        return self.algo.action(state, self.u, greedy=greedy)

    def learn(self, state, u, action, reward, done, next_state, next_u):
        return self.algo.learn(state, u, action, reward, done, next_state, next_u)

    def update_agent(
        self, state, action, reward, terminated, truncated, is_positive_trace, next_state, labels, learning=True, **kwargs
    ):
        loss = None
        next_u = 0

        if learning:
            loss = self.learn(
                state, self.u, action, reward, terminated or truncated, next_state, next_u
            )

        self.u = next_u
        return loss, False, False

    def project_labels(self, labels):
        return tuple(labels)
