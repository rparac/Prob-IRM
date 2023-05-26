import os
from typing import TYPE_CHECKING, Optional

from ..algo import QRM

if TYPE_CHECKING:
    from ..algo import Algo
    from ..reward_machine import RewardMachine


class RewardMachineAgent:
    def __init__(
        self, agent_id: str, rm: "RewardMachine", algo_cls: "Algo" = QRM, algo_kws: dict = None
    ):
        self.agent_id = agent_id
        algo_kws = algo_kws or {}
        self.algo = algo_cls(**algo_kws)
        self.rm = rm
        self._log_folder = None

        self.reset()

    @property
    def log_folder(self):
        if self._log_folder is None:
            raise RuntimeError("log_folder should be set")
        return self._log_folder

    def set_log_folder(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)
        self._log_folder = folder

    def reset(self, seed: Optional[int] = None):
        self.u = self.rm.u0

    def action(self, state, greedy: bool = False):
        return self.algo.action(state, self.u, greedy=greedy)

    def learn(self, state, u, action, reward, done, next_state, next_u):
        return self.algo.learn(state, u, action, reward, done, next_state, next_u)

    def update_agent(
        self, state, action, reward, terminated, truncated, next_state, labels, learning=True
    ):
        loss = None
        next_u = self.rm.get_next_state(self.u, labels)

        if learning:
            loss = self.learn(
                state, self.u, action, reward, terminated or truncated, next_state, next_u
            )

        self.u = next_u
        return loss, False

    def project_labels(self, labels):
        return tuple(e for e in labels if e in self.rm.get_valid_events())
