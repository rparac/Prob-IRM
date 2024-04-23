from typing import Optional, Type

from ..algo import QRM
from ._base import Agent
from ..algo import Algo


# TODO: We can probably just have this class set up a default_rm_agent; removes duplication while making clear we are
#  not using an RM
class NoRMAgent(Agent):
    def __init__(
            self, agent_id: str, algo: Algo,
    ):
        super().__init__(agent_id)

        self.algo = algo
        self.reset()

    def reset(self, seed: Optional[int] = None):
        self.u = 0
        self.algo.on_env_reset()

    def action(self, state, greedy: bool = False, testing: bool = False):
        return self.algo.action(state, self.u, greedy=greedy, testing=testing)

    def learn(self, state, u, action, reward, done, next_state, next_u, **kwargs):
        return self.algo.learn(state, u, action, reward, done, next_state, next_u)

    def update_agent(
            self, state, action, reward, terminated, truncated, is_positive_trace, next_state, labels, learning=True,
            **kwargs
    ):
        loss = None
        next_u = 0

        if learning:
            loss = self.learn(
                state, self.u, action, reward, terminated or truncated, next_state, next_u
            )

        self.u = next_u
        return loss, set(), None

    def project_labels(self, labels):
        return tuple(labels)

    def set_log_folder(self, folder):
        super().set_log_folder(folder)
        self.algo.set_save_path(folder)
