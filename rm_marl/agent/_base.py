from typing import Type
import abc
import os

from ..algo import Algo
from ..algo import QRM


class Agent(abc.ABC):
    """
    Base class for agents

    This class defines the common interface that any implementation of a RL agent should follow to be
    compatible with the codebase.

    """

    def __init__(
        self,
        agent_id: str,
        algo_cls: Type[Algo] = QRM,
        algo_kws: dict = None
    ):

        self.agent_id = agent_id
        self._log_folder = None

        algo_kws = algo_kws or {}
        self.algo = algo_cls(**algo_kws)

    @property
    def log_folder(self):
        if self._log_folder is None:
            raise RuntimeError("log_folder should be set")
        return self._log_folder

    def set_log_folder(self, folder):

        if not os.path.exists(folder):
            os.mkdir(folder)
        self._log_folder = folder
        self.algo.set_save_path(folder)

    @abc.abstractmethod
    def reset(self, seed):
        raise NotImplementedError('reset')

    @abc.abstractmethod
    def action(self, state, greedy=False):
        raise NotImplementedError('action')

    @abc.abstractmethod
    def learn(self, state, u, action, reward, done, next_state, next_u):
        raise NotImplementedError('learn')

    @abc.abstractmethod
    def update_agent(
            self, state, action, reward, terminated, truncated, is_positive_trace, next_state, labels, learning=True, **kwargs
    ):
        raise NotImplementedError('update_agent')

    @abc.abstractmethod
    def project_labels(self, labels):
        raise NotImplementedError('project_labels')

