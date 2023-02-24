import abc
import os

from ..utils.logging import getLogger

LOGGER = getLogger(__name__)


class RMLearner(metaclass=abc.ABCMeta):

    def __init__(self, agent_id):
        self.agent_id = agent_id
        self._log_folder = None

        self.rm_learning_counter = 0

    @property
    def log_folder(self):
        if self._log_folder is None:
            raise RuntimeError("log_folder should be set")
        return self._log_folder

    def set_log_folder(self, folder):
        if not os.path.exists(folder):
            os.mkdir(folder)
        self._log_folder = folder

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError("learn")

    def process_examples(self, examples):
        return examples



