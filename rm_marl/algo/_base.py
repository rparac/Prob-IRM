import abc
from dataclasses import dataclass
from typing import Any, Union

import numpy as np


class Algo(abc.ABC):
    """
    Base class for agent training algorithms

    This class defines the common interface that any implementation of an agent training algorithm
    should follow to be compatible with the codebase.
    """

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        raise NotImplementedError("learn")

    @abc.abstractmethod
    def action(self, *args, **kwargs):
        raise NotImplementedError("action")

    @abc.abstractmethod
    def on_env_reset(self):
        raise NotImplementedError("on_env_reset")

    @abc.abstractmethod
    def on_rm_reset(self, rm, *args, **kwargs):
        """
        We require a rm as NN implementations need
        to know its dimension to properly initialize
        """
        raise NotImplementedError("reset")

    @abc.abstractmethod
    def get_statistics(self):
        raise NotImplementedError("get_statistics")

    @abc.abstractmethod
    def set_save_path(self, path, **kwargs):
        """
        Define the path to a folder that can be used to persist complex algorithm state

        Invoking this method basically grants the Algo instance permission to use the specified folder to store
        any type of information that it might need to persist, e.g: when implementing custom (de)serialization logic.

        Parameters
        ----------
        path The path to be registered as the save path
        kwargs Additional keyword arguments

        """
        raise NotImplementedError("set_save_path")
