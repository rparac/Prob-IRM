from dataclasses import dataclass
from typing import DefaultDict, Any


@dataclass
class TrainState:
    """Stores the state of the training procedure"""
    episodes_completed: int
    steps: DefaultDict[Any, list]
    cumulative_steps: DefaultDict[Any, list]
    losses: DefaultDict[Any, list]
    rewards: DefaultDict[Any, list]
    shaping_rewards: DefaultDict[Any, list]
    successes: DefaultDict[Any, int]
    failures: DefaultDict[Any, int]
    timeouts: DefaultDict[Any, int]

    def get(self, key: str):
        return self.__dict__[key]
