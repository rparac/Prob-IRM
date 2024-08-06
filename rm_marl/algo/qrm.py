import math
from collections import defaultdict
from typing import Optional

import gymnasium.spaces
import gymnasium
import numpy as np
from gymnasium.utils import seeding

from ._base import Algo
from ..reward_machine import RewardMachine


class QRM(Algo):
    _np_random: Optional[np.random.Generator] = None

    def __init__(
            self,
            action_space: "gym.spaces.Space" = None,
            temperature: float = 50.0,
            alpha: float = 0.8,
            gamma: float = 0.9,
            seed: int = 123,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.0,
            epsilon_decay: int = 100,
    ):
        assert isinstance(action_space, gymnasium.spaces.Discrete) or isinstance(action_space, gymnasium.spaces.Discrete)
        self.action_space = action_space
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self._seed = seed
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        # Controls the rate of exponential decay of epsilon, higher means slower decay
        self.epsilon_decay = epsilon_decay

        self.q = defaultdict(self._q_sa_constructor)

        # Policy statistics
        self._policy_age = 0

        # Q-table doesn't need RM information
        self.on_rm_reset(rm=None, seed=seed)

    @property
    def epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1 * self._policy_age / self.epsilon_decay)

    def _q_a_constructor(self):
        return np.zeros((self.action_space.n))

    def _q_sa_constructor(self):
        return defaultdict(self._q_a_constructor)

    def on_env_reset(self, *args, **kwargs):
        # Nothing to do
        pass

    def on_rm_reset(self, rm: Optional[RewardMachine], seed: Optional[int] = None, **kwargs):
        seed = self._seed if seed is None else seed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self.q.clear()

        self._policy_age = 0

    @staticmethod
    def _to_hashable_state_(state):
        dict_state = {k: v if isinstance(v, int) else tuple(v) for k, v in state.items()}
        return tuple(
            sorted(dict_state.items(), key=lambda i: i[0])
        )

    @staticmethod
    def _to_hashable_rm_state(u):
        if isinstance(u, (int, str)):
            return u

        # hashing np.ndarray[float]
        # This case is very rare and mainly used for sanity check of the implementation
        # In general, DeepQ should be used instead.

        # Represent array as str -> can be hashed
        # u_str = ",".join([str(np.round(elem, decimals=3)) for elem in u])
        u_str = ",".join([str(np.round(elem, decimals=2)) for elem in u])
        return u_str

    def learn(self, state, u, action, reward, done, next_state, next_u):
        next_q = np.amax(self.q[self._to_hashable_rm_state(next_u)][self._to_hashable_state_(next_state)])
        target_q = reward + (1 - int(done)) * self.gamma * next_q

        current_q = self.q[self._to_hashable_rm_state(u)][self._to_hashable_state_(state)][action]

        loss = np.abs(current_q - target_q)

        # Bellman update
        self.q[self._to_hashable_rm_state(u)][self._to_hashable_state_(state)][action] = (
                                                                                                 1 - self.alpha
                                                                                         ) * current_q + self.alpha * target_q

        self._policy_age += 1

        return loss

    def action(self, state, u, greedy: bool = False, testing: bool = False, **kwargs):
        random_act_selection = self._np_random.random() < self.epsilon
        if random_act_selection and not testing:
            action = self._np_random.choice(range(self.action_space.n))
        else:
            best_actions = np.where(
                self.q[self._to_hashable_rm_state(u)][self._to_hashable_state_(state)]
                == np.max(self.q[self._to_hashable_rm_state(u)][self._to_hashable_state_(state)])
            )[0]
            action = self._np_random.choice(best_actions)

        return action

    def get_statistics(self):

        stats = {
            "policy_age": self._policy_age,
            "epsilon": self.epsilon
        }

        return stats

    def set_save_path(self, path, **kwargs):
        """
        Do nothing

        Since QRM instances can simply be serialized and restore with python's default
        pickle behaviour, this method is effectively useless. It is, however, implemented to
        both document this detail and to comply with the Algo interface.

        """
        pass
