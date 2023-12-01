from collections import defaultdict
from typing import Optional

import gym.spaces
import numpy as np
from gym.utils import seeding

from ._base import Algo

class QRM(Algo):

    _np_random: Optional[np.random.Generator] = None

    def __init__(
        self,
        action_space: "gym.spaces.Space" = None,
        epsilon: float = 0.0,
        temperature: float = 50.0,
        alpha: float = 0.8,
        gamma: float = 0.9,
        seed: int = 123,
    ):
        assert isinstance(action_space, gym.spaces.Discrete)
        self.action_space = action_space
        self.epsilon = epsilon
        self.temperature = temperature
        self.alpha = alpha
        self.gamma = gamma
        self._seed = seed

        self.q = defaultdict(self._q_sa_constructor)

        self.reset(seed=seed)

    def _q_a_constructor(self):
        return np.zeros((self.action_space.n))

    def _q_sa_constructor(self):
        return defaultdict(self._q_a_constructor)

    def reset(self, seed: Optional[int] = None):
        seed = self._seed if seed is None else seed
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self.q.clear()

    @staticmethod
    def _to_hashable_state_(state):
        return tuple(
            sorted({k: tuple(v) for k, v in state.items()}.items(), key=lambda i: i[0])
        )

    @staticmethod
    def _to_hashable_rm_state(u):
        if isinstance(u, (int, str)):
            return u

        # hashing np.ndarray[float]
        # This case is very rare and mainly used for sanity check of the implementation
        # In general, DeepQ should be used instead.

        # Represent array as str -> can be hashed
        u_str = ",".join([str(np.round(elem, decimals=3)) for elem in u])
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

        return loss

    def action(self, state, u, greedy: bool = False):

        if self._np_random.random() < self.epsilon:
            action = self._np_random.choice(range(self.action_space.n))
        elif not greedy:
            pr_sum = np.sum(
                np.exp(self.q[self._to_hashable_rm_state(u)][self._to_hashable_state_(state)] * self.temperature)
            )
            pr = (
                    np.exp(self.q[self._to_hashable_rm_state(u)][self._to_hashable_state_(state)] * self.temperature)
                    / pr_sum
            )

            # If any q-values are so large that the softmax function returns infinity,
            # make the corresponding actions equally likely
            if any(np.isnan(pr)):
                print("BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX.")
                temp = np.array(np.isnan(pr), dtype=float)
                pr = temp / np.sum(temp)

            cdf = np.insert(np.cumsum(pr), 0, 0)

            randn = self._np_random.random()
            for a in range(self.action_space.n):
                if randn >= cdf[a] and randn <= cdf[a + 1]:
                    action = a
                    break
        else:
            best_actions = np.where(
                self.q[self._to_hashable_rm_state(u)][self._to_hashable_state_(state)]
                == np.max(self.q[self._to_hashable_rm_state(u)][self._to_hashable_state_(state)])
            )[0]
            action = self._np_random.choice(best_actions)

        return action
