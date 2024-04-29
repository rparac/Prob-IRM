"""
Taken from:
https://github.com/ertsiger/hrm-learning/blob/main/src/reinforcement_learning/replay.py
"""

import collections
import random

import numpy as np
import torch


class EpisodeExperienceBuffer:
    """
    Experience replay buffer that stores and samples sequences of experiences within episodes.
    """

    def __init__(self, capacity, seed):
        self.buffer = collections.deque(maxlen=capacity)

        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.buffer)

    def append(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size, seq_length):
        states = []
        observations = []
        actions = []
        next_states = []
        next_observations = []
        rewards = []
        is_terminals = []

        episode_ids = self._rng.sample(range(len(self.buffer)), batch_size)
        for episode_id in episode_ids:
            episode_length = len(self.buffer[episode_id])

            if seq_length >= episode_length:  # use the full episode
                start_ts = 0
                end_ts = episode_length
            else:  # use a subsequence of the episode (but trying to maximize its length for history accuracy)
                start_ts = self._np_rng.integers(0, episode_length - seq_length)
                end_ts = start_ts + seq_length

            sequence = self.buffer[episode_id][start_ts:end_ts]

            state, obs, action, reward, is_terminal, next_state, next_obs = zip(*sequence)
            states.append(state)
            observations.append(obs)
            actions.append(action)
            next_states.append(next_state)
            next_observations.append(next_obs)
            rewards.append(reward)
            is_terminals.append(tuple(torch.tensor(terminal) for terminal in is_terminal))

        return states, observations, actions, next_states, next_observations, rewards, is_terminals,

    def clear(self):
        self.buffer.clear()
