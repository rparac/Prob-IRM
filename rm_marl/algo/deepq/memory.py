import random
from collections import namedtuple, deque

from rm_marl.algo.deepq.exceptions import NotEnoughExperiencesError


class ReplayMemoryRM:
    """
    Experience Replay Memory buffer implementation for Reward Machines

    This class implements the circular buffer used to implement the Experience Replay mechanism described by
    Minh et al. in their seminal paper "Playing Atari with Deep Reinforcement Learning" from 2015.
    """

    Experience = namedtuple('Experience', [
        'u',
        'state',
        'action',
        'reward',
        'done',
        'new_state',
        'new_u'
    ])

    def __init__(self, size, seed):
        """
        Initialize the replay memory, given its maximum capacity

        Once the memory is full, the oldest experiences are discarded to make space for the newer ones.

        Parameters
        ----------
        size The maximum capacity of the replay memory, in number of experience samples

        """

        self._size = size
        self._buffer = deque(maxlen=self._size)

        self._rng = random.Random(seed)

    def __len__(self):
        return len(self._buffer)

    def push(self, state, u, action, reward, done, new_state, new_u):
        """
        Push a new experience sample into the replay memory

        Parameters
        ----------
        state       The environmental state where the experience began
        u           The reward machine state where the experience began
        action      The action taken by the agent
        reward      The reward obtained by the agent in this experience
        done        True if the episode ended after this experience
        new_state   The new state reached by the agent after executing its action
        new_u       The new state reached by the agent's RM after this experience
        """

        assert new_state is not None, 'Tried to push an experience leading to a "None" new env state'
        assert new_u is not None, 'Tried to push an experience leading to a "None" new RM state'

        experience_sample = ReplayMemoryRM.Experience(u, state, action, reward, done, new_state, new_u)
        self._buffer.append(experience_sample)

    def sample(self, batch_size):
        """
        Sample a random batch of experiences relating to a specific RM state from the replay memory

        Note that if a batch is requested with a bigger size than the current number of entries in the replay memory,
        an exception is raised. This makes sure that, if the method returns, the user is always provided an actual
        random batch of the desired size.

        Parameters
        ----------
        batch_size The size of the requested batch, in number of samples

        Returns
        -------
        A list of Experiences

        """

        if len(self._buffer) < batch_size:
            raise NotEnoughExperiencesError(requested=batch_size, available=len(self._buffer))

        return self._rng.sample(self._buffer, batch_size)

    def clear(self):
        """
        Empty out the replay memory
        """

        self._buffer.clear()
