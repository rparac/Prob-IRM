from collections import namedtuple, deque

import random


Experience = namedtuple('Experience', [
    'state',
    'action',
    'next_state',
    'reward'
])


class ReplayMemory:
    """
    Experience Replay Memory buffer implementation

    This class implements the circular buffer used to implement the Experience Replay mechanism described by
    Minh et al. in their seminal paper "Playing Atari with Deep Reinforcement Learning" from 2013.
    """

    def __init__(self, size):
        """
        Initialize the replay memory, given its maximum capacity

        Once the memory is full, the oldest experiences are discarded to make space for the newer ones.

        Parameters
        ----------
        size The maximum capacity of the replay memory, in number of experience samples

        """

        self._size = size
        self._buffer = deque(maxlen=size)

    def push(self, state, action, new_state, reward):
        """
        Push a new experience sample into the replay memory

        Parameters
        ----------
        state The environmental state where the experience began
        action The action taken by the agent
        new_state The new state reached by the agent after executing its action
        reward The reward obtained by the agent in this experience

        """

        experience_sample = Experience(state, action, new_state, reward)
        self._buffer.append(experience_sample)

    def sample(self, batch_size):
        """
        Sample a random batch of experiences from the replay memory

        Note that if a batch is requested with a bigger size than the current number of entries in the replay memory,
        an exception is raised. This makes sure that, if the method returns, the user is always provided an actual
        random batch of the desired size.

        Parameters
        ----------
        batch_size The size of the requested batch, in number of samples

        Returns
        -------
        A list of Experience instances

        """

        if len(self._buffer) < batch_size:
            raise NotEnoughExperiencesError(requested=batch_size, available=len(self._buffer))

        return random.sample(self._buffer, batch_size)


class NotEnoughExperiencesError(Exception):

    def __init__(self, *, requested, available):
        super().__init__(f'A batch of size {requested} was requested, but only {available} experiences are available in the replay memory')
