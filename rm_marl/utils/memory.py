import random
from collections import namedtuple, deque

from rm_marl.algo.deepq.exceptions import NotEnoughExperiencesError


class ExperienceBuffer:
    """
    Experience Replay Memory buffer implementation for Reward Machines

    This class implements the circular buffer used to implement the Experience Replay mechanism described by
    Minh et al. in their seminal paper "Playing Atari with Deep Reinforcement Learning" from 2015.
    """

    def __init__(self, size, seed, minimum_size: int = None):
        """
        Initialize the replay memory, given its maximum capacity

        Once the memory is full, the oldest experiences are discarded to make space for the newer ones.

        Parameters
        ----------
        experience : collections.namedtuple containing the memory contents
        size The maximum capacity of the replay memory, in number of experience samples
        min_size The minimum size of the replay memory for which experience can be sampled

        """

        self._size = size
        self._buffer = deque(maxlen=self._size)

        self.minimum_size = minimum_size

        self._rng = random.Random(seed)

    def __len__(self):
        return len(self._buffer)

    def push(self, experience):
        """
        Push a new experience sample into the replay memory

        Parameters
        ----------
        experience : namedtuple containing the experience
        """
        self._buffer.append(experience)

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

        # Use minimum size if it exists. Otherwise use batch_size
        if len(self._buffer) < self.minimum_size or batch_size:
            raise NotEnoughExperiencesError(requested=batch_size, available=len(self._buffer))

        return self._rng.sample(self._buffer, batch_size)

    def clear(self):
        """
        Empty out the replay memory
        """

        self._buffer.clear()
