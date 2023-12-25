
from collections import namedtuple, deque, defaultdict
import random




class ReplayMemoryRM:
    """
    Experience Replay Memory buffer implementation for Reward Machines

    This class implements the circular buffer used to implement the Experience Replay mechanism described by
    Minh et al. in their seminal paper "Playing Atari with Deep Reinforcement Learning" from 2015. However, it also
    includes a few modifications that allow it to be conveniently used with in reward machine-based scenarios.
    Specifically, this implementation grants its users the ability to sample only from the experiences that relate to
    a specific RM state.
    """

    Experience = namedtuple('Experience', [
        'state',
        'action',
        'reward',
        'done',
        'new_state',
        'new_u'
    ])

    def __init__(self, size):
        """
        Initialize the replay memory, given its maximum capacity

        Once the memory is full, the oldest experiences are discarded to make space for the newer ones.

        Parameters
        ----------
        size The maximum capacity of the replay memory, in number of experience samples

        """

        self._size = size
        self._buffers = defaultdict(self._init_buffer)

    def _init_buffer(self):
        """
        Initialize a replay buffer, which can be used to store the experiences relating to any RM state.
        The size of the buffer is the same specified when this ReplayMemoryRM instance was created.

        Returns
        -------
        A new replay buffer

        """

        return deque(maxlen=self._size)

    def __len__(self):
        """
        Return the current total number of experience entries stored in the replay memory

        Returns
        -------
        The total number of entries in the memory
        """

        return sum(len(buffer) for buffer in self._buffers.values())

    def n_entries_for_state(self, u):
        """
        Return the number of entries in the memory associated with the given RM state

        Parameters
        ----------
        u The RM state of interest

        Returns
        -------
        The number of entries associated with the given RM state
        """

        return len(self._buffers[u])

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

        experience_sample = ReplayMemoryRM.Experience(state, action, reward, done, new_state, new_u)
        self._buffers[u].append(experience_sample)

    def sample(self, u, batch_size):
        """
        Sample a random batch of experiences relating to a specific RM state from the replay memory

        Note that if a batch is requested with a bigger size than the current number of entries in the replay memory,
        an exception is raised. This makes sure that, if the method returns, the user is always provided an actual
        random batch of the desired size.

        Parameters
        ----------
        u          The RM state of interest
        batch_size The size of the requested batch, in number of samples

        Returns
        -------
        A list of Experience instances relating to the given RM state

        """

        if len(self._buffers[u]) < batch_size:
            raise NotEnoughExperiencesError(requested=batch_size, available=len(self._buffers[u]))

        return random.sample(self._buffers[u], batch_size)

    def clear(self):
        """
        Empty out the replay memory, eliminating every experience it contains for any RM state

        """

        self._buffers.clear()


class NotEnoughExperiencesError(Exception):

    def __init__(self, *, requested, available):
        super().__init__(f'A batch of size {requested} was requested, but only {available} experiences are available in the replay memory')
