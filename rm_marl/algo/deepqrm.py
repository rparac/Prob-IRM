"""
Deep Q-Learning with Reward Machines (DQRM) algorithm implementation.

This implementation is based on the algorithm described by Icarte et al. in their awesome paper:

"Using Reward Machines for High-Level Task Specification and Decomposition in Reinforcement Learning
 Rodrigo Toro Icarte, Toryn Klassen, Richard Valenzano, Sheila McIlraith
 Proceedings of the 35th International Conference on Machine Learning, PMLR 80:2107-2116, 2018."

Moreover, the code of this module was developed using, as a guide, the implementation of the DQRM algorithm as provided
Icarte himself in this repository: https://bitbucket.org/RToroIcarte/lrm/src

We kindly thank the original authors for their amazing contribution to the neuro-symbolic reinforcement learning
literature and for making their code freely available for the research community.
"""

import numpy as np
import torch.nn as nn
import torch
import gym


from collections import namedtuple, deque, defaultdict
import random
import os
import os.path


from ._base import Algo


class DeepQRM(Algo):

    # TODO: Seeding

    RMStatePolicy = namedtuple("RMStatePolicy", [
        'policy_network',
        'policy_optimizer',
        'target_network'
    ])

    def __init__(
        self,
        obs_space: "gym.spaces.Space",
        action_space: "gym.spaces.Discrete",
        num_policy_layers: int = 5,
        policy_layer_size: int = 64,
        replay_size: int = 10000,
        batch_size: int = 32,
        policy_train_freq: int = 1,
        target_update_freq: int = 100,
        gamma: float = 0.9,
        epsilon: float = 0.1,
        lr: float = 1e-4
    ):

        super().__init__()

        # Sub-policies-related parameters
        self._obs_space = obs_space
        self._dim_obs = gym.spaces.utils.flatdim(obs_space)
        self._num_policy_layers = num_policy_layers
        self._layers_size = policy_layer_size
        self._num_actions = action_space.n

        # Mappings from RM states to RMStatePolicy instances
        self._q_networks = defaultdict(self._init_q_network_pair)

        # Replay Memory
        self._replay_size = replay_size
        self._replay_memory = ReplayMemoryRM(self._replay_size)

        # Learning parameters
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._policy_train_freq = policy_train_freq
        self._gamma = gamma
        self._epsilon = epsilon
        self._learning_rate = lr

        # Internal parameters used to make decisions
        self._policies_train_timer = self._policy_train_freq
        self._target_update_timer = self._target_update_freq

        # PRNGs
        self._np_random, _ = gym.utils.seeding.np_random()

        # Statistics
        # NB: These are NOT reset when self.reset() is called
        self._learn_calls = 0
        self._target_updates = 0
        self._subpolicy_updates = defaultdict(lambda: 0)

        # Save path to store/load instances
        self._save_path = None

    def _init_q_network(self):
        """
        Initialize a new Q-Network, using the structural parameters given when this DeepQRM instance
        was created.

        Returns
        -------
        A newly created DeepQNetwork instance
        """

        return DeepQNetwork(
            self._dim_obs,
            self._num_policy_layers,
            self._layers_size,
            self._num_actions
        )

    def _init_q_network_pair(self):
        """
        Initialize a pair of Q-network

        Specifically, this method returns a pair of two Q-networks, where the first can be used as a policy
        network for any given RM state, while the second can be used as its associated target network.

        Returns
        -------
        A pair of DeepQNetworks (policy_network, target_network)
        """

        # Create the neural networks
        policy_network = self._init_q_network()
        target_network = self._init_q_network()

        # Create the optimizer that will update the policy network parameters
        policy_optimizer = torch.optim.AdamW(
            policy_network.parameters(),
            lr=self._learning_rate,
            amsgrad=True
        )

        # Synch the weights from the policy network to the target network
        policy_state = policy_network.state_dict()
        target_network.load_state_dict(policy_state)

        # Disable computation for gradients for target network, as its weights
        # are not trained but copied periodically from the policy network
        target_network.requires_grad_(False)

        return DeepQRM.RMStatePolicy(policy_network, policy_optimizer, target_network)

    def _update_target_networks(self):
        """
        Update the weghts of each target network by copying the weights of their associated policy networks.
        """

        self._target_updates += 1

        for rm_state, (q_net, _, t_net) in self._q_networks.items():
            q_net_state = q_net.state_dict()
            t_net.load_state_dict(q_net_state)

    def learn(self, state, u, action, reward, done, next_state, next_u):

        self._learn_calls += 1

        # Convert s, s' and reward to torch.Tensor before storing them, to avoid the need for
        # carrying out the conversion when sampling from the replay memory
        flat_state = torch.as_tensor(
            gym.spaces.utils.flatten(self._obs_space, state),
            dtype=torch.float
        )
        flat_next_state = torch.as_tensor(
            gym.spaces.utils.flatten(self._obs_space, next_state),
            dtype=torch.float
        )
        reward = torch.as_tensor([reward])
        action = torch.as_tensor([action])

        # Add experience to the replay buffer
        self._replay_memory.push(flat_state, u, action, reward, done, flat_next_state, next_u)

        # Check if the sub-policies need to be trained
        self._policies_train_timer -= 1
        if self._policies_train_timer > 0:
            return np.NAN

        # On each actual learning step, update the policy associated with every RM state
        losses = []
        for rm_state in self._q_networks.copy().keys():
            loss = self._subpolicy_training_step(rm_state)
            losses.append(loss)

        # Training has been done: reset the corresponding timer
        self._policies_train_timer = self._policy_train_freq

        # Check if the target networks need to be updated
        self._target_update_timer -= 1
        if self._target_update_timer == 0:
            self._update_target_networks()
            self._target_update_timer = self._target_update_freq

        # The overall loss is the mean loss obtained among all sub-policies
        valid_losses = [loss for loss in losses if loss is not None]
        return sum(valid_losses) / len(valid_losses) if len(valid_losses) > 0 else np.NAN

    def _subpolicy_training_step(self, rm_state):
        """
        Perform a training iteration for the sub-policy associated with the given RM state

        Note that the sub-policy will only be trained if the replay memory contains a number of experiences
        associated with the given RM state higher than the batch size requested when the DeepQRM instance
        was initialized.

        Parameters
        ----------
        rm_state The reward machine state associated with the sub-policy that needs to be trained.

        Returns
        -------
        The value for the loss function obtained in the training step

        """

        self._subpolicy_updates[rm_state] += 1

        # Only learn if we have enough experiences to form a batch
        if self._replay_memory.n_entries_for_state(rm_state) < self._batch_size:
            return None

        q_net, optimizer, _ = self._q_networks[rm_state]

        batch = self._replay_memory.sample(rm_state, self._batch_size)
        states, actions, rewards, dones, next_states, next_rm_states = tuple(zip(*batch))
        states = torch.stack(states)
        rewards = torch.stack(rewards)
        actions = torch.stack(actions)
        next_states = torch.stack(next_states)

        # Use the policy network to compute the Q-value estimates for each (s, a) pair
        q_estimates = q_net(states).gather(1, actions)

        # Determine which s' and u' are not terminal states
        non_terminal_mask = [not d for d in dones]
        non_terminal_next_states = next_states[non_terminal_mask]
        non_terminal_next_rm_states = [u for u, d in zip(next_rm_states, dones) if not d]

        # Gather the values predicted by the target network associated with the next RM states of each experience
        target_values = torch.stack([
            self._q_networks[u].target_network(s).max().unsqueeze(0)
            for s, u
            in zip(non_terminal_next_states, non_terminal_next_rm_states)
        ])

        # Compute the value estimates V(s') = max_{a'} Q(s', a') for next states according to the target network
        # By definition, V(s) = 0 for a terminal state s
        next_states_values = torch.zeros((self._batch_size, 1))
        next_states_values[non_terminal_mask] = target_values

        # Compute the Q-values we expected to produce with our policy network
        expected_q_estimates = (self._gamma * next_states_values) + rewards
        loss = nn.MSELoss()(q_estimates, expected_q_estimates)

        # Reset gradients to zero to avoid being influenced by previous batches
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    def action(self, state, u, greedy: bool = True):
        """
        Compute an action to be taken based on the state of the policy

        The action is computed following an epsilon-greedy strategy based on the value for epsilon given at the time
        the DeepQRM instance was created.

        Parameters
        ----------
        state The current environmental state
        u The current agent's RM state
        greedy Ignored parameter; needed for compatibility with Trainer code # TODO: Add action-selection softmax

        Returns
        -------
        An action, selected according to an epsilon-greedy strategy

        """

        if self._np_random.random() < self._epsilon:
            return self._np_random.choice(range(self._num_actions))

        else:
            flat_state = torch.as_tensor(
                gym.spaces.utils.flatten(self._obs_space, state),
                dtype=torch.float
            )
            action_values = self._q_networks[u].policy_network(flat_state)
            best_action_value = action_values.max()
            best_actions_mask = action_values == best_action_value
            best_actions = [t.item() for t in torch.nonzero(best_actions_mask)]
            return self._np_random.choice(best_actions)

    def reset(self):
        """
        Resets the DeepQRM policy to its initial state

        Invoking this method discards all the experiences contained in the replay memory and every sub-policy
        trained so far. This allows the instance to be re-used without the need for creating a new one.

        """

        self._q_networks.clear()
        self._replay_memory.clear()

        self._policies_train_timer = self._policy_train_freq
        self._target_update_timer = self._target_update_freq

        self._np_random, _ = gym.utils.seeding.np_random()

    def set_save_path(self, path, **kwargs):

        assert os.path.isdir(path), f"An invalid save path was specified: {path}"
        self._save_path = path

    def __getstate__(self):
        """
        Pickle-related: return the serializable portion of the instances' state

        See https://docs.python.org/3/library/pickle.html#pickle-state for more info
        """

        state = self.__dict__.copy()

        # Since the policy and target networks are implemented via PyTorch, we want to
        # manually handle their loading and saving
        del state['_q_networks']

        # Similarly, we do not want to store the entire replay buffer
        del state['_replay_memory']

        # We also need to manually handle the serialization of this attribue
        # as it relies on a lambda function
        state['_subpolicy_updates'] = dict(self._subpolicy_updates)

        # Manually store only the policy network parameters as they can be used
        # to also restore the target networks as well
        for rm_state, (q_net, _, _) in self._q_networks.items():
            file = os.path.join(self._save_path, f'subpolicy_{rm_state}.pth')
            torch.save(q_net.state_dict(), file)

        return state

    def __setstate__(self, state):
        """
        Pickle-related: restore the additional state required by the instance

        See https://docs.python.org/3/library/pickle.html#pickle-state for more info
        """

        self.__dict__.update(state)

        # Create a new, empty Replay Buffer
        self._replay_memory = ReplayMemoryRM(self._replay_size)

        # Restore the _subpolicy_updates statistics
        self._subpolicy_updates = defaultdict(lambda: 0, self._subpolicy_updates)

        # Restore the Q-networks based on the files found in the save path
        self._q_networks = defaultdict(self._init_q_network_pair)
        for subpolicy_state_file in [f for f in os.listdir(self._save_path) if f.endswith('.pth')]:

            # Load the state dictionary
            full_path = os.path.join(self._save_path, subpolicy_state_file)
            subpolicy_state = torch.load(full_path)

            rm_state_str = subpolicy_state_file.removeprefix('subpolicy_').removesuffix('.pth')

            # Handle both integer and string-based RM state IDs
            try:
                rm_state = int(rm_state_str)
            except ValueError:
                rm_state = rm_state_str

            subpolicy_network = self._q_networks[rm_state].policy_network
            subpolicy_network.load_state_dict(subpolicy_state)

        # Finally, propagate all the parameters to the target networks
        self._update_target_networks()
        self._target_updates -= 1  # Compensate for the +1 made in _update_target_networks()


class DeepQNetwork(nn.Module):

    def __init__(self, dim_obs, num_layers, layer_size, num_actions):
        super(DeepQNetwork).__init__()

        # Initialize input layer
        self._input_layer = nn.Sequential(
            nn.Linear(dim_obs, layer_size),
            nn.ReLU()
        )

        # Initialize hidden layers
        self._hidden_layers = nn.Sequential()
        for i in range(num_layers):
            self._hidden_layers.append(nn.Linear(layer_size, layer_size))
            self._hidden_layers.append(nn.ReLU())

        # Initialize output layer: since we produce Q-values, not action probabilities,
        # we resort to a linear output layer
        self._output_layer = nn.Linear(layer_size, num_actions)

        # For ease of use, concatenate all the layers in a single Sequential module
        self._model = nn.Sequential(
            self._input_layer,
            self._hidden_layers,
            self._output_layer
        )

    def forward(self, x):
        return self._model(x)


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

        assert new_state is not None, 'Tried to push an experience leading to a "None" new env state'
        assert new_u is not None, 'Tried to push an experience leading to a "None" new RM state'

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
