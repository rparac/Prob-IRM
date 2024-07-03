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
import math

import numpy as np
import torch.nn as nn
import torch
from torch.optim import Optimizer, AdamW
import gymnasium as gym

from typing import Type
from collections import namedtuple, defaultdict
import os
import os.path

from ._base import Algo
from .deepq.networks import DeepQNetwork, OfficeQNetwork, SimpleQNetwork
from ..reward_machine import RewardMachine
from ..utils.memory import ExperienceBuffer


DeepQExperience = namedtuple('DeepQExperience', [
    'u',
    'state',
    'action',
    'reward',
    'done',
    'new_state',
    'new_u'
])

class DeepQRM(Algo):

    RMStatePolicy = namedtuple("RMStatePolicy", [
        'policy_network',
        'policy_optimizer',
        'target_network'
    ])

    def __init__(
            self, *
            obs_space: "gym.spaces.Space",
            action_space: "gym.spaces.Discrete",
            num_policy_layers: int = 5,
            policy_layer_size: int = 64,
            replay_size: int = 10000,
            batch_size: int = 32,
            policy_train_freq: int = 1,
            target_update_freq: int = 100,
            gamma: float = 0.9,
            epsilon_start: float = 1.0,
            epsilon_end: float = 0.0,
            epsilon_decay: int = 100,
            temperature: float = 50.0,
            optimizer_cls: Type[Optimizer] = AdamW,
            optimizer_kws: dict = None,
            policy_reset_method: str = 'default',
            seed: int = 0,
            use_simpleqnet: bool = False,
            use_crm: bool = False,
            use_dropout: bool = False,
            num_rm_states: int = 1,
            max_rm_states: int = 15,
    ):

        super().__init__()

        # Sub-policies-related parameters
        self._obs_space = obs_space
        self._dim_obs = gym.spaces.utils.flatdim(obs_space)
        self._num_policy_layers = num_policy_layers
        self._layers_size = policy_layer_size
        self._num_actions = action_space.n
        self._num_rm_states = num_rm_states
        self._max_rm_states = max_rm_states

        # PRNGs
        self._curr_seed = seed
        torch.manual_seed(seed)
        self._np_random, self._curr_seed = gym.utils.seeding.np_random(self._curr_seed)

        # Optimizers
        self._optimizer_cls = optimizer_cls
        self._optimizer_kws = optimizer_kws or {"lr": 1e-4, "amsgrad": True}  # Defaults assume AdamW

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Q-Network related configuration
        self._use_simpleqnet = use_simpleqnet
        self._use_crm = use_crm
        self._use_dropout = use_dropout

        # Mappings from RM states to RMStatePolicy instances
        self._init_q_networks()

        # Replay Memory
        self._replay_size = replay_size
        self._init_replay_memory()

        # Learning parameters
        self._batch_size = batch_size
        self._target_update_freq = target_update_freq
        self._policy_train_freq = policy_train_freq
        self._gamma = gamma

        # Internal parameters used to make decisions
        self._policies_train_timer = self._policy_train_freq
        self._target_update_timer = self._target_update_freq
        self._temperature = temperature
        self._epsilon_start = epsilon_start
        self._epsilon_end = epsilon_end
        self._epsilon_decay = epsilon_decay
        self._policy_reset_method = policy_reset_method

        # Statistics
        self._policy_age = 0
        self._learn_calls = 0
        self._target_updates = 0
        self._subpolicy_updates = defaultdict(lambda: 0)

        # Save path to store/load instances
        self._save_path = None

    @property
    def _epsilon(self):
        return self._epsilon_end + (self._epsilon_start - self._epsilon_end) * math.exp(
            -1 * self._policy_age / self._epsilon_decay)

    def _init_memory(self):
        return ExperienceBuffer(self._replay_size, self._curr_seed)

    def _init_q_network(self):
        """
        Initialize a new Q-Network, using the structural parameters given when this DeepQRM instance
        was created.

        Returns
        -------
        A newly created DeepQNetwork instance
        """

        if self._use_crm:
            return OfficeQNetwork(
                self._dim_obs,
                self._max_rm_states,
                self._num_actions,
                self._use_dropout
            ).to(self.device)

            # return CRMNetwork(
            #     self._dim_obs,
            #     self._max_rm_states,  # Max number of allowed RM states is used due to padding
            #     self._num_policy_layers,
            #     self._layers_size,
            #     self._num_actions,
            # ).to(self.device)

        if self._use_simpleqnet:
            return SimpleQNetwork(
                self._dim_obs,
                self._max_rm_states,
                num_actions=4,
                learn_biases=False
            )

        return DeepQNetwork(
            self._dim_obs,
            self._num_policy_layers,
            self._layers_size,
            self._num_actions
        ).to(self.device)

    def _init_q_network_pair(self):
        """
        Initialize a pair of Q-networks

        Specifically, this method returns a pair of two Q-networks, where the first can be used as a policy
        network for any given RM state, while the second can be used as its associated target network (DDQN).

        Returns
        -------
        A pair of DeepQNetworks (policy_network, target_network)
        """

        # Create the neural networks
        policy_network = self._init_q_network()
        target_network = self._init_q_network()

        # Create the optimizer that will update the policy network parameters
        policy_optimizer = self._optimizer_cls(
            policy_network.parameters(),
            **self._optimizer_kws
        )

        # Synch the weights from the policy network to the target network
        policy_state = policy_network.state_dict()
        target_network.load_state_dict(policy_state)

        # Disable computation for gradients for target network, as its weights
        # are not trained but copied periodically from the policy network
        target_network.requires_grad_(False)

        return DeepQRM.RMStatePolicy(policy_network, policy_optimizer, target_network)

    def _reset_q_networks(self, *, method):

        assert method in ['sum_gaussian', 'random_init'], f'Unrecognized reset method: "{method}"'

        if method == 'sum_gaussian':

            for q_net, _, t_net in self._q_networks.values():

                with torch.no_grad():

                    for param in q_net.network.parameters():
                        std_dev, mean = torch.std_mean(param)
                        gaussian_noise = ((0.015 * std_dev) ** 0.5) * torch.randn_like(param)
                        param.add_(gaussian_noise)

                    self._update_target_networks()

        elif method == 'random_init':

            self._init_q_networks()

        else:
            assert False, f'Unrecognized method: "{method}"'

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
            dtype=torch.float32
        )
        flat_next_state = torch.as_tensor(
            gym.spaces.utils.flatten(self._obs_space, next_state),
            dtype=torch.float32
        )
        reward = torch.as_tensor([reward], dtype=torch.float32, device=self.device)
        action = torch.as_tensor([action], device=self.device)

        # Add experience to the replay buffer
        experience = DeepQExperience(u, flat_state, action, reward, done, flat_next_state, next_u)
        if self._use_crm or self._use_simpleqnet:
            self._replay_memory[None].push(experience)
        else:
            self._replay_memory[u].push(experience)

        # Check if the sub-policies need to be trained
        self._policies_train_timer -= 1
        if self._policies_train_timer > 0:
            return np.NAN

        # On each actual learning step, update the policy associated with every RM state
        losses = []
        for u, (policy_net, optimizer, target_net) in self._q_networks.items():
            memory = self._replay_memory[u]
            loss = self._subpolicy_training_step(policy_net, optimizer, target_net, memory)
            losses.append(loss)

        # Training has been done: reset the corresponding timer and update policy age
        self._policies_train_timer = self._policy_train_freq
        self._policy_age += 1

        # Check if the target networks need to be updated
        self._target_update_timer -= 1
        if self._target_update_timer == 0:
            self._update_target_networks()
            self._target_update_timer = self._target_update_freq

        # The overall loss is the mean loss obtained among all sub-policies
        valid_losses = [loss for loss in losses if loss is not None]
        return sum(valid_losses) / len(valid_losses) if len(valid_losses) > 0 else np.NAN

    # def _subpolicy_training_step(self, rm_state):
    def _subpolicy_training_step(self, q_net, optimizer, t_net, memory):

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

        # TODO: add this at call site
        # self._subpolicy_updates[rm_state] += 1

        # Only learn if we have enough experiences to form a batch
        if len(memory) < self._batch_size:
            return None

        batch = memory.sample(self._batch_size)
        curr_rm_states, states, actions, rewards, dones, next_states, next_rm_states = tuple(zip(*batch))
        states = torch.stack(states).to(self.device)
        rewards = torch.stack(rewards).to(self.device)
        actions = torch.stack(actions).to(self.device)
        next_states = torch.stack(next_states).to(self.device)
        curr_rm_states = torch.stack([self._vectorize(u) for u in curr_rm_states]).to(self.device)

        # Use the policy network to compute the Q-value estimates for each (s, a) pair
        q_estimates = q_net(states, curr_rm_states).gather(1, actions)

        # Determine which s' and u' are not terminal states
        non_terminal_mask = [not d for d in dones]
        non_terminal_next_states = next_states[non_terminal_mask]
        non_terminal_next_rm_states = [self._vectorize(u) for u, d in zip(next_rm_states, dones) if not d]

        # Gather the values predicted by the target network associated with the next RM states of each experience
        target_values = []
        for s, u in zip(non_terminal_next_states, non_terminal_next_rm_states):
            # TODO: make double DQN a parameter; in DQN the action would be chosen by target_network
            best_act = q_net(s, u).argmax()
            t = t_net(s, u)[best_act].unsqueeze(0)
            t_old = t_net(s, u).max().unsqueeze(0)
            target_values.append(t)

        #
        # target_values = torch.stack([
        #     self._q_networks[u].target_network(s).max().unsqueeze(0)
        #     for s, u
        #     in zip(non_terminal_next_states, non_terminal_next_rm_states)
        # ])

        # Compute the value estimates V(s') = max_{a'} Q(s', a') for next states according to the target network
        # By definition, V(s) = 0 for a terminal state s
        next_states_values = torch.zeros((self._batch_size, 1), device=self.device)

        if target_values:
            target_values = torch.stack(target_values)
            next_states_values[non_terminal_mask] = target_values

        # Compute the Q-values we expected to produce with our policy network
        expected_q_estimates = (self._gamma * next_states_values) + rewards

        if self._use_crm:
            loss = nn.HuberLoss()(q_estimates, expected_q_estimates)
        else:
            loss = nn.MSELoss()(q_estimates, expected_q_estimates)

        # Reset gradients to zero to avoid being influenced by previous batches
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def action(self, state, u, greedy: bool = True, testing: bool = False, **kwargs):
        """
        Compute an action to be taken based on the state of the policy

        The action is computed following an epsilon-greedy strategy based on the value for epsilon given at the time
        the DeepQRM instance was created.

        Parameters
        ----------
        state The current environmental state
        u The current agent's RM state
        greedy Ignored parameter; needed for compatibility with Trainer code
        testing Ignored parameter; are we running the code in test mode

        Returns
        -------
        An action, selected according to an epsilon-greedy strategy

        """

        if not testing and self._np_random.random() < self._epsilon:
            return self._np_random.choice(range(self._num_actions))

        # A non-random action is to be taken, prepare the flat env state and feed it to the Q-network
        flat_state = torch.as_tensor(
            gym.spaces.utils.flatten(self._obs_space, state),
            dtype=torch.float,
            device=self.device
        )
        if self._use_crm or self._use_simpleqnet:
            # TODO: need to deal with string case
            rm_state = self._vectorize(u)
            action_values = self._q_networks[None].policy_network(flat_state, rm_state)
        else:
            action_values = self._q_networks[u].policy_network(flat_state, u)

        # Softmax action selection
        if False and not greedy:
            exp_values = torch.exp(action_values * self._temperature)
            action_probabilities = exp_values / sum(exp_values)

            # Check for NANs, ie: q-values are too large and softmax returns infinity
            # If so, make every corresponding action equally likely
            if any(torch.isnan(action_probabilities)):
                print("BOLTZMANN CONSTANT TOO LARGE IN ACTION-SELECTION SOFTMAX")
                temp = torch.as_tensor(torch.isnan(action_probabilities), dtype=torch.float)
                action_probabilities = temp / torch.sum(temp)

            cumulated_probabilities = torch.cat(
                (torch.tensor([0], device=self.device), torch.cumsum(action_probabilities, dim=0)))
            rand = self._np_random.random()
            for action in range(self._num_actions):
                if cumulated_probabilities[action] <= rand <= cumulated_probabilities[action + 1]:
                    return action

        # Greedy action selection
        else:
            best_action_value = action_values.max()
            best_actions_mask = action_values == best_action_value
            best_actions = [t.item() for t in torch.nonzero(best_actions_mask)]
            return self._np_random.choice(best_actions)

    def _vectorize(self, u):
        """
        Return a vector representation of the given RM state

        If the RM state is represented as a simple integer, its one-hot encoding its returned.
        Otherwise, if the RM is already a vector, it is simply converted to a PyTorch tensor.

        In any case, the returned representation is padded to the maximum number of allowed RM states, as specified
        when the DeepQRM instance was initialized.

        Parameters
        ----------
        u The RM state representation that needs to be vectorized

        Returns
        -------
        A torch.Tensor instance containing the vector representation of the given RM state

        """
        padded_rm_state = torch.zeros(self._max_rm_states, device=self.device, dtype=torch.float32)

        if isinstance(u, int):
            padded_rm_state[u] = 1
        else:
            rm_state = torch.tensor(u, device=self.device, dtype=torch.float32)
            padded_rm_state[:len(rm_state)] = rm_state

        return padded_rm_state

    def on_env_reset(self, *args, **kwargs):
        # Nothing to do
        return

    def on_rm_reset(self, rm: RewardMachine, **kwargs):
        """
        Resets the DeepQRM policy to its initial state

        Invoking this method discards all the experiences contained in the replay memory and every sub-policy
        trained so far. This allows the instance to be re-used without the need for creating a new one.

        """

        assert len(rm.states) < self._max_rm_states, f'New RM contains to many states: {len(rm.states)} > {self._max_rm_states}'

        # Clear the sub-policies
        self._num_rm_states = len(rm.states)

        if self._policy_reset_method != 'default':
            self._reset_q_networks(method=self._policy_reset_method)
        else:
            if self._use_crm or self._use_simpleqnet:
                self._reset_q_networks(method='sum_gaussian')
            else:
                self._reset_q_networks(method='random_init')

        # Clear replay memory
        self._init_replay_memory()

        # Reset internal timers
        self._policies_train_timer = self._policy_train_freq
        self._target_update_timer = self._target_update_freq

        # Re-seed the PRNGs
        self._np_random, self._curr_seed = gym.utils.seeding.np_random(self._curr_seed)

        # Reset statistics
        self._policy_age = 0

    def get_statistics(self):

        stats = {
            "policy_age": self._policy_age,
            "epsilon": self._epsilon
        }

        return stats

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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create a new, empty Replay Buffer
        self._init_replay_memory()

        # Restore the _subpolicy_updates statistics
        self._subpolicy_updates = defaultdict(lambda: 0, self._subpolicy_updates)

        # Restore the Q-networks based on the files found in the save path
        self._init_q_networks()
        for subpolicy_state_file in [f for f in os.listdir(self._save_path) if f.endswith('.pth')]:

            # Load the state dictionary
            full_path = os.path.join(self._save_path, subpolicy_state_file)
            subpolicy_state = torch.load(full_path, map_location=self.device)

            rm_state_str = subpolicy_state_file
            if rm_state_str.startswith('subpolicy_'):
                rm_state_str = rm_state_str[len('subpolicy_'):]
            if rm_state_str.endswith('.pth'):
                rm_state_str = rm_state_str[:-len('.pth')]

            # Handle both integer and string-based RM state IDs
            try:
                if rm_state_str == "None":
                    rm_state = None
                else:
                    rm_state = int(rm_state_str)
            except ValueError:
                rm_state = rm_state_str

            subpolicy_network = self._q_networks[rm_state].policy_network
            subpolicy_network.load_state_dict(subpolicy_state)

        # Finally, propagate all the parameters to the target networks
        self._update_target_networks()
        self._target_updates -= 1  # Compensate for the +1 made in _update_target_networks()

    def _init_q_networks(self):
        if self._use_crm:
            self._q_networks = {None: self._init_q_network_pair()}
        else:
            self._q_networks = defaultdict(self._init_q_network_pair)

    def _init_replay_memory(self):
        if self._use_crm:
            self._replay_memory = {None: ExperienceBuffer(self._replay_size, self._curr_seed)}
        else:
            self._replay_memory = defaultdict(self._init_memory)
