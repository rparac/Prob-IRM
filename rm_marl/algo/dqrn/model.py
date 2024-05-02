import os
from collections import namedtuple
from typing import Dict, Type

import gym
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Optimizer, AdamW, RMSprop, Adadelta
from torch.optim.lr_scheduler import ExponentialLR

from rm_marl.algo import Algo
from rm_marl.algo.dqrn.buffer import EpisodeExperienceBuffer
from rm_marl.algo.dqrn.definitions import EpsilonAnnealingTimescale, DQRNStep
from rm_marl.algo.dqrn.networks import MinigridDRQN
from rm_marl.algo.dqrn.utils import embed_labels, get_annealed_exploration_rate
from rm_marl.utils.math_utils import randargmax
from rm_marl.utils.memory import ExperienceBuffer


# TODO: Daniel doesn't start training until

class DQRN(Algo):
    DQRNStatePolicy = namedtuple("DQRNStatePolicy", [
        'policy_network',
        'policy_optimizer',
        'lr_scheduler',
        'target_network',
    ])

    def __init__(self,
                 obs_space: "gym.spaces.Space",
                 action_space: "gym.spaces.Discrete",
                 num_observables: int,
                 seed: int = 123,
                 buffer_size: int = 1000,
                 policy_train_freq: int = 1,  # 16,
                 target_update_freq: int = 1500,
                 er_start_size: int = 1,  # size of experience replay before experiences can be sampled
                 er_sequence_length: int = 4,
                 # 4,  # 128,  # Sequence length for an experience sampled from replay buffer
                 er_batch_size: int = 1,  # batch size for the experience replay
                 gamma: float = 0.99,
                 optimizer_cls: Type[Optimizer] = Adadelta,  # Chosen
                 optimizer_kws: dict = None,
                 lstm_hidden_state=8,
                 embedding_num_layers=2,
                 embedding_layer_size=16,
                 embedding_output_size=8,
                 use_double_dqn=True,
                 use_gradient_clipping=False,
                 exploration_rate_annealing_timescale: EpsilonAnnealingTimescale = EpsilonAnnealingTimescale.EPISODES,
                 exploration_rate_init=1.0,
                 exploration_rate_final=0.1,
                 exploration_rate_annealing_duration: int = 2000,  # 5000,  # 300000,
                 ):

        self.obs_space = obs_space
        self.action_space = action_space
        self.num_observables = num_observables

        self._policy_train_freq = policy_train_freq
        self._target_update_freq = target_update_freq

        self._er_batch_size = er_batch_size
        self._er_start_size = er_start_size
        self._er_sequence_length = er_sequence_length

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._buffer_size = buffer_size
        self._replay_memory = EpisodeExperienceBuffer(buffer_size, seed)

        # Internal parameters
        self._policies_train_timer = self._policy_train_freq
        self._target_update_timer = self._target_update_freq

        # Network parameters
        # The chosen method for the network, state+obs seems to make the most sense
        #  https://github.com/ertsiger/hrm-learning/blob/main/src/reinforcement_learning/drqn_algorithm.py#L35
        self._lstm_method = "obs"  # "state+obs"
        self._lstm_hidden_size = lstm_hidden_state
        self._use_double_dqn = use_double_dqn
        self._use_gradient_clipping = use_gradient_clipping
        self._gamma = gamma

        # Optimizer
        self._lr_scheduler = None
        self._optimizer_cls = optimizer_cls
        self._optimizer_kws = optimizer_kws or {"lr": 0.1, "rho": 0.95}  # {"lr": 5e-4}

        # Network parameters
        self._embedding_num_layers = embedding_num_layers
        self._embedding_layer_size = embedding_layer_size
        self._embedding_output_size = embedding_output_size

        # Networks
        self._q_networks = self._init_q_network_pair()

        # Exploration
        self._exploration_rate_annealing_timescale = exploration_rate_annealing_timescale
        self._exploration_rate_init = exploration_rate_init
        self._exploration_rate_final = exploration_rate_final
        self._exploration_rate_annealing_duration = exploration_rate_annealing_duration

        # Re-seed the PRNG
        self._curr_seed = seed
        torch.manual_seed(seed)
        self._np_random, self._curr_seed = gym.utils.seeding.np_random(self._curr_seed)

        # Saving
        self._save_path = None

        # Statistics
        self._learn_calls = 0
        self._num_steps = 0
        self._curr_episode_num = 0

        # Internal state
        self._last_hidden_state = None
        self._last_labels = None
        self._episode_steps = []

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
        policy_optimizer = self._optimizer_cls(
            policy_network.parameters(),
            **self._optimizer_kws
        )

        lr_scheduler = ExponentialLR(policy_optimizer, gamma=1)  # 0.995)

        # Synch the weights from the policy network to the target network
        policy_state = policy_network.state_dict()
        target_network.load_state_dict(policy_state)

        # Disable computation for gradients for target network, as its weights
        # are not trained but copied periodically from the policy network
        target_network.requires_grad_(False)

        return DQRN.DQRNStatePolicy(policy_network, policy_optimizer, lr_scheduler, target_network)

    def _init_q_network(self):
        # TODO: fix obs_space being a dict
        return MinigridDRQN(
            self.obs_space["A1"].n,  # type: ignore
            self.num_observables,
            self.action_space.n,
            self._lstm_method,
            self._lstm_hidden_size,
            self._embedding_num_layers,
            self._embedding_layer_size,
            self._embedding_output_size,
        ).to(self.device)

    # Converts the state from the environment into a representation for a network
    def _embed_state(self, state):
        return torch.as_tensor(
            gym.spaces.flatten(self.obs_space, state),
            dtype=torch.float32,
            device=self.device,
        )

    def learn(self, state, _u, action, reward, done, next_state, _next_u, labels=None):
        self._learn_calls += 1

        flat_state = self._embed_state(state)
        flat_next_state = self._embed_state(next_state)

        reward = torch.as_tensor([reward], dtype=torch.float32, device=self.device)
        action = torch.as_tensor([action], device=self.device)

        t_labels = embed_labels(self._last_labels, self.num_observables)
        # Labels from the environment are associated with the next state
        t_next_labels = embed_labels(labels, self.num_observables)

        assert len(t_labels) == self.num_observables

        experience = DQRNStep(flat_state, t_labels, action, reward, done, flat_next_state, t_next_labels)
        self._episode_steps.append(experience)

        self._last_labels = labels

        if len(self._replay_memory) < self._er_start_size:
            return np.NAN

        self._policies_train_timer -= 1
        if self._policies_train_timer > 0:
            return np.NAN
        self._policies_train_timer = self._policy_train_freq

        if len(self._replay_memory) < self._er_batch_size:
            return np.NAN

        experience_batch = self._replay_memory.sample(self._er_batch_size, self._er_sequence_length)
        loss = self._update_policy_network(experience_batch)

        self._target_update_timer -= 1
        if self._target_update_timer == 0:
            self._update_target_network()
            self._target_update_timer = self._target_update_freq

        return loss

    def _update_target_network(self):
        """
        Update the of weights each target network by copying the weights of their associated policy networks.
        """
        q_net, _, _, t_net = self._q_networks
        t_net.load_state_dict(q_net.state_dict())

    def _update_policy_network(self, experience_batch: DQRNStep):
        """
        Adapted from https://github.com/ertsiger/hrm-learning/blob/main/src/reinforcement_learning/drqn_algorithm.py
        """
        # Translating names to ertsiger nomenclature
        _net = self._q_networks.policy_network
        _tgt_net = self._q_networks.target_network
        _optimizer = self._q_networks.policy_optimizer

        states, label_batch, actions, next_states, next_label_batch, rewards, is_terminal, = experience_batch

        def _make_padded_tensor(l, dtype):
            # Convert each member of the batch into a tensor and pad it
            sequences = list(map(lambda x: torch.stack(x), l))
            return pad_sequence(
                sequences,
                batch_first=True
            ).to(device=self.device)
            # return pad_sequence(
            #     list(map(lambda x: torch.tensor(np.array(x), dtype=dtype, device=self.device), l)),
            #     batch_first=True
            # )

        # Convert to arrays of tensors
        states_v = _make_padded_tensor(states, torch.float32)
        labels_v = _make_padded_tensor(label_batch, torch.float32)
        actions_v = _make_padded_tensor(actions, torch.int64)
        next_states_v = _make_padded_tensor(next_states, torch.float32)
        next_labels_v = _make_padded_tensor(next_label_batch, torch.float32)
        rewards_v = _make_padded_tensor(rewards, torch.float32)
        is_terminal_v = _make_padded_tensor(is_terminal, torch.bool)

        hidden_state = _net.get_zero_hidden_state(batch_size=self._er_batch_size, device=self.device)
        loss = torch.tensor(0.0, device=self.device)

        seq_lengths = torch.tensor(list(map(len, states)), dtype=torch.int, device=self.device)
        num_timesteps = seq_lengths.max()
        for timestep in range(num_timesteps):
            q_values, hidden_state = _net(states_v[:, timestep, :], labels_v[:, timestep, :], hidden_state)
            q_values = q_values.gather(1, actions_v[:, timestep]).squeeze(-1)

            with torch.no_grad():
                if self._use_double_dqn:
                    next_actions = _net(
                        next_states_v[:, timestep, :], next_labels_v[:, timestep, :], hidden_state
                    )[0].max(1)[1]
                    next_q_values = _tgt_net(
                        next_states_v[:, timestep, :], next_labels_v[:, timestep, :], hidden_state
                    )[0].gather(1, next_actions.unsqueeze(-1)).squeeze(-1)
                else:
                    next_q_values = _tgt_net(
                        next_states_v[:, timestep, :], next_labels_v[:, timestep, :], hidden_state
                    )[0].max(1)[0]
                next_q_values[is_terminal_v[:, timestep]] = 0.0

            expected_q_values = rewards_v[:, timestep] + self._gamma * next_q_values
            error = (q_values - expected_q_values) ** 2
            loss += torch.sum(error.masked_fill(timestep >= seq_lengths, 0))

        loss = loss / sum(seq_lengths)

        _optimizer.zero_grad()
        loss.backward()
        # The original DRQN paper used gradient clipping
        if self._use_gradient_clipping:
            torch.nn.utils.clip_grad_value_(_net.parameters(), clip_value=10)

        _optimizer.step()
        return loss.item()

    @property
    def epsilon(self):
        if self._exploration_rate_annealing_timescale == EpsilonAnnealingTimescale.STEPS:
            return get_annealed_exploration_rate(
                self._num_steps, self._exploration_rate_init, self._exploration_rate_final,
                self._exploration_rate_annealing_duration
            )
        elif self._exploration_rate_annealing_timescale == EpsilonAnnealingTimescale.EPISODES:
            return get_annealed_exploration_rate(
                self._curr_episode_num, self._exploration_rate_init, self._exploration_rate_final,
                self._exploration_rate_annealing_duration
            )
        raise RuntimeError(f"Error: Unknown timescale for exploration '{self._exploration_rate_annealing_timescale}'.")

    def on_env_reset(self):
        if self._curr_episode_num > max(self._er_batch_size, self._er_start_size) + 1:
            self._q_networks.lr_scheduler.step()

        self._last_hidden_state = self._q_networks.policy_network.get_zero_hidden_state(batch_size=1,
                                                                                        device=self.device)
        self._last_labels = None

        if self._episode_steps:
            self._replay_memory.append(self._episode_steps)
            self._episode_steps = []

        self._curr_episode_num += 1

    def on_rm_reset(self, rm, *args, **kwargs):
        # Doesn't use reward machines. Nothing needs to happen
        pass

    def action(self, state, u, greedy: bool = True, testing: bool = False, labels: dict = None, **kwargs):
        self._num_steps += 1

        state_v = self._embed_state(state).unsqueeze(0)
        # state_v = torch.tensor(np.array([state]), dtype=torch.float32, device=self.device)
        labels_v = embed_labels(labels, self.num_observables).float().to(self.device).unsqueeze(0)
        if not testing and np.random.uniform(low=0, high=1) <= self.epsilon:
            action, new_hidden_state = self._get_random_action_from_history(
                state_v, labels_v, self._last_hidden_state
            )
        else:
            action, new_hidden_state = self._get_greedy_action_from_history(state_v, labels_v,
                                                                            self._last_hidden_state)
        self._last_hidden_state = new_hidden_state
        return action

    def _get_greedy_action_from_history(self, state, labels, hidden_state):
        with torch.no_grad():
            q_values, new_hidden_state = self._q_networks.policy_network(state, labels,
                                                                         hidden_state)
        return randargmax(q_values.cpu().numpy()), new_hidden_state

    def _get_random_action_from_history(self, state, labels, hidden_state):
        action = self._np_random.choice(range(0, self.action_space.n))
        with torch.no_grad():
            # Even though a random action is chosen, we still need the new hidden state.
            _, new_hidden_state = self._q_networks.policy_network(state, labels, hidden_state)
        return action, new_hidden_state

    def __getstate__(self):
        """
        Pickle-related: return the serializable portion of the instances' state
        """

        state = self.__dict__.copy()

        # Since the policy and target networks are implemented via PyTorch, we want to
        # manually handle their loading and saving
        del state['_q_networks']
        # Similarly, we do not want to store the entire replay buffer
        del state['_replay_memory']

        # Manually store only the policy network parameters as they can be used
        # to also restore the target networks as well
        q_net, _, _, _ = self._q_networks
        file = os.path.join(self._save_path, f'policy_network.pth')
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
        self._replay_memory = EpisodeExperienceBuffer(self._buffer_size, self._curr_seed)

        # Restore the Q-networks based on the files found in the save path
        self._q_networks = self._init_q_network_pair()

        # Load the state dictionary
        full_path = os.path.join(self._save_path, 'policy_network.pth')
        subpolicy_state = torch.load(full_path, map_location=self.device)

        self._q_networks.policy_network.load_state_dict(subpolicy_state)

        # Finally, propagate all the parameters to the target networks
        self._update_target_network()

    def set_save_path(self, path, **kwargs):
        self._save_path = path

    def get_statistics(self):
        raise NotImplementedError("TODO")
