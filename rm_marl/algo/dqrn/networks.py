"""
This file is taken from https://github.com/ertsiger/hrm-learning/blob/main/src/reinforcement_learning/model.py
"""
import torch
from torch import nn
import torch.nn.functional as F

from rm_marl.algo.deepq.networks import DeepQNetwork


class OfficeWorldEmbedding(nn.Module):
    def __init__(self, state_size, num_layers, layer_size, embedding_dim):
        super(OfficeWorldEmbedding, self).__init__()

        self.input_size = state_size
        self.embedding_dim = embedding_dim
        self._input_layer = nn.Sequential(
            nn.Linear(state_size, layer_size),
            nn.ReLU()
        )

        self._hidden_layers = nn.Sequential()
        for i in range(num_layers):
            self._hidden_layers.append(nn.Linear(layer_size, layer_size))
            self._hidden_layers.append(nn.ReLU())

        self._output_layer = nn.Linear(layer_size, embedding_dim)

        self._model = nn.Sequential(
            self._input_layer,
            self._hidden_layers,
            self._output_layer
        )

    def forward(self, x):
        # convert to 1-hot vector
        return self._model(x)

    def get_embedding_size(self):
        return self.embedding_dim


class OfficeWorldConv(nn.Module):

    def forward(self, s):
        # u is ignored in DeepQNetwork, but it requires the argument
        return self.network(s, None)

    def get_embedding_size(self):
        return self.dim_obs


class MinigridDRQN(nn.Module):
    """
    DRQN network for Minigrid. The architecture is based on that used in our hierarchical approach: same convolutional
    network and LSTM layer with the same size as the linear layer used in our approach.
    """

    def __init__(self, obs_shape, num_observables, num_actions, lstm_method, hidden_size, embedding_num_layers=2,
                 embedding_layer_size=16, embedding_output_size=8):
        super(MinigridDRQN, self).__init__()

        self.obs_shape = obs_shape
        self.lstm_method = lstm_method
        self.hidden_size = hidden_size

        # TODO: parameterize
        self.conv = OfficeWorldEmbedding(
            obs_shape,
            num_layers=embedding_num_layers,
            layer_size=embedding_layer_size,
            embedding_dim=embedding_output_size,
        )

        if lstm_method == "state":
            lstm_in_size = self.conv.get_embedding_size()
            linear_in_size = self.hidden_size
        elif lstm_method == "state+obs":
            lstm_in_size = self.conv.get_embedding_size() + num_observables
            linear_in_size = self.hidden_size
        elif lstm_method == "obs":
            lstm_in_size = num_observables
            linear_in_size = self.conv.get_embedding_size() + self.hidden_size
        else:
            raise RuntimeError(f"Error: Unknown method for using the LSTM memory '{lstm_method}'.")

        self.lstm_cell = nn.LSTMCell(
            input_size=lstm_in_size,
            hidden_size=hidden_size
        )
        # the input to the LSTM is a transformation of the state obs. concatenated with the set of seen propositions
        LSTM_METHOD_OBS = "obs"  # the input to the LSTM is the set of observed propositions, which is then concatenated with the processed state observation
        self.linear = nn.Linear(linear_in_size, num_actions)

    def get_zero_hidden_state(self, batch_size, device):
        return (
            torch.zeros((batch_size, self.hidden_size), dtype=torch.float32, device=device),
            torch.zeros((batch_size, self.hidden_size), dtype=torch.float32, device=device)
        )

    def forward(self, obs, observables, hidden_state):
        conv_out = self.conv(obs)

        if self.lstm_method == "state":
            lstm_hidden, lstm_cell = self._forward_lstm(conv_out, hidden_state)
            linear_in = lstm_hidden
        elif self.lstm_method == "state+obs":
            lstm_hidden, lstm_cell = self._forward_lstm(
                torch.concat((conv_out, observables), dim=1),
                hidden_state
            )
            linear_in = lstm_hidden
        elif self.lstm_method == "obs":
            lstm_hidden, lstm_cell = self._forward_lstm(
                observables, hidden_state
            )
            linear_in = torch.concat((conv_out, lstm_hidden), dim=1)
        else:
            raise RuntimeError(f"Error: Unknown method for using the LSTM memory '{self.lstm_method}'.")

        # Take the LSTM output of each step in the sequence and pass it to the linear layer to output the Q-values for
        # each step.
        q_vals = self.linear(linear_in)

        return q_vals, (lstm_hidden, lstm_cell)

    def _forward_lstm(self, in_seq, hidden_state):
        return self.lstm_cell(in_seq, hidden_state)
