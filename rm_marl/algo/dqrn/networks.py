"""
This file is taken from https://github.com/ertsiger/hrm-learning/blob/main/src/reinforcement_learning/model.py
"""
import torch
from torch import nn
import torch.nn.functional as F


class MinigridConv(nn.Module):
    """
    Convolutional neural network formed by 3 convolutional layers. This network is based on that of the following
    resources:
      - "Prioritized Level Replay" by Minqi Jiang, Edward Grefenstette, Tim Rocktäschel (2021).
        Code: https://github.com/facebookresearch/level-replay. The exact file is the following:
        https://github.com/facebookresearch/level-replay/blob/main/level_replay/model.py#L413
      - rl-starter-files GitHub repo (https://github.com/lcswillems/rl-starter-files). The exact file is the following:
        https://github.com/lcswillems/rl-starter-files/blob/master/model.py
    In the first source, they also used the 'full' observation provided by Minigrid although they don't use a DQN but
    PPO. The second source contains a very similar network to the one shown in the first, also used for PPO.
    """
    DEFAULT_IN_CHANNELS = 3
    KERNEL_SIZE = (2, 2)

    def __init__(self, obs_shape, num_out_channels=(16, 32, 32), use_max_pool=False):
        num_conv_layers = len(num_out_channels)

        assert (obs_shape[0] == MinigridConv.DEFAULT_IN_CHANNELS) or (
                obs_shape[0] == MinigridConv.DEFAULT_IN_CHANNELS - 1), \
            f"Error: Minigrid observations must consist of {MinigridConv.DEFAULT_IN_CHANNELS} matrices if colors are " \
            f"not removed, or {MinigridConv.DEFAULT_IN_CHANNELS - 1} if they are."
        assert num_conv_layers >= 1 and num_conv_layers <= 3, \
            "Error: The number of convolutional layers must be between 1 and 3."

        super(MinigridConv, self).__init__()

        self.conv1 = self._make_conv(obs_shape[0], num_out_channels[0])
        self.conv2 = self._make_conv(num_out_channels[0], num_out_channels[1]) if num_conv_layers >= 2 else None
        self.conv3 = self._make_conv(num_out_channels[1], num_out_channels[2]) if num_conv_layers == 3 else None
        self.use_max_pool = use_max_pool

        n, m = obs_shape[-2], obs_shape[-1]
        res_n, res_m, res_channels = n - 1, m - 1, num_out_channels[0]  # first convolution
        if self.use_max_pool:
            res_n, res_m = res_n // 2, res_m // 2
        if self.conv2 is not None:
            res_n, res_m, res_channels = res_n - 1, res_m - 1, num_out_channels[1]
        if self.conv3 is not None:
            res_n, res_m, res_channels = res_n - 1, res_m - 1, num_out_channels[2]
        self.embedding_size = res_n * res_m * res_channels

    def _make_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, MinigridConv.KERNEL_SIZE),
            nn.ReLU()
        )

    def get_embedding_size(self):
        return self.embedding_size

    def forward(self, obs):
        # Tip: follow advice in https://stackoverflow.com/questions/55667005/manage-memory-differently-on-train-and-test-time-pytorch
        out = self.conv1(obs)
        if self.use_max_pool:
            out = F.max_pool2d(out, MinigridConv.KERNEL_SIZE)
        if self.conv2 is not None:
            out = self.conv2(out)
        if self.conv3 is not None:
            out = self.conv3(out)
        out = out.flatten(1, -1)
        return out


class MinigridDRQN(nn.Module):
    """
    DRQN network for Minigrid. The architecture is based on that used in our hierarchical approach: same convolutional
    network and LSTM layer with the same size as the linear layer used in our approach.
    """

    def __init__(self, obs_shape, num_observables, num_actions, lstm_method, hidden_size=256, use_max_pool=False):
        super(MinigridDRQN, self).__init__()

        self.obs_shape = obs_shape
        self.lstm_method = lstm_method
        self.hidden_size = hidden_size

        self.conv = MinigridConv(
            obs_shape,
            (16, 32, 32),
            use_max_pool
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
