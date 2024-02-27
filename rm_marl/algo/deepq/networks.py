import torch
from torch import nn


class DeepQNetwork(nn.Module):

    def __init__(self, dim_obs, num_layers, layer_size, num_actions):
        super(DeepQNetwork, self).__init__()

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

    # u is intentionally ignored; by design, there is a separate network for each RM state
    def forward(self, s, u):
        return self._model(s)


class CRMNetwork(nn.Module):

    def __init__(self, dim_obs, num_rm_states, num_layers, layer_size, num_actions):
        super(CRMNetwork, self).__init__()

        self._model = DeepQNetwork(dim_obs + num_rm_states, num_layers, layer_size, num_actions)

    def forward(self, s, u):
        x = torch.cat((s, u), dim=-1)
        return self._model(x, x)
