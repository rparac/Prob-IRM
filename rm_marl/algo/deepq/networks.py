import torch
from torch import nn


class OfficeQNetwork(nn.Module):

    def __init__(self, dim_obs, max_rm_states, num_actions=4, dropout=False):
        super(OfficeQNetwork, self).__init__()

        embedding_size = 3
        self._obs_embedding = nn.Linear(dim_obs, embedding_size)

        base_layer_size = embedding_size + max_rm_states
        hidden_layer_1 = nn.Linear(base_layer_size, base_layer_size)
        hidden_layer_2 = nn.Linear(base_layer_size, 2 * base_layer_size)
        hidden_layer_3 = nn.Linear(2 * base_layer_size, base_layer_size)

        if not dropout:
            self.network = nn.Sequential(
                hidden_layer_1,
                nn.ReLU(),
                hidden_layer_2,
                nn.ReLU(),
                hidden_layer_3,
                nn.ReLU(),
                nn.Linear(base_layer_size, num_actions)
            )
        else:
            self.network = nn.Sequential(
                hidden_layer_1,
                nn.ReLU(),
                nn.Dropout(0.2),
                hidden_layer_2,
                nn.ReLU(),
                nn.Dropout(0.2),
                hidden_layer_3,
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(base_layer_size, num_actions)
            )

        # Initialize hidden layers using He initialization
        torch.nn.init.kaiming_uniform_(hidden_layer_1.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(hidden_layer_2.weight, nonlinearity='relu')
        torch.nn.init.kaiming_uniform_(hidden_layer_3.weight, nonlinearity='relu')

    def forward(self, s, u):

        obs_embeddings = self._obs_embedding(s)
        rm_state_belief = (100 * u).round_()

        # Normalize beliefs in [-1, 1]
        rm_state_belief -= rm_state_belief.min()
        rm_state_belief /= rm_state_belief.max()
        rm_state_belief = 2 * rm_state_belief - 1

        network_input = torch.cat((obs_embeddings, rm_state_belief), dim=-1)
        return self.network(network_input)


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


class SimpleQNetwork(nn.Module):

    def __init__(self, dim_obs, max_rm_states, num_actions=4, *, learn_biases=False):
        super(SimpleQNetwork, self).__init__()

        self._q_table = nn.Linear(dim_obs + max_rm_states, num_actions, bias=learn_biases)
        torch.nn.init.zeros_(self._q_table.weight)

    def forward(self, s, u):

        quantized_belief = (100 * u).round_()
        table_entry = torch.cat((s, quantized_belief), dim=-1)
        return self._q_table(table_entry)


class CRMNetwork(nn.Module):

    def __init__(self, dim_obs, num_rm_states, num_layers, layer_size, num_actions):
        super(CRMNetwork, self).__init__()

        self._model = DeepQNetwork(dim_obs + num_rm_states, num_layers, layer_size, num_actions)

    def forward(self, s, u):
        x = torch.cat((s, u), dim=-1)
        return self._model(x, x)
