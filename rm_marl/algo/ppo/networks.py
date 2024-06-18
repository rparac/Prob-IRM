"""
Starting with the configuration from Icarte's Minigrid experiments
"""
import torch
from torch import nn


# Toro Icarte's group encoding
class ObservationEncoder(nn.Module):
    def __init__(self):
        super(ObservationEncoder, self).__init__()

        # TODO: needs to match the observation + rm_belief size
        self.input_size = -1
        # From Icarte
        self.hidden_size = 64
        self.output_size = 64
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size),
        )

    def forward(self, current_obs, rm_belief):
        # Skip the batch dimension
        x = torch.stack([current_obs, rm_belief], dim=1)
        return self.net(x)


class HistoryEncoder(nn.Module):
    def __init__(self):
        super(HistoryEncoder, self).__init__()

        # Should match the ObservationEncoder output
        self.input_size = 64
        # Toro Icarte information
        self.hidden_size = 64

        self.obs_encoder = ObservationEncoder()
        self.rec = nn.GRUCell(self.input_size, self.hidden_size)

    def forward(self, current_obs, rm_belief, hidden_state):
        x = self.obs_encoder(current_obs, rm_belief)
        x = self.rec(x, hidden_state)
        return x


class ActorNetwork(nn.Module):
    def __init__(self):
        super(ActorNetwork, self).__init__()

        # TODO
        self.input_size = -1

        # From Toro Icarte group
        self.hidden_size = 64

        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

    def forward(self):
        raise NotImplementedError


class CriticNetwork(nn.Module):
    def __init__(self):
        super(CriticNetwork, self).__init__()

        self.input_size = -1

        self.hidden_size = 64
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU()
        )

    def forward(self):
        raise NotImplementedError
