from dataclasses import dataclass

import torch
from ray.rllib.core.models.base import ENCODER_OUT
from ray.rllib.core.models.configs import MLPEncoderConfig, _framework_implemented
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.encoder import TorchMLPEncoder

import torch.nn as nn


# TODO: this should probably not be an MLPEncoder
@dataclass
class ObsAndRMStateEncoderConfig(MLPEncoderConfig):
    """
    Configuration for ObsAndRMEncoder
    """

    @_framework_implemented(torch=True, tf2=False)
    def build(self, framework: str = "torch") -> "Encoder":
        self._validate(framework)

        if framework == "torch":
            return ObsAndRMStateEncoder(self)
        else:
            raise NotImplementedError(framework)


class ObsAndRMStateEncoder(TorchMLPEncoder, TorchModel):
    """

    """

    def __init__(self, config: ObsAndRMStateEncoderConfig):
        super().__init__(config)

        # TODO: design a network
        self.net = nn.Sequential(

        )

    def _forward(self, inputs: dict, **kwargs) -> dict:
        tensor_input = torch.cat((inputs["obs"], inputs["rm_state"]), dim=1)
        return {ENCODER_OUT: (self.net(tensor_input))}
