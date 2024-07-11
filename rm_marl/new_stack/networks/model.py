from dataclasses import dataclass
from typing import Optional

from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import Encoder, Model, ENCODER_OUT
from ray.rllib.core.models.configs import ModelConfig, MLPEncoderConfig, \
    _framework_implemented
from ray.rllib.core.models.specs.specs_base import Spec, TensorSpec
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.encoder import TorchMLPEncoder, TorchActorCriticEncoder
from ray.rllib.core.models.torch.primitives import TorchMLP
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.models import MODEL_DEFAULTS as _MODEL_DEFAULTS, MODEL_DEFAULTS
from ray.rllib.utils.framework import try_import_torch

import gymnasium as gym

from rm_marl.new_stack.connectors.RM_state_connector import RMStateConnector
from src.ray.rllib.core.columns import Columns
from src.ray.rllib.core.models.specs.specs_dict import SpecDict
from src.ray.rllib.utils.annotations import override

torch, nn = try_import_torch()


class PPORMTorchRLModule(TorchRLModule):
    """
    Implement this if we need to deal with custom RL module
    https://github.com/ray-project/ray/blob/bcbdcf2906408f3bac1f5a6af92e1e000bb94f59/rllib/examples/rl_modules/classes/tiny_atari_cnn_rlm.py#L16
    """
    pass


class PPORMLearningCatalog(PPOCatalog):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, model_config_dict: dict):
        super().__init__(observation_space, action_space, model_config_dict)

    @classmethod
    def _get_encoder_config(
            cls,
            observation_space: gym.Space,
            model_config_dict: dict,
            **kwargs,
            # model_config_dict: dict,
            # action_space: gym.Space = None,
            # view_requirements=None,
    ) -> ModelConfig:
        # Subgoal automata space is converted automatically to Dict with
        #   (Box(0,1,int)) -observation and rm_state
        # TODO: fix the observation space, it is not actually a Dict I think?
        #   It definitely does not have the ORIG_OBS_KEY
        if not isinstance(observation_space, gym.spaces.Dict):
            return super()._get_encoder_config(observation_space, **kwargs)

        return RMLearningMLPEncoderConfig(
            input_dims=observation_space[RMStateConnector.ORIG_OBS_KEY].shape,
            num_rm_states=observation_space[RMStateConnector.RM_STATE_KEY].shape[0],
        )


@dataclass
class RMLearningMLPEncoderConfig(MLPEncoderConfig):
    """
    Configuration for an MLP that creates an encoding based on
    observation and RM state
    """
    # latent space size
    num_rm_states: int = None

    @_framework_implemented(torch=True, tf2=False)
    def build(self, framework: str = "torch") -> "Encoder":
        self._validate(framework)

        if framework == "torch":
            return TorchRMLearningMLPEncoder(self)
        else:
            raise NotImplementedError(framework)


class TorchRMLearningMLPEncoder(TorchMLPEncoder, Encoder):
    def __init__(self, config: RMLearningMLPEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        self.config = config

        # Create the neural network.
        # TODO: use hidden_layer_dims
        self.net = nn.Linear(config.input_dims[0] + config.num_rm_states, config.output_dims[0])

    """
    We require both observation and RM state for the encoder
    """

    @override(Model)
    def get_input_specs(self) -> Optional[Spec]:
        return SpecDict(
            {
                Columns.OBS: TensorSpec(
                    "b, d", d=self.config.input_dims[0], framework="torch"
                ),
                RMStateConnector.RM_STATE_KEY: TensorSpec(
                    "b, d", d=self.config.num_rm_states, framework="torch"
                )
            }
        )

    @override(Model)
    def _forward(self, inputs: dict, **kwargs) -> dict:
        x = torch.cat((inputs[Columns.OBS], inputs[RMStateConnector.RM_STATE_KEY]), dim=1)
        out = self.net(x)
        return {
            ENCODER_OUT: out
        }
