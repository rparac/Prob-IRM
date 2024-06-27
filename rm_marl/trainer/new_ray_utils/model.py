from dataclasses import dataclass
from typing import Dict

from gymnasium.spaces import Box, Discrete
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.base import Encoder
from ray.rllib.core.models.configs import ModelConfig, RecurrentEncoderConfig, CNNEncoderConfig, MLPEncoderConfig, \
    _framework_implemented
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.encoder import TorchMLPEncoder
from ray.rllib.core.models.torch.primitives import TorchMLP
from ray.rllib.models import MODEL_DEFAULTS as _MODEL_DEFAULTS
from ray.rllib.models.utils import (
    get_filter_config,
)
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

SHARED_HIDDEN_LAYERS_REF = "actor::shared_layers"
MODEL_DEFAULTS = (_MODEL_DEFAULTS | {"shared_hidden_layers_indices": tuple(),
                                     "shared_hidden_layers_ref": SHARED_HIDDEN_LAYERS_REF})

import gymnasium as gym


class PPORMLearningCatalog(PPOCatalog):
    """
    We are implementing the catalog instead of the RLModuleDirectly as we
    want to use the shared encoder (eventually).
    Also, we want to use the default model_config_dict
    """

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, model_config_dict: Dict):
        super().__init__(observation_space, action_space, model_config_dict)

    @classmethod
    def _get_encoder_config(
            cls,
            observation_space: gym.Space,
            model_config_dict: dict,
            action_space: gym.Space = None,
            view_requirements=None,
    ) -> ModelConfig:
        """Returns an EncoderConfig for the given input_space and model_config_dict.

        Encoders are usually used in RLModules to transform the input space into a
        latent space that is then fed to the heads. The returned EncoderConfig
        objects correspond to the built-in Encoder classes in RLlib.
        For example, for a simple 1D-Box input_space, RLlib offers an
        MLPEncoder, hence this method returns the MLPEncoderConfig. You can overwrite
        this method to produce specific EncoderConfigs for your custom Models.

        The following input spaces lead to the following configs:
        - 1D-Box: MLPEncoderConfig
        - 3D-Box: CNNEncoderConfig
        # TODO (Artur): Support more spaces here
        # ...

        Args:
            observation_space: The observation space to use.
            model_config_dict: The model config to use.
            action_space: The action space to use if actions are to be encoded. This
                is commonly the case for LSTM models.
            view_requirements: The view requirements to use if anything else than
                observation_space or action_space is to be encoded. This signifies an
                advanced use case.

        Returns:
            The encoder config.
        """
        # TODO (Artur): Make it so that we don't work with complete MODEL_DEFAULTS
        model_config_dict = {**MODEL_DEFAULTS, **model_config_dict}

        fcnet_hiddens = model_config_dict["fcnet_hiddens"]
        # TODO (sven): Move to a new ModelConfig object (dataclass) asap, instead of
        #  "linking" into the old ModelConfig (dict)! This just causes confusion as to
        #  which old keys now mean what for the new RLModules-based default models.
        encoder_latent_dim = (
                model_config_dict["encoder_latent_dim"] or fcnet_hiddens[-1]
        )
        use_lstm = model_config_dict["use_lstm"]
        use_attention = model_config_dict["use_attention"]

        if use_lstm:
            encoder_config = RecurrentEncoderConfig(
                input_dims=observation_space.shape,
                recurrent_layer_type="lstm",
                hidden_dim=model_config_dict["lstm_cell_size"],
                hidden_weights_initializer=model_config_dict[
                    "lstm_weights_initializer"
                ],
                hidden_weights_initializer_config=model_config_dict[
                    "lstm_weights_initializer_config"
                ],
                hidden_bias_initializer=model_config_dict["lstm_bias_initializer"],
                hidden_bias_initializer_config=model_config_dict[
                    "lstm_bias_initializer_config"
                ],
                batch_major=not model_config_dict["_time_major"],
                num_layers=1,
                tokenizer_config=cls.get_tokenizer_config(
                    observation_space,
                    model_config_dict,
                    view_requirements,
                ),
            )
        elif use_attention:
            raise NotImplementedError
        else:
            # TODO (Artur): Maybe check for original spaces here
            if isinstance(observation_space, Discrete):
                # In order to guarantee backward compatability with old configs,
                # we need to check if no latent dim was set and simply reuse the last
                # fcnet hidden dim for that purpose.
                if model_config_dict["encoder_latent_dim"]:
                    hidden_layer_dims = model_config_dict["fcnet_hiddens"]
                else:
                    hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
                encoder_config = RMLearningMLPEncoderConfig(
                    # input_dims=(int(observation_space.n),),
                    input_dims=(1,),
                    hidden_layer_dims=hidden_layer_dims,
                    hidden_layer_activation=model_config_dict["fcnet_activation"],
                    hidden_layer_weights_initializer=model_config_dict[
                        "fcnet_weights_initializer"
                    ],
                    hidden_layer_weights_initializer_config=model_config_dict[
                        "fcnet_weights_initializer_config"
                    ],
                    hidden_layer_bias_initializer=model_config_dict[
                        "fcnet_bias_initializer"
                    ],
                    hidden_layer_bias_initializer_config=model_config_dict[
                        "fcnet_bias_initializer_config"
                    ],
                    output_layer_dim=encoder_latent_dim,
                    output_layer_activation=model_config_dict["fcnet_activation"],
                    output_layer_weights_initializer=model_config_dict[
                        "post_fcnet_weights_initializer"
                    ],
                    output_layer_weights_initializer_config=model_config_dict[
                        "post_fcnet_weights_initializer_config"
                    ],
                    output_layer_bias_initializer=model_config_dict[
                        "post_fcnet_bias_initializer"
                    ],
                    output_layer_bias_initializer_config=model_config_dict[
                        "post_fcnet_bias_initializer_config"
                    ],
                )
            elif isinstance(observation_space, Box) and len(observation_space.shape) == 1:
               # In order to guarantee backward compatability with old configs,
               # we need to check if no latent dim was set and simply reuse the last
               # fcnet hidden dim for that purpose.
               if model_config_dict["encoder_latent_dim"]:
                   hidden_layer_dims = model_config_dict["fcnet_hiddens"]
               else:
                   hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
               encoder_config = RMLearningMLPEncoderConfig(
                   input_dims=observation_space.shape,
                   hidden_layer_dims=hidden_layer_dims,
                   hidden_layer_activation=model_config_dict["fcnet_activation"],
                   hidden_layer_weights_initializer=model_config_dict[
                       "fcnet_weights_initializer"
                   ],
                   hidden_layer_weights_initializer_config=model_config_dict[
                       "fcnet_weights_initializer_config"
                   ],
                   hidden_layer_bias_initializer=model_config_dict[
                       "fcnet_bias_initializer"
                   ],
                   hidden_layer_bias_initializer_config=model_config_dict[
                       "fcnet_bias_initializer_config"
                   ],
                   output_layer_dim=encoder_latent_dim,
                   output_layer_activation=model_config_dict["fcnet_activation"],
                   output_layer_weights_initializer=model_config_dict[
                       "post_fcnet_weights_initializer"
                   ],
                   output_layer_weights_initializer_config=model_config_dict[
                       "post_fcnet_weights_initializer_config"
                   ],
                   output_layer_bias_initializer=model_config_dict[
                       "post_fcnet_bias_initializer"
                   ],
                   output_layer_bias_initializer_config=model_config_dict[
                       "post_fcnet_bias_initializer_config"
                   ],
               )

            # input_space is a 3D Box
            elif (
                isinstance(observation_space, Box) and len(observation_space.shape) == 3
            ):
                if not model_config_dict.get("conv_filters"):
                    model_config_dict["conv_filters"] = get_filter_config(
                        observation_space.shape
                    )

                encoder_config = CNNEncoderConfig(
                    input_dims=observation_space.shape,
                    cnn_filter_specifiers=model_config_dict["conv_filters"],
                    cnn_activation=model_config_dict["conv_activation"],
                    cnn_use_layernorm=model_config_dict.get(
                        "conv_use_layernorm", False
                    ),
                    cnn_kernel_initializer=model_config_dict["conv_kernel_initializer"],
                    cnn_kernel_initializer_config=model_config_dict[
                        "conv_kernel_initializer_config"
                    ],
                    cnn_bias_initializer=model_config_dict["conv_bias_initializer"],
                    cnn_bias_initializer_config=model_config_dict[
                        "conv_bias_initializer_config"
                    ],
                )
            # input_space is a 2D Box
            else:
                # NestedModelConfig
                raise ValueError(
                    f"No default encoder config for obs space={observation_space},"
                    f" lstm={use_lstm} and attention={use_attention} found."
                )

        return encoder_config


@dataclass
class RMLearningMLPEncoderConfig(MLPEncoderConfig):
    """Configuration for an MLP that acts as an encoder.
    """

    @_framework_implemented(torch=True, tf2=False)
    def build(self, framework: str = "torch") -> "Encoder":
        self._validate(framework)

        if framework == "torch":
            return TorchRMLearningMLPEncoder(self)
        else:
            raise NotImplementedError(framework)


class TorchRMLearningMLPEncoder(TorchMLPEncoder, TorchModel):
    def __init__(self, config: RMLearningMLPEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        # Create the neural network.
        self.net = TorchMLP(
            input_dim=config.input_dims[0],
            hidden_layer_dims=config.hidden_layer_dims,
            hidden_layer_activation=config.hidden_layer_activation,
            hidden_layer_use_layernorm=config.hidden_layer_use_layernorm,
            hidden_layer_use_bias=config.hidden_layer_use_bias,
            hidden_layer_weights_initializer=config.hidden_layer_weights_initializer,
            hidden_layer_weights_initializer_config=(
                config.hidden_layer_weights_initializer_config
            ),
            hidden_layer_bias_initializer=config.hidden_layer_bias_initializer,
            hidden_layer_bias_initializer_config=(
                config.hidden_layer_bias_initializer_config
            ),
            output_dim=config.output_layer_dim,
            output_activation=config.output_layer_activation,
            output_use_bias=config.output_layer_use_bias,
            output_weights_initializer=config.output_layer_weights_initializer,
            output_weights_initializer_config=(
                config.output_layer_weights_initializer_config
            ),
            output_bias_initializer=config.output_layer_bias_initializer,
            output_bias_initializer_config=config.output_layer_bias_initializer_config,
        )
