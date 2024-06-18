from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import gymnasium as gym
import ray
from gymnasium.spaces import Box, Dict
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.models.base import ACTOR, CRITIC, Encoder, Model
from ray.rllib.core.models.configs import (
    CNNEncoderConfig,
    MLPEncoderConfig,
    ModelConfig,
    RecurrentEncoderConfig,
    _framework_implemented,
)
from ray.rllib.core.models.torch.base import TorchModel
from ray.rllib.core.models.torch.encoder import TorchActorCriticEncoder, TorchMLPEncoder
from ray.rllib.core.models.torch.primitives import TorchMLP
from ray.rllib.core.rl_module import RLModule
from ray.rllib.models import MODEL_DEFAULTS as _MODEL_DEFAULTS
from ray.rllib.models.utils import (
    get_activation_fn,
    get_filter_config,
    get_initializer_fn,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()

MODEL_DEFAULTS = (_MODEL_DEFAULTS | {"shared_hidden_layers_indices": tuple(),
                                     "shared_hidden_layers_ref": "actor::shared_layers"})


class PPORMTorchRLModule(PPOTorchRLModule):
    pass


class PPORMLearningCatalog(PPOCatalog):

    def __init__(self, observation_space: gym.Space, action_space: gym.Space, model_config_dict: Dict):
        super().__init__(observation_space, action_space, model_config_dict)

        self.actor_critic_encoder_config = RMLearningActorCriticEncoderConfig(
            base_encoder_config=self._encoder_config,
            shared=self._model_config_dict["vf_share_layers"],
        )

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
            # input_space is a 1D Box
            if isinstance(observation_space, Box) and len(observation_space.shape) == 1:
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
            elif (
                    isinstance(observation_space, Box) and len(observation_space.shape) == 2
            ):
                # RLlib used to support 2D Box spaces by silently flattening them
                raise ValueError(
                    f"No default encoder config for obs space={observation_space},"
                    f" lstm={use_lstm} and attention={use_attention} found. 2D Box "
                    f"spaces are not supported. They should be either flattened to a "
                    f"1D Box space or enhanced to be a 3D box space."
                )
            # input_space is a possibly nested structure of spaces.
            else:
                # NestedModelConfig
                raise ValueError(
                    f"No default encoder config for obs space={observation_space},"
                    f" lstm={use_lstm} and attention={use_attention} found."
                )

        return encoder_config


class PPORMLearningSharedLayersCatalog(PPORMLearningCatalog):

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
            # input_space is a 1D Box
            if isinstance(observation_space, Box) and len(observation_space.shape) == 1:
                # In order to guarantee backward compatability with old configs,
                # we need to check if no latent dim was set and simply reuse the last
                # fcnet hidden dim for that purpose.
                if model_config_dict["encoder_latent_dim"]:
                    hidden_layer_dims = model_config_dict["fcnet_hiddens"]
                else:
                    hidden_layer_dims = model_config_dict["fcnet_hiddens"][:-1]
                encoder_config = RMLearningSharedMLPEncoderConfig(
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
                    shared_hidden_layers_indices=model_config_dict[
                        "shared_hidden_layers_indices"
                    ],
                    shared_hidden_layers_ref=model_config_dict[
                        "shared_hidden_layers_ref"
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
            elif (
                    isinstance(observation_space, Box) and len(observation_space.shape) == 2
            ):
                # RLlib used to support 2D Box spaces by silently flattening them
                raise ValueError(
                    f"No default encoder config for obs space={observation_space},"
                    f" lstm={use_lstm} and attention={use_attention} found. 2D Box "
                    f"spaces are not supported. They should be either flattened to a "
                    f"1D Box space or enhanced to be a 3D box space."
                )
            # input_space is a possibly nested structure of spaces.
            else:
                # NestedModelConfig
                raise ValueError(
                    f"No default encoder config for obs space={observation_space},"
                    f" lstm={use_lstm} and attention={use_attention} found."
                )

        return encoder_config

    @staticmethod
    def build_shared_layers_ref(model_config_dict) -> ray.ObjectRef:
        model_config_dict = MODEL_DEFAULTS | model_config_dict

        name = model_config_dict["shared_hidden_layers_ref"]

        _actor = SharedLayers.options(name=name, lifetime="detached").remote(
            hidden_layer_dims=model_config_dict["fcnet_hiddens"],
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
            ]
        )

        return model_config_dict["shared_hidden_layers_ref"]


@dataclass
class RMLearningActorCriticEncoderConfig(ModelConfig):
    """Configuration for an ActorCriticEncoder.

    The base encoder functions like other encoders in RLlib. It is wrapped by the
    ActorCriticEncoder to provides a shared encoder Model to use in RLModules that
    provides twofold outputs: one for the actor and one for the critic. See
    ModelConfig for usage details.

    Attributes:
        base_encoder_config: The configuration for the wrapped encoder(s).
        shared: Whether the base encoder is shared between the actor and critic.
    """

    base_encoder_config: ModelConfig = None
    shared: bool = True

    @_framework_implemented(tf2=False)
    def build(self, framework: str = "torch") -> "Encoder":
        if framework == "torch":
            if isinstance(self.base_encoder_config, RecurrentEncoderConfig):
                raise NotImplementedError("TorchStatefulActorCriticEncoder")
            else:
                return RMLearningTorchActorCriticEncoder(self)
        else:
            raise NotImplementedError(framework)


class RMLearningTorchActorCriticEncoder(TorchActorCriticEncoder):
    pass


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


@dataclass
class RMLearningSharedMLPEncoderConfig(MLPEncoderConfig):
    """Configuration for an MLP that acts as an encoder.
    """
    shared_hidden_layers_indices: tuple = tuple()
    shared_hidden_layers_ref: object = None

    @_framework_implemented(torch=True, tf2=False)
    def build(self, framework: str = "torch") -> "Encoder":
        self._validate(framework)

        if framework == "torch":
            return TorchRMLearningSharedMLPEncoder(self)
        else:
            raise NotImplementedError(framework)

    def _validate(self, framework: str):
        super()._validate(framework)

        if not all(
                (0 < i and i < len(self.hidden_layer_dims) + 1)  # we add the output layer (+1)
                for i in self.shared_hidden_layers_indices
        ):
            raise ValueError(
                "Trying to share a layer that does not exist"
            )


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


class TorchRMLearningSharedMLPEncoder(TorchMLPEncoder, TorchModel):
    def __init__(self, config: RMLearningSharedMLPEncoderConfig) -> None:
        TorchModel.__init__(self, config)
        Encoder.__init__(self, config)

        # Create the neural network.
        self.net = TorchRMLearningSharedMLP(
            input_dim=config.input_dims[0],
            hidden_layer_dims=config.hidden_layer_dims,
            shared_layers_indices=config.shared_hidden_layers_indices,
            shared_layers_ref=config.shared_hidden_layers_ref,
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


class TorchRMLearningSharedMLP(nn.Module):
    """A multi-layer perceptron with N dense layers.

    All layers (except for an optional additional extra output layer) share the same
    activation function, bias setup (use bias or not), and LayerNorm setup
    (use layer normalization or not).

    If `output_dim` (int) is not None, an additional, extra output dense layer is added,
    which might have its own activation function (e.g. "linear"). However, the output
    layer does NOT use layer normalization.
    """

    def __init__(
            self,
            *,
            input_dim: int,
            hidden_layer_dims: List[int],
            shared_layers_indices: tuple = tuple(),
            shared_layers_ref: object = None,
            hidden_layer_activation: Union[str, Callable] = "relu",
            hidden_layer_use_bias: bool = True,
            hidden_layer_use_layernorm: bool = False,
            hidden_layer_weights_initializer: Optional[Union[str, Callable]] = None,
            hidden_layer_weights_initializer_config: Optional[Union[str, Callable]] = None,
            hidden_layer_bias_initializer: Optional[Union[str, Callable]] = None,
            hidden_layer_bias_initializer_config: Optional[Dict] = None,
            output_dim: Optional[int] = None,
            output_use_bias: bool = True,
            output_activation: Union[str, Callable] = "linear",
            output_weights_initializer: Optional[Union[str, Callable]] = None,
            output_weights_initializer_config: Optional[Dict] = None,
            output_bias_initializer: Optional[Union[str, Callable]] = None,
            output_bias_initializer_config: Optional[Dict] = None,
    ):
        """Initialize a TorchMLP object.

        Args:
            input_dim: The input dimension of the network. Must not be None.
            hidden_layer_dims: The sizes of the hidden layers. If an empty list, only a
                single layer will be built of size `output_dim`.
            hidden_layer_use_layernorm: Whether to insert a LayerNormalization
                functionality in between each hidden layer's output and its activation.
            hidden_layer_use_bias: Whether to use bias on all dense layers (excluding
                the possible separate output layer).
            hidden_layer_activation: The activation function to use after each layer
                (except for the output). Either a torch.nn.[activation fn] callable or
                the name thereof, or an RLlib recognized activation name,
                e.g. "ReLU", "relu", "tanh", "SiLU", or "linear".
            hidden_layer_weights_initializer: The initializer function or class to use
                forweights initialization in the hidden layers. If `None` the default
                initializer of the respective dense layer is used. Note, only the
                in-place initializers, i.e. ending with an underscore "_" are allowed.
            hidden_layer_weights_initializer_config: Configuration to pass into the
                initializer defined in `hidden_layer_weights_initializer`.
            hidden_layer_bias_initializer: The initializer function or class to use for
                bias initialization in the hidden layers. If `None` the default
                initializer of the respective dense layer is used. Note, only the
                in-place initializers, i.e. ending with an underscore "_" are allowed.
            hidden_layer_bias_initializer_config: Configuration to pass into the
                initializer defined in `hidden_layer_bias_initializer`.
            output_dim: The output dimension of the network. If None, no specific output
                layer will be added and the last layer in the stack will have
                size=`hidden_layer_dims[-1]`.
            output_use_bias: Whether to use bias on the separate output layer,
                if any.
            output_activation: The activation function to use for the output layer
                (if any). Either a torch.nn.[activation fn] callable or
                the name thereof, or an RLlib recognized activation name,
                e.g. "ReLU", "relu", "tanh", "SiLU", or "linear".
            output_layer_weights_initializer: The initializer function or class to use
                for weights initialization in the output layers. If `None` the default
                initializer of the respective dense layer is used. Note, only the
                in-place initializers, i.e. ending with an underscore "_" are allowed.
            output_layer_weights_initializer_config: Configuration to pass into the
                initializer defined in `output_layer_weights_initializer`.
            output_layer_bias_initializer: The initializer function or class to use for
                bias initialization in the output layers. If `None` the default
                initializer of the respective dense layer is used. Note, only the
                in-place initializers, i.e. ending with an underscore "_" are allowed.
            output_layer_bias_initializer_config: Configuration to pass into the
                initializer defined in `output_layer_bias_initializer`.
        """
        super().__init__()
        assert input_dim > 0

        self.input_dim = input_dim

        hidden_activation = get_activation_fn(
            hidden_layer_activation, framework="torch"
        )
        hidden_weights_initializer = get_initializer_fn(
            hidden_layer_weights_initializer, framework="torch"
        )
        hidden_bias_initializer = get_initializer_fn(
            hidden_layer_bias_initializer, framework="torch"
        )
        output_weights_initializer = get_initializer_fn(
            output_weights_initializer, framework="torch"
        )
        output_bias_initializer = get_initializer_fn(
            output_bias_initializer, framework="torch"
        )

        if shared_layers_ref:
            shared_layers_actor = ray.get_actor(shared_layers_ref)
            get_shared_layer = lambda i: ray.get(
                shared_layers_actor.get_shared_layer.remote(i)
            )
        elif shared_layers_indices:
            raise RuntimeError("`get_shared_layer` must be provided")
        else:
            get_shared_layer = lambda i: None

        layers = []
        dims = (
                [self.input_dim]
                + list(hidden_layer_dims)
                + ([output_dim] if output_dim else [])
        )
        for i in range(0, len(dims) - 1):
            # Whether we are already processing the last (special) output layer.
            is_output_layer = output_dim is not None and i == len(dims) - 2

            if i in shared_layers_indices:
                layer = get_shared_layer(i)
            else:
                layer = nn.Linear(
                    dims[i],
                    dims[i + 1],
                    bias=output_use_bias if is_output_layer else hidden_layer_use_bias,
                )
                # Initialize layers, if necessary.
                if is_output_layer:
                    # Initialize output layer weigths if necessary.
                    if output_weights_initializer:
                        output_weights_initializer(
                            layer.weight, **output_weights_initializer_config or {}
                        )
                    # Initialize output layer bias if necessary.
                    if output_bias_initializer:
                        output_bias_initializer(
                            layer.bias, **output_bias_initializer_config or {}
                        )
                # Must be hidden.
                else:
                    # Initialize hidden layer weights if necessary.
                    if hidden_layer_weights_initializer:
                        hidden_weights_initializer(
                            layer.weight, **hidden_layer_weights_initializer_config or {}
                        )
                    # Initialize hidden layer bias if necessary.
                    if hidden_layer_bias_initializer:
                        hidden_bias_initializer(
                            layer.bias, **hidden_layer_bias_initializer_config or {}
                        )

            layers.append(layer)

            # We are still in the hidden layer section: Possibly add layernorm and
            # hidden activation.
            if not is_output_layer:
                # Insert a layer normalization in between layer's output and
                # the activation.
                if hidden_layer_use_layernorm:
                    # We use an epsilon of 0.001 here to mimick the Tf default behavior.
                    layers.append(nn.LayerNorm(dims[i + 1], eps=0.001))
                # Add the activation function.
                if hidden_activation is not None:
                    layers.append(hidden_activation())

        # Add output layer's (if any) activation.
        output_activation = get_activation_fn(output_activation, framework="torch")
        if output_dim is not None and output_activation is not None:
            layers.append(output_activation())

        self.mlp = nn.Sequential(*layers)

        self.expected_input_dtype = torch.float32

    def forward(self, x):
        return self.mlp(x.type(self.expected_input_dtype))


@ray.remote
class SharedLayers:
    def __init__(
            self,
            hidden_layer_dims: List[int],
            hidden_layer_activation: Union[str, Callable] = "relu",
            hidden_layer_use_bias: bool = True,
            hidden_layer_use_layernorm: bool = False,
            hidden_layer_weights_initializer: Optional[Union[str, Callable]] = None,
            hidden_layer_weights_initializer_config: Optional[Union[str, Callable]] = None,
            hidden_layer_bias_initializer: Optional[Union[str, Callable]] = None,
            hidden_layer_bias_initializer_config: Optional[Dict] = None,
    ):

        hidden_activation = get_activation_fn(
            hidden_layer_activation, framework="torch"
        )
        hidden_weights_initializer = get_initializer_fn(
            hidden_layer_weights_initializer, framework="torch"
        )
        hidden_bias_initializer = get_initializer_fn(
            hidden_layer_bias_initializer, framework="torch"
        )

        layers = []
        for i in range(0, len(hidden_layer_dims) - 1):

            layer = nn.Linear(
                hidden_layer_dims[i],
                hidden_layer_dims[i + 1],
                bias=hidden_layer_use_bias,
            )

            if hidden_layer_weights_initializer:
                hidden_weights_initializer(
                    layer.weight, **hidden_layer_weights_initializer_config or {}
                )
            if hidden_layer_bias_initializer:
                hidden_bias_initializer(
                    layer.bias, **hidden_layer_bias_initializer_config or {}
                )

            layers.append(layer)

            if hidden_layer_use_layernorm:
                # We use an epsilon of 0.001 here to mimick the Tf default behavior.
                layers.append(nn.LayerNorm(hidden_layer_dims[i], eps=0.001))
            if hidden_activation is not None:
                layers.append(hidden_activation())

        self._layers = layers

    def get_shared_layer(self, i):
        return self._layers[i - 1]