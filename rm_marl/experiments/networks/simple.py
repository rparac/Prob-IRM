from typing import Any, Dict, Optional

import numpy as np

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import TensorType

torch, nn = try_import_torch()


class SimpleNN(TorchRLModule, ValueFunctionAPI):
    @override(TorchRLModule)
    def setup(self):
        """Use this method to create all the model components that you require.

        Feel free to access the following useful properties in this class:
        - `self.model_config`: The config dict for this RLModule class,
        which should contain flxeible settings, for example: {"hiddens": [256, 256]}.
        - `self.observation|action_space`: The observation and action space that
        this RLModule is subject to. Note that the observation space might not be the
        exact space from your env, but that it might have already gone through
        preprocessing through a connector pipeline (for example, flattening,
        frame-stacking, mean/std-filtering, etc..).
        """

        # assert self.model_config["fcnet_activation"] == "relu"
        # assert self.model_config["fcnet_weights_initializer"] == "orthogonal_"
        # assert self.model_config["vf_share_layers"]

        self._num_labels = self.model_config.get("num_labels")

        # Assume a simple Box(1D) tensor as input shape.
        in_size = self.observation_space.shape[0]

        coffee_pred_layers = [nn.Linear(in_size, 1), nn.Sigmoid()]
        self._coffee_predictor = nn.Sequential(*coffee_pred_layers)

        self._curr_val = nn.Parameter(torch.tensor(0.0))

        # Build a sequential stack.
        layers = []

        # +1 as it takes the "memory" parameter
        in_size = in_size + 1
        dense_layers = self.model_config.get("fcnet_hiddens")
        for out_size in dense_layers:
            layer = nn.Linear(in_size, out_size)
            nn.init.orthogonal_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
            # Dense layer.
            layers.append(layer)
            # ReLU activation.
            layers.append(nn.ReLU())
            in_size = out_size

        self._fc_net = nn.Sequential(*layers)

        # Logits layer (no bias, no activation).
        self._pi_head = nn.Linear(in_size, self.action_space.n)
        # Single-node value layer.
        self._values = nn.Linear(in_size, 1)

    def predict_label(self, obs):
        return self._coffee_predictor(obs)

    @override(TorchRLModule)
    def get_initial_state(self) -> Any:
        # TODO: can we convert this to tensor
        return {
            "h": np.zeros(shape=(1,), dtype=np.float32),
        }

    @override(TorchRLModule)
    def _forward(self, batch, **kwargs):
        # Compute the basic 1D embedding tensor (inputs to policy- and value-heads).
        embeddings, state_outs = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)

        # Return logits as ACTION_DIST_INPUTS (categorical distribution).
        # Note that the default `GetActions` connector piece (in the EnvRunner) will
        # take care of argmax-"sampling" from the logits to yield the inference (greedy)
        # action.
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: state_outs,
        }

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        # Same logic as _forward, but also return embeddings to be used by value
        # function branch during training.
        embeddings, state_outs = self._compute_embeddings_and_state_outs(batch)
        logits = self._pi_head(embeddings)
        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.STATE_OUT: state_outs,
            Columns.EMBEDDINGS: embeddings,
        }

    # We implement this RLModule as a ValueFunctionAPI RLModule, so it can be used
    # by value-based methods like PPO or IMPALA.
    @override(ValueFunctionAPI)
    def compute_values(
        self, batch: Dict[str, Any], embeddings: Optional[Any] = None
    ) -> TensorType:
        if embeddings is None:
            embeddings, _ = self._compute_embeddings_and_state_outs(batch)
        values = self._values(embeddings).squeeze(-1)
        return values

    def _compute_embeddings_and_state_outs(self, batch):
        obs = batch[Columns.OBS]
        state_in = batch[Columns.STATE_IN]
        h = state_in["h"]

        curr_pred = self._coffee_predictor(obs)


        # 1 - history dimension
        hs = []
        for i in range(obs.shape[1]):
            # Probabilistic "or" operator (assumes independence)
            h = h + curr_pred[:, i, :] - h * curr_pred[:, i, :]
            hs.append(h)

        new_h = torch.stack(hs, dim=1)
        inp = torch.cat((obs, new_h), dim=-1)
        embeddings = self._fc_net(inp)

        return embeddings, {"h": new_h[:,-1,:]}
