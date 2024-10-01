from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.models.configs import ActorCriticEncoderConfig, MLPHeadConfig, MLPEncoderConfig
from ray.rllib.models.torch.torch_distributions import TorchCategorical
from ray.rllib.utils import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class NewCustomNet(PPOTorchRLModule):
    @override(PPORLModule)
    def setup(self):
        states = self.config.model_config_dict["num_rm_states"]
        # super().setup()
        # Not calling super.setup() as it deletes some of the keys
        self._inference_only_state_dict_keys = None
        # The configuration observation space includes extra dimension for RM
        #  Here we update the input dimension to the new observation space size
        input_dim = self.config.observation_space.shape[0] - self.config.model_config_dict[
            "num_rm_states"] + states,
        mlp_encoder_config = MLPEncoderConfig(
            input_dims=input_dim,
            hidden_layer_dims=[16, 16],
            # hidden_layer_activation="relu",
            hidden_layer_use_layernorm=False,
            output_layer_dim=None,
        )
        # mlp_encoder_config = MobileNetV2EncoderConfig()
        # Since we want to use PPO, which is an actor-critic algorithm, we need to
        # use an ActorCriticEncoderConfig to wrap the base encoder config.
        actor_critic_encoder_config = ActorCriticEncoderConfig(
            base_encoder_config=mlp_encoder_config,
        )
        self.encoder = actor_critic_encoder_config.build(framework="torch")
        mlp_encoder_output_dims = mlp_encoder_config.output_dims
        pi_config = MLPHeadConfig(
            input_dims=mlp_encoder_output_dims,
            output_layer_dim=int(self.config.action_space.n),
            # hidden_layer_dims=[16, 16],
        )
        vf_config = MLPHeadConfig(
            input_dims=mlp_encoder_output_dims, output_layer_dim=1,
        )
        self.pi = pi_config.build(framework="torch")
        self.vf = vf_config.build(framework="torch")
        self.action_dist_cls = TorchCategorical
