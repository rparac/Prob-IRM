from typing import Optional, Tuple, Type

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.torch_policy_v2 import TorchPolicyV2
from ray.rllib.utils.annotations import override


class CustomPPOTorchPolicy(PPOTorchPolicy):

    @override(TorchPolicyV2)
    def make_model_and_action_dist(
            self,
    ) -> Tuple[ModelV2, Type[TorchDistributionWrapper]]:
        """Create model.

        Note: only one of make_model or make_model_and_action_dist
        can be overridden.

        Returns:
            ModelV2 model.
        """
        dist_class, logit_dim = ModelCatalog.get_action_dist(
            self.action_space, self.config["model"], framework=self.framework
        )
        model = ModelCatalog.get_model_v2(
            obs_space=self.observation_space,
            action_space=self.action_space,
            num_outputs=logit_dim,
            model_config=self.config["model"],
            framework=self.framework,
        )
        return model, dist_class


class PPORM(PPO):

    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config) -> Optional[Type[Policy]]:
        if config["framework"] != "torch":
            raise NotImplementedError(f"framework {config['framework']} not supported")

        return CustomPPOTorchPolicy
