from ray import tune
from ray.rllib.algorithms import PPOConfig, AlgorithmConfig, PPO, Algorithm
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils import override

from rm_marl.reward_machine import RewardMachine
from rm_marl.trainer.new_ray_utils.model import PPORMLearningCatalog, PPORMTorchRLModule


class PPORMConfig(PPOConfig):

    def __init__(self, algo_class=None, rm: RewardMachine = None):
        super().__init__(algo_class or PPORM)

        self.rm = rm

    def get_catalog(self):
        return PPORMLearningCatalog

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> SingleAgentRLModuleSpec:

        if self.framework_str == "torch":

            return SingleAgentRLModuleSpec(
                module_class=PPOTorchRLModule,
                catalog_class=self.get_catalog(),
            )
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Only 'torch' is supported."
            )

    def rm(self, rm: RewardMachine):
        self.rm = rm


class PPORM(PPO):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMConfig()


tune.register_trainable("PPORM", PPORM)
