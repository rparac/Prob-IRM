from typing import Optional, Callable, Dict

import ray
from ray import tune
from ray.rllib.algorithms import PPOConfig, AlgorithmConfig, PPO, Algorithm
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils import override
from ray.rllib.utils.typing import ResultDict
from ray.tune.logger import Logger

from rm_marl.agent import RewardMachineAgent
from rm_marl.new_stack.connectors.RM_state_connector import RMStateConnector
from rm_marl.new_stack.connectors.storage_connector import TraceStorage
from rm_marl.new_stack.learner.NewProbFFNSLLearner import NewProbFFNSLLearner
from rm_marl.new_stack.networks.model import PPORMLearningCatalog
from rm_marl.reward_machine import RewardMachine


# TODO: automatically set the connectors that are required
class PPORMConfig(PPOConfig):

    def __init__(self, algo_class=None, rm: RewardMachine = None):
        super().__init__(algo_class or PPORM)

        self._rm = rm
        self._setup_connectors()

    def _setup_connectors(self):
        self.env_runners(
            env_to_module_connector=lambda env: RMStateConnector(
                input_action_space=env.action_space,
                input_observation_space=env.observation_space,
                rm=self._rm,
                as_learner_connector=False,
            ),
        )
        self.training(
            learner_connector=(
                lambda input_observation_space, input_action_space: RMStateConnector(
                    input_action_space=input_action_space,
                    input_observation_space=input_observation_space,
                    rm=self._rm,
                    as_learner_connector=True,
                )
            ),
        )

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


class PPORM(PPO):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMConfig()


tune.register_trainable("PPORM", PPORM)


class PPORMLearningConfig(PPORMConfig):
    def __init__(self, algo_class=None):
        rm = RewardMachineAgent.default_rm()
        self._rm_learner = NewProbFFNSLLearner.remote(rm)
        super().__init__(algo_class or PPORMLearning, rm)

    def rm(self, rm: RewardMachine):
        self._rm = rm
        self._setup_connectors()
        return self

    def _setup_connectors(self):
        super()._setup_connectors()

        self.training(
            learner_connector=(
                lambda input_observation_space, input_action_space: [
                    RMStateConnector(
                        input_action_space=input_action_space,
                        input_observation_space=input_observation_space,
                        rm=self._rm,
                        as_learner_connector=True,
                    ),
                    TraceStorage(
                        input_action_space=input_action_space,
                        input_observation_space=input_observation_space,
                        learner=self._rm_learner,
                    )
                ]
            )
        )


class PPORMLearning(PPORM):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMLearningConfig()

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        results = super().training_step()

        # TODO: currently trying to relearn every step; we may want to
        #   reduce the frequency
        result_ref = self.config._rm_learner.relearn_rm.remote()
        new_rm = ray.get(result_ref)
        if new_rm:
            self._reset_with_rm(new_rm)

        return results

    # TODO: verify if this is hte correct way to reset the policy
    def _reset_with_rm(self, new_rm):
        new_config = self.get_config().copy()
        new_config._is_frozen = False
        new_config = new_config.rm(new_rm)
        policy_ids = new_config._rl_module_spec.module_specs.keys()
        new_config.rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    policy_id: self.config.get_default_rl_module_spec() for policy_id in policy_ids
                }
            )
        )
        self.config = new_config


tune.register_trainable("PPORMLearning", PPORMLearning)
