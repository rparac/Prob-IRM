import itertools
import os
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Type

import gymnasium as gym
import numpy as np
import ray
from ray import tune
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig, NotProvided
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import PublicAPI, override
from ray.rllib.utils.metrics import TIMERS, TRAINING_ITERATION_TIMER
from ray.rllib.utils.serialization import space_from_dict, space_to_dict
from ray.rllib.utils.typing import ResultDict

from rm_marl.reward_machine import RewardMachine

LEARNER_RM_TIMER = "rm_learning"


class PPORMConfig(PPOConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class or PPORM)

        self.shared_hidden_layers: tuple = tuple()  # (0, 1)

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> SingleAgentRLModuleSpec:

        if self.framework_str == "torch":

            return SingleAgentRLModuleSpec(
                module_class=PPORMTorchRLModule, catalog_class=PPOCatalog
            )
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Only 'torch' is supported."
            )

    def network(
            self,
            *,
            shared_hidden_layers: Optional[tuple] = NotProvided,
    ) -> AlgorithmConfig:

        if shared_hidden_layers is not NotProvided:
            self.shared_hidden_layers = shared_hidden_layers

        return self

    @property
    @override(AlgorithmConfig)
    def _model_config_auto_includes(self) -> Dict[str, Any]:
        return super()._model_config_auto_includes | {"vf_share_layers": True}


class PPORM(PPO):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMConfig()


tune.register_trainable("PPORM", PPORM)


class PPORMLearningConfig(PPORMConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class or PPORMLearning)

        self.rm_learning_freq = 1
        self.traces_buffer = None
        self.rm_learner_class = None
        self.rm_learner_kws = None

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> SingleAgentRLModuleSpec:

        if self.framework_str == "torch":

            catalog_class = PPORMLearningSharedLayersCatalog if self.shared_hidden_layers else PPORMLearningCatalog

            return SingleAgentRLModuleSpec(
                module_class=PPORMTorchRLModule,
                model_config_dict=self._model_config_dict,
                catalog_class=catalog_class
            )
        else:
            raise ValueError(
                f"The framework {self.framework_str} is not supported. "
                "Only 'torch' is supported."
            )

    def rm(
            self,
            *,
            rm_learning_freq: Optional[int] = NotProvided,
            traces_buffer: Optional[ray.ObjectRef] = NotProvided,
            rm_learner_class: Optional[type] = NotProvided,
            rm_learner_kws: Optional[dict] = NotProvided,
    ) -> AlgorithmConfig:

        if rm_learning_freq is not NotProvided:
            self.rm_learning_freq = rm_learning_freq

        if traces_buffer is not NotProvided:
            self.traces_buffer = traces_buffer

        if rm_learner_class is not NotProvided:
            self.rm_learner_class = rm_learner_class

        if rm_learner_kws is not NotProvided:
            self.rm_learner_kws = rm_learner_kws

        return self

    def validate(self) -> None:
        super().validate()

        if self.traces_buffer is None:
            raise ValueError("A `traces_buffer` must be provided.")

        if self.rm_learner_class is None:
            raise ValueError("A `rm_learner_class` must be provided.")


class PPORMLearning(PPORM):

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMLearningConfig()

    @override(PPO)
    def setup(self, config: AlgorithmConfig) -> None:
        super().setup(config)

        self.traces_buffer = config.traces_buffer
        if config.rm_learner_class:
            self.rm_learner = config.rm_learner_class(**config.rm_learner_kws)
            self.rm_learner.set_log_folder(os.path.join(self._logdir, "rm"))
        else:
            self.rm_learner = None

        self.init_policy_weights = {
            k: deepcopy(v)
            for k, v in self.workers.local_worker().get_weights().items()
        }

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMLearningConfig()

    @PublicAPI
    def get_rm(self) -> RewardMachine:
        def _get_rm(w):
            return w.env.get_rm()

        rms = list(
            set(self.workers.foreach_worker(_get_rm))
        )
        assert len(rms) == 1, f"More than 1 RM: {rms}"
        return rms[0]

    @PublicAPI
    def get_action_space(self) -> dict:
        def _get_action_space(env):
            # `env` is a gymnasium.vector.Env.
            if hasattr(env, "single_action_space") and isinstance(
                    env.single_action_space, gym.Space
            ):
                return space_to_dict(env.single_action_space)
            # `env` is a gymnasium.Env.
            elif hasattr(env, "action_space") and isinstance(
                    env.action_space, gym.Space
            ):
                return space_to_dict(env.action_space)

            return None

        action_spaces = list(
            itertools.chain.from_iterable(self.workers.foreach_env(_get_action_space))
        )
        return space_from_dict(action_spaces[0])

    @PublicAPI
    def get_obs_space(self) -> dict:
        def _get_obs_space(env):
            if hasattr(env, "single_observation_space") and isinstance(
                    env.single_observation_space, gym.Space
            ):
                return space_to_dict(env.single_observation_space)
            # `env` is a gymnasium.Env.
            elif hasattr(env, "observation_space") and isinstance(
                    env.observation_space, gym.Space
            ):
                return space_to_dict(env.observation_space)

            return None

        obs_spaces = list(
            itertools.chain.from_iterable(self.workers.foreach_env(_get_obs_space))
        )
        return space_from_dict(obs_spaces[0])

    @PublicAPI
    def set_rm(self, rm: RewardMachine) -> None:
        def _set_rm(w):
            w.env.set_rm(rm)

        self.workers.foreach_worker(_set_rm)

    def _rm_to_img(self, rm: RewardMachine) -> None:
        import io

        from PIL import Image

        data = io.BytesIO()
        data.write(rm.to_digraph().pipe(format="png"))
        data.seek(0)

        return Image.open(data)

    @PublicAPI
    def reset_policies(self, rm: RewardMachine) -> None:

        init_state = deepcopy(self.init_policy_weights)

        def _reset(w):
            w.set_weights(init_state)
            w._needs_initial_reset = True

            # rl_module_spec = w.config.get_marl_module_spec(
            #     spaces={
            #         pid: (
            #             w.env.observation_space.get(pid, next(iter(w.env.observation_space.values()))),
            #             w.env.action_space.get(pid, next(iter(w.env.action_space.values())))
            #         )
            #         for pid in w.config.policies
            #     }
            # )
            # w.config._is_frozen = False
            # w.config.rl_module(
            #     rl_module_spec=rl_module_spec
            # )
            # w.config._is_frozen = True

            # w._env_to_module = w.config.build_env_to_module_connector(w.env)
            # w._cached_to_module = None
            # w.module = w._make_module()
            # w._module_to_env = w.config.build_module_to_env_connector(w.env)
            # w._needs_initial_reset = True

        self.workers.foreach_worker(_reset)

    def _learn_rm(self):
        if (
                self._timers[TRAINING_ITERATION_TIMER].count and
                self._timers[TRAINING_ITERATION_TIMER].count % self.config.rm_learning_freq
                == 0
        ):
            pos, dend, inc = ray.get(self.traces_buffer.get_all_examples.remote())
            rm = self.get_rm()

            candidate_rm = self.rm_learner.learn(rm, pos, dend, inc)
            if candidate_rm:
                self.set_rm(candidate_rm)

                # DEBUG
                assert (
                        candidate_rm == self.get_rm()
                ), "Something went wrong when setting the new RM"

                current_rm_plot = os.path.join(self._logdir, "rm", "current")
                candidate_rm.plot(current_rm_plot)

                self.reset_policies(candidate_rm)

    def _training_step_new_api_stack(self) -> ResultDict:
        super()._training_step_new_api_stack()

        with self.metrics.log_time((TIMERS, LEARNER_RM_TIMER)):
            self._learn_rm()

        return self.metrics.reduce()

    def _training_step_old_and_hybrid_api_stacks(self) -> ResultDict:
        training_results = super()._training_step_old_and_hybrid_api_stacks()

        with self.metrics.log_time((TIMERS, LEARNER_RM_TIMER)):
            self._learn_rm()

        return training_results


tune.register_trainable("PPORMLearning", PPORMLearning)
