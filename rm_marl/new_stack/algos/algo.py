import gymnasium as gym
import numpy as np
import ray
from gymnasium.vector.utils import batch_space, create_empty_array
from ray import tune
from ray.rllib.algorithms import PPOConfig, AlgorithmConfig, PPO, Algorithm
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module import MultiRLModuleSpec
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.utils import override
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.serialization import space_from_dict, space_to_dict
from ray.rllib.utils.typing import ResultDict

from rm_marl.agent import RewardMachineAgent
from rm_marl.new_stack.learner.NewProbFFNSLLearner import NewProbFFNSLLearner
from rm_marl.new_stack.modules.net import NewCustomNet
# from rm_marl.new_stack.networks.model import PPORMLearningCatalog
from rm_marl.reward_machine import RewardMachine


# TODO: automatically set the connectors that are required
class PPORMConfig(PPOConfig):

    def __init__(self, algo_class=None, rm: RewardMachine = RewardMachineAgent.default_rm()):
        super().__init__(algo_class or PPORM)

        self._rm = rm
        self._setup_connectors()

    def _setup_connectors(self):
        pass
        # self.env_runners(
        #     env_to_module_connector=lambda env: RMStateConnector(
        #         input_action_space=env.action_space,
        #         input_observation_space=env.observation_space,
        #         rm=self._rm,
        #         as_learner_connector=False,
        #     ),
        # )
        # self.training(
        #     learner_connector=(
        #         lambda input_observation_space, input_action_space: RMStateConnector(
        #             input_action_space=input_action_space,
        #             input_observation_space=input_observation_space,
        #             rm=self._rm,
        #             as_learner_connector=True,
        #         )
        #     ),
        # )

    # def get_catalog(self):
    #     return PPORMLearningCatalog

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> RLModuleSpec:

        if self.framework_str == "torch":

            return RLModuleSpec(
                module_class=PPOTorchRLModule,
                # catalog_class=self.get_catalog(),
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


class PPORMLearningConfig(PPOConfig):
    def __init__(self, algo_class=None):
        self._actor_name = None
        super().__init__(algo_class or PPORMLearning)

    def rm(self, rm: RewardMachine):
        self._rm = rm
        self._setup_connectors()
        return self

    def actor_name(self, actor_name: str):
        self._actor_name = actor_name
        return self

    def _setup_connectors(self):
        pass
        # super()._setup_connectors()
        #
        # self.training(
        #     learner_connector=(
        #         lambda input_observation_space, input_action_space: [
        #             RMStateConnector(
        #                 input_action_space=input_action_space,
        #                 input_observation_space=input_observation_space,
        #                 rm=self._rm,
        #                 as_learner_connector=True,
        #             ),
        #             TraceStorage(
        #                 input_action_space=input_action_space,
        #                 input_observation_space=input_observation_space,
        #                 learner=self._rm_learner,
        #             )
        #         ]
        #     )
        # )


class PPORMLearning(PPO):

    # TODO: need to fix the default_rm double usage
    def setup(self, config: AlgorithmConfig) -> None:
        rm = RewardMachineAgent.default_rm()
        self._rm_learner = NewProbFFNSLLearner.options(name=config._actor_name).remote(rm)  # type: ignore
        super().setup(config)
        # raise RuntimeError(self.get_module(DEFAULT_MODULE_ID))

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMLearningConfig()

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        # Execute a training step for the underlying agents
        results = super().training_step()

        # TODO: currently trying to relearn every step; we may want to
        #   reduce the frequency
        result_ref = self._rm_learner.relearn_rm.remote()
        new_rm = ray.get(result_ref)
        if new_rm:
            self.set_rm(new_rm)  # type: ignore
            self.reset_policies(new_rm)
            # self._reset_with_rm(new_rm)  # type: ignore

        return results

    @PublicAPI
    def get_action_space(self) -> gym.Space:
        def _get_action_space(w):
            env = w.env
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

        action_spaces = self.env_runner_group.foreach_worker(_get_action_space)
        return space_from_dict(action_spaces[0])

    @PublicAPI
    def get_obs_space(self) -> gym.Space:
        def _get_obs_space(w):
            env = w.env.envs[0]
            # env = w.env
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

        obs_spaces = self.env_runner_group.foreach_worker(_get_obs_space)
        return space_from_dict(obs_spaces[0])

    def reset_policies(self, rm) -> None:
        obs_spaces = self.get_obs_space()
        act_spaces = self.get_action_space()

        def _update_config(w):
            w.config._is_frozen = False
            w.config._rl_module_spec = None
            rl_module_spec = w.config.get_rl_module_spec(
                spaces={
                    DEFAULT_MODULE_ID: (obs_spaces, act_spaces),
                }
            )
            w.config.rl_module(
                rl_module_spec=rl_module_spec
            )
            w.config.env_config["rm"] = rm
            w.config._is_frozen = True

        def _reset_worker(w):
            _update_config(w)

            observation_space = w.env.envs[0].observation_space
            num_envs = len(w.env.envs)
            w.env.single_observation_space = observation_space
            w.env.observation_space = batch_space(observation_space, n=num_envs)
            w.env.observations = create_empty_array(
                observation_space, n=num_envs, fn=np.zeros
            )

            w._env_to_module = w.config.build_env_to_module_connector(w.env, debugging=True)
            w._cached_to_module = None

            w.make_env()
            w.module = w.config.get_rl_module_spec().build()
            # w.module = w._make_module()
            w._module_to_env = w.config.build_module_to_env_connector(w.env)
            w._needs_initial_reset = True

        # self.env_runner.env.single_observation_space = obs_spaces

        self.env_runner_group.foreach_worker(_reset_worker)

        if self.eval_env_runner_group is not None:
            self.eval_env_runner_group.foreach_worker(_reset_worker)

        _update_config(self)

        def _reset_learner(l):
            _update_config(l)

            # raise RuntimeError(l._module_spec, l.config.rl_module_spec)
            l._module_spec = MultiRLModuleSpec(module_specs={
                DEFAULT_MODULE_ID: l.config.rl_module_spec,
            })
            l._module = l._make_module()
            # l._module = l._module_spec.build()
            # raise RuntimeError(l._module.framework)
            l.configure_optimizers()

        self.learner_group.foreach_learner(_reset_learner)

        x = self.get_module(DEFAULT_MODULE_ID)
        # raise RuntimeError(x)

        # def _test(w):
        #     raise RuntimeError(w.env.single_observation_space)

        # self.env_runner_group.foreach_worker(_test)

        # raise RuntimeError(x)

    @PublicAPI
    def set_rm(self, rm: RewardMachine) -> None:
        def _set_rm(w):
            w.env.update_rm(rm)

        self.env_runner_group.foreach_worker(
            lambda env_runner: [
                env.update_rm(rm) for env in env_runner.env.envs  # Access each individual environment
            ]
        )

        if self.eval_env_runner_group is not None:
            self.eval_env_runner_group.foreach_worker(
                lambda env_runner: [
                    env.update_rm(rm) for env in env_runner.env.envs  # Access each individual environment
                ]
            )
            # self.evaluation_workers.foreach_worker(_set_rm)

    # TODO: verify if this is hte correct way to reset the policy
    def _reset_with_rm(self, new_rm):
        def _reset_worker(w):
            w._env_to_module = w.config.build_env_to_module_connector(w.env)
            w._cached_to_module = None
            w._module_to_env = w.config.build_module_to_env_connector(w.env)
            w._needs_initial_reset = True

        # self.env_runner_group.foreach_worker(
        #     lambda env_runner: [
        #         env.update_rm(new_rm) for env in env_runner.env.envs  # Access each individual environment
        #     ]
        # )

        # self.env_runner_group.foreach_worker(
        #     lambda env_runner: env_runner.env.call_async(name="update_rm", rm=new_rm)
        # )
        # self.env_runner_group.foreach_worker(
        #     lambda env_runner: env_runner.env.call_wait()
        # )

        self.env_runner_group.foreach_worker(_reset_worker)
        self.eval_env_runner_group.foreach_worker(_reset_worker)

        x = self.get_module(DEFAULT_MODULE_ID)
        assert isinstance(x, NewCustomNet)
        x.setup_architecture(len(new_rm.states))


tune.register_trainable("PPORMLearning", PPORMLearning)

# def _new_reset_with_rm(self, new_rm):
#     obs_spaces = self.get_obs_space()
#     act_spaces = self.get_action_space()
#
#     def _update_config(w):
#         w.config._is_frozen = False
#         w.config._rl_module_spec = None
#         rl_module_spec = w.config.get_marl_module_spec(
#             spaces={
#                 pid: (obs_spaces[pid], act_spaces[pid])
#                 for pid in obs_spaces
#             }
#         )
#         w.config.rl_module(
#             rl_module_spec=rl_module_spec
#         )
#         w.config._is_frozen = True
#
#     def _reset_worker(w):
#         _update_config(w)
#
#         w._env_to_module = w.config.build_env_to_module_connector(w.env)
#         w._cached_to_module = None
#         w.module = w._make_module()
#         w._module_to_env = w.config.build_module_to_env_connector(w.env)
#         w._needs_initial_reset = True
#
#     self.workers.foreach_worker(_reset_worker)
#
#     if self.evaluation_workers is not None:
#         self.evaluation_workers.foreach_worker(_reset_worker)
#
#     _update_config(self)
#
#     def _reset_learner(l):
#         _update_config(l)
#
#         l._module_spec = l.config.rl_module_spec
#         l._module = l._make_module()
#         l.configure_optimizers()
#
#     self.learner_group.foreach_learner(_reset_learner)
