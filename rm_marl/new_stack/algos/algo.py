import sys
import time
import pickle
import uuid
import os
from functools import partial
from typing import Optional

import gymnasium as gym
import ray
from ray import tune
from ray.rllib.algorithms import PPOConfig, AlgorithmConfig, PPO, Algorithm
from ray.rllib.utils import override
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.checkpoints import Checkpointable
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.utils.serialization import space_from_dict, space_to_dict
from ray.rllib.utils.typing import ResultDict, EnvConfigDict

from rm_marl.new_stack.learner.rm_learner import RMLearner
from rm_marl.new_stack.utils.env import GET_PERFECT_RM, GET_DEFAULT_RM
from rm_marl.reward_machine import RewardMachine

save_name = "ilasp_learner.pkl"

# TODO: automatically set the connectors that are required
class PPORMConfig(PPOConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class or PPORM)

    def environment(
            self,
            *,
            env_config: Optional[EnvConfigDict] = NotProvided,
            **kwargs
    ) -> "AlgorithmConfig":
        new_env_config = env_config or {}
        new_env_config["rm"] = GET_PERFECT_RM
        return super().environment(env_config=new_env_config, **kwargs)


class PPORM(PPO):
    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMConfig()


tune.register_trainable("PPORM", PPORM)


class PPORMLearningConfig(PPOConfig):
    def __init__(self, algo_class=None):
        self._actor_name = None

        # RM learner params
        self.rm_learner_params = {
            "edge_cost": 2,
            "n_phi_cost": 2,
            "ex_penalty_multiplier": 1,
            "min_penalty": 1,
            "cross_entropy_threshold": 0.5,
            "base_dir": "not_categorized",
            "replay_experience": True,
            "rebalance_classes": True,
            "new_inc_examples": True,
            "max_container_size": None,
        }

        super().__init__(algo_class or PPORMLearning)

    def actor_name(self, actor_name: str):
        self._actor_name = actor_name
        return self

    def environment(
            self,
            *,
            env_config: Optional[EnvConfigDict] = NotProvided,
            **kwargs
    ) -> "AlgorithmConfig":
        new_env_config = env_config or {}
        new_env_config["rm"] = GET_DEFAULT_RM
        return super().environment(env_config=new_env_config, **kwargs)

    def rm_learner(self,
                   edge_cost=NotProvided,
                   n_phi_cost=NotProvided,
                   ex_penalty_multiplier=NotProvided,
                   min_penalty=NotProvided,
                   cross_entropy_threshold=NotProvided,
                   base_dir=NotProvided,
                   rebalance_classes=NotProvided,
                   new_inc_examples=NotProvided,
                   replay_experience=NotProvided,
                   max_container_size=NotProvided,
                   ):
        if edge_cost is not NotProvided:
            self.rm_learner_params["edge_cost"] = edge_cost
        if n_phi_cost is not NotProvided:
            self.rm_learner_params["n_phi_cost"] = n_phi_cost
        if ex_penalty_multiplier is not NotProvided:
            self.rm_learner_params["ex_penalty_multiplier"] = ex_penalty_multiplier
        if min_penalty is not NotProvided:
            self.rm_learner_params["min_penalty"] = min_penalty
        if cross_entropy_threshold is not NotProvided:
            self.rm_learner_params["cross_entropy_threshold"] = cross_entropy_threshold
        if base_dir is not NotProvided:
            self.rm_learner_params["base_dir"] = base_dir
        if rebalance_classes is not NotProvided:
            self.rm_learner_params["rebalance_classes"] = rebalance_classes
        if new_inc_examples is not NotProvided:
            self.rm_learner_params["new_inc_examples"] = new_inc_examples
        if replay_experience is not NotProvided:
            self.rm_learner_params["replay_experience"] = replay_experience
        if max_container_size is not NotProvided:
            self.rm_learner_params["max_container_size"] = max_container_size 


class PPORMLearning(PPO):

    def setup(self, config: AlgorithmConfig) -> None:
        actor_name = str(uuid.uuid4())
        print(f"Actor name is {actor_name}")

        rm = RewardMachine.default_rm()
        kwargs = {"rm_learner_actor": actor_name}

        self.config._is_frozen = False
        self.config.callbacks_class = partial(self.config.callbacks_class, **kwargs)
        self.config._is_frozen = True

        # placement_group = ray.util.placement_group(
        #     bundles=[
        #         {"CPU": 4},
        #     ]
        # )
        # ray.get(placement_group.ready())

        self._rm_learner = (RMLearner.options(name=actor_name)  # type: ignore
                            .remote(rm, actor_name, **self.config.rm_learner_params))  # type: ignore
        super().setup(config)

    @override(Checkpointable)
    def restore_from_path(self, path, *args, **kwargs):
        experiment_file = os.path.join(path, save_name)
        print(experiment_file)
        with open(experiment_file, "rb") as f:
            state = pickle.load(f)
        actor_name = state["actor_name"]
        print(f"Old name is {actor_name}")

        self.config._is_frozen = False
        self.config.callbacks_class = partial(self.config.callbacks_class, **kwargs)
        self.config._is_frozen = True

        rm = RewardMachineAgent.default_rm()
        self._rm_learner = (RMLearner.options(name=actor_name)  # type: ignore
                            .remote(rm, actor_name, **self.config.rm_learner_params))  # type: ignore
        self._rm_learner.set_state_dict.remote(state)

        self.callbacks.set_rm_learner(actor_name)

        rm = ray.get(self._rm_learner.get_curr_rm.remote())
        print(f"Setting rm to {rm}")
        self.set_rm(rm)
        self.reset_policies(rm)

        super().restore_from_path(path, *args, **kwargs)

    def get_rm_learner(self):
        return self._rm_learner

    @classmethod
    @override(Algorithm)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPORMLearningConfig()

    @override(Algorithm)
    def training_step(self) -> ResultDict:
        # Execute a training step for the underlying agents
        results = super().training_step()

        
        curr_time = time.time()
        result_ref = self._rm_learner.relearn_rm.remote()
        new_rm = ray.get(result_ref)
        end_time = time.time()
        print(f"It took {end_time - curr_time} to check if it should relearn RM", file=sys.stderr)
        if new_rm:
            self.set_rm(new_rm)  # type: ignore
            self.reset_policies(new_rm)

        return results

    @PublicAPI
    def get_action_space(self) -> gym.Space:
        def _get_action_space(w):
            env = w.env.unwrapped
            # `env` is a gymnasium.vector.Env.
            if hasattr(env, "single_action_space") and isinstance(
                    env.single_action_space, gym.Space
            ):
                return space_to_dict(env.single_action_space)
            # `env` is a gymnasium.Env.
            elif hasattr(env, "action_spaces") and isinstance(
                    env.action_spaces, gym.Space
            ):
                return space_to_dict(env.action_spaces)

            return None

        action_spaces = self.env_runner_group.foreach_worker(_get_action_space)
        return space_from_dict(action_spaces[0])

    @PublicAPI
    def get_obs_space(self) -> gym.Space:
        def _get_obs_space(w):
            env = w.env.unwrapped
            if hasattr(env, "single_observation_space") and isinstance(
                    env.single_observation_space, gym.Space
            ):
                return space_to_dict(env.single_observation_space)
            # `env` is a gymnasium.Env.
            elif hasattr(env, "observation_spaces") and isinstance(
                    env.observation_spaces, gym.Space
            ):
                return space_to_dict(env.observation_spaces)

            return None

        obs_spaces = self.env_runner_group.foreach_worker(_get_obs_space)
        ret = space_from_dict(obs_spaces[0])
        return ret

    def reset_policies(self, rm) -> None:
        obs_spaces = self.get_obs_space()
        act_spaces = self.get_action_space()

        def _update_config(w):
            initially_frozen = w.config._is_frozen
            w.config._is_frozen = False
            w.config._rl_module_spec = None
            # Apply the same observation space to every agent
            num_agents = w.config.env_config['num_agents']
            rl_module_spec = w.config.get_multi_rl_module_spec(
                spaces={
                    f"p{pid}": (obs_spaces[0], act_spaces[0])
                    for pid in range(num_agents)
                }

            )
            # rl_module_spec = w.config.get_marl_module_spec(
            #     spaces={
            #         f"p{pid}": (obs_spaces[pid], act_spaces[pid])
            #         for pid in obs_spaces
            #     }
            # )
            w.config.rl_module(
                rl_module_spec=rl_module_spec
            )
            w.config.env_config["rm"] = rm
            w.config._is_frozen = initially_frozen

        def _reset_worker(w):
            _update_config(w)

            w._env_to_module = w.config.build_env_to_module_connector(w.env.unwrapped)
            w._cached_to_module = None

            w.make_env()

            num_agents = w.config.env_config['num_agents']
            w.module = w.config.get_multi_rl_module_spec(
                spaces={
                    f"p{pid}": (obs_spaces[0], act_spaces[0])
                    for pid in range(num_agents)
                }
            ).build()

            w._module_to_env = w.config.build_module_to_env_connector(w.env.unwrapped)
            w._needs_initial_reset = True

        # self.env_runner.env.single_observation_space = obs_spaces

        self.env_runner_group.foreach_worker(_reset_worker)

        if self.eval_env_runner_group is not None:
            self.eval_env_runner_group.foreach_worker(_reset_worker)

        _update_config(self)

        def _reset_learner(l):
            _update_config(l)

            l._module_spec = l.config.rl_module_spec
            l._module = l._make_module()

            l.configure_optimizers()

        self.learner_group.foreach_learner(_reset_learner)

    @PublicAPI
    def set_rm(self, rm: RewardMachine) -> None:
        def _set_rm(w):
            w.env.unwrapped.update_rm(rm)

        self.env_runner_group.foreach_worker(_set_rm)

        if self.eval_env_runner_group is not None:
            self.eval_env_runner_group.foreach_worker(_set_rm)


tune.register_trainable("PPORMLearning", PPORMLearning)
