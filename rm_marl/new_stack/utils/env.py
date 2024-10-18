import gymnasium as gym
import numpy as np
from ray.rllib.env import EnvContext

from rm_marl.agent import RewardMachineAgent
from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldOfficeLabelingFunctionWrapper, \
    OfficeWorldPlantLabelingFunctionWrapper, OfficeWorldCoffeeLabelingFunctionWrapper
from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import NoisyLabelingFunctionComposer, ProbabilisticRewardShaping, RewardMachineWrapper
from rm_marl.new_stack.env.augment_labels_wrapper import AugmentLabelsWrapper
from rm_marl.new_stack.env.rm_wrapper import RMWrapper
from rm_marl.reward_machine import RewardMachine

GET_PERFECT_RM = "perfect"
NO_RM = "none"


def hydra_env_creator(env_config):
    def thunk(_env_ctx: EnvContext):
        # curr_id = _env_ctx.worker_index - 1
        # curr_id = _env_ctx.vector_index
        curr_id = _env_ctx["curr_id"]

        # env = gym.make("CartPole-v1")
        env = gym.make(env_config["name"], render_mode=env_config["render_mode"],
                       params={"generation": "random", "environment_seed": env_config["seed"] + curr_id,
                               "hide_state_variables": True})
        env = NewGymSubgoalAutomataAdapter(env, max_episode_length=250)  # type: ignore
        # raise RuntimeError(env.observation_space.shape)

        labeling_funs = [label_factory(env) for label_factory in env_config["label_factories"]]

        env = NoisyLabelingFunctionComposer(labeling_funs)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.experimental.wrappers.DtypeObservationV0(env, **{"dtype": np.float32})
        rm = _env_ctx.get("rm", None)

        if rm == NO_RM:
            env = AugmentLabelsWrapper(env)
            return env

        if rm == GET_PERFECT_RM:
            rm = env.get_perfect_rm()
        elif rm is None:
            rm = RewardMachineAgent.default_rm()
        elif isinstance(rm, RewardMachine):
            rm = rm
        else:
            raise RuntimeError(f"Unexpected RM provided {rm}")

        if env_config["use_rs"]:
            env = ProbabilisticRewardShaping(env, shaping_rm=rm)
        env = RMWrapper(env, rm=rm)

        # raise RuntimeError(env.observation_space.shape)
        return env

    return thunk


def env_creator(env_id):
    def thunk(_env_ctx: EnvContext):
        curr_id = _env_ctx.worker_index - 1

        # env = gym.make("CartPole-v1")
        env = gym.make(env_id, render_mode="rgb_array",
                       params={"generation": "random", "environment_seed": 6 + curr_id,
                               "hide_state_variables": True})
        env = NewGymSubgoalAutomataAdapter(env, max_episode_length=250)  # type: ignore
        # raise RuntimeError(env.observation_space.shape)

        labeling_funs = [
            OfficeWorldOfficeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            OfficeWorldPlantLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            # OfficeWorldALabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            # OfficeWorldBLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            # OfficeWorldCLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            # OfficeWorldDLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=0.9814815521240234,
                                                     sensor_false_confidence=0.9814815521240234),
        ]

        env = NoisyLabelingFunctionComposer(labeling_funs)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.experimental.wrappers.DtypeObservationV0(env, **{"dtype": np.float32})
        rm = _env_ctx.get("rm", None)
        if rm is None:
            rm = RewardMachineAgent.default_rm()
            env = ProbabilisticRewardShaping(env, shaping_rm=rm)
            env = RMWrapper(env, rm=rm)
        elif isinstance(rm, RewardMachine):
            env = ProbabilisticRewardShaping(env, shaping_rm=rm)
            env = RMWrapper(env, rm=rm)
        elif rm == GET_PERFECT_RM:
            rm = env.get_perfect_rm()
            env = ProbabilisticRewardShaping(env, shaping_rm=rm)
            env = RMWrapper(env, rm=rm)
        elif rm == NO_RM:
            env = AugmentLabelsWrapper(env)
        else:
            raise RuntimeError(f"Unexpected RM provided {rm}")

        # raise RuntimeError(env.observation_space.shape)
        return env

    return thunk
