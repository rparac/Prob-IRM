import gymnasium as gym
import numpy as np
from ray.rllib.env import EnvContext

from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldOfficeLabelingFunctionWrapper, \
    OfficeWorldPlantLabelingFunctionWrapper, OfficeWorldCoffeeLabelingFunctionWrapper
from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import NoisyLabelingFunctionComposer
from rm_marl.new_stack.env.rm_wrapper import RMWrapper


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
            OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
        ]

        env = NoisyLabelingFunctionComposer(labeling_funs)
        env = gym.wrappers.FlattenObservation(env)
        env = gym.experimental.wrappers.DtypeObservationV0(env, **{"dtype": np.float32})
        env = RMWrapper(env, rm=_env_ctx.get("rm", None))

        # raise RuntimeError(env.observation_space.shape)
        return env

    return thunk
