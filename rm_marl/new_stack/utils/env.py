import gymnasium as gym
import numpy as np
from ray.rllib.env import EnvContext

from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldOfficeLabelExtractor, \
    OfficeWorldPlantLabelExtractor, OfficeWorldCoffeeLabelExtractor
from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.envs.pretrained_label_extractor import PretrainedLabelExtractor
from rm_marl.envs.wrappers import LabelingFunctionWrapper, NoisyLabelingFunctionComposer, ProbabilisticRewardShaping, RewardMachineWrapper
from rm_marl.envs.wrappers import LabelThresholding
from rm_marl.new_stack.env.augment_labels_wrapper import AugmentLabelsWrapper
from rm_marl.new_stack.env.rm_wrapper import RMWrapper
from rm_marl.new_stack.utils.gymnasium import gym_getattr
from rm_marl.reward_machine import RewardMachine

GET_PERFECT_RM = "perfect"
GET_DEFAULT_RM = "default_rm"


def hydra_env_creator(env_config):
    def thunk(_env_ctx: EnvContext):
        # curr_id = _env_ctx.worker_index - 1
        # curr_id = _env_ctx.vector_index
        curr_id = _env_ctx["curr_id"]

        # env = gym.make("CartPole-v1")
        env = gym.make(env_config["name"], render_mode=env_config["render_mode"],
                       params={"generation": "random", "environment_seed": env_config["seed"] + curr_id,
                               "hide_state_variables": True, "num_plants": 1})
        env = NewGymSubgoalAutomataAdapter(env, max_episode_length=env_config["max_episode_length"], num_random_seeds=env_config["num_random_seeds"])  # type: ignore
        # raise RuntimeError(env.observation_space.shape)

        if not env_config["use_pretrained_model"]:
            labeling_funs = [label_factory(seed=env_config["seed"]) for label_factory in env_config["label_factories"]]

            label_extractor = NoisyLabelingFunctionComposer(labeling_funs)
        else:
            label_extractor = PretrainedLabelExtractor(env, env_config["tb_storage_path"], curr_id)

        env = LabelingFunctionWrapper(env, label_extractor, use_probability=not env_config["use_old_rm_learner"])

        if env_config['use_thresholding']:
            env = LabelThresholding(env, env_config['labelling_threshold'])
        env = gym.wrappers.FlattenObservation(env)
        env = gym.wrappers.DtypeObservation(env, dtype=np.float32)
        rm = _env_ctx.get("rm", None)

        if rm is None: 
            env = AugmentLabelsWrapper(env)
            return env

        if rm == GET_PERFECT_RM:
            rm = gym_getattr(env, 'get_perfect_rm')()
        elif rm == GET_DEFAULT_RM:
            rm = RewardMachine.default_rm()
        elif isinstance(rm, RewardMachine):
            rm = rm
        else:
            raise RuntimeError(f"Unexpected RM provided {rm}")
        if env_config["use_rs"]:
            env = ProbabilisticRewardShaping(
                env, 
                shaping_rm=rm, 
                use_deterministic_transitioner=env_config["use_old_rm_learner"], 
                discount_factor=env_config["rs_discount"]
            )
        env = RMWrapper(env, rm=rm, use_deterministic_transitioner=env_config["use_old_rm_learner"])

        # raise RuntimeError(env.observation_space.shape)
        return env

    return thunk
