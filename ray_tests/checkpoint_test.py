"""
This example continues training for PPORM agent that did not terminate.
Things commented out were attempts to continue training. Will need to ask Leo for that
"""

from ray import tune
from ray.rllib.algorithms import PPOConfig
from ray.train.base_trainer import BaseTrainer
from ray.tune import ResumeConfig, register_env

from rm_marl.new_stack.algos.algo import PPORMConfig, PPORM
from rm_marl.new_stack.env.multi_env_with_rm import make_multi_agent_with_rm
from rm_marl.new_stack.utils.env import env_creator

env_name = 'gym_subgoal_automata:OfficeWorldDeliverCoffee-v0'
register_env("env", make_multi_agent_with_rm(env_creator(env_name)))

tuner_path = "/home/rp218/ray_results/PPORM_2024-10-07_14-55-14"

tuner = tune.Tuner.restore(
    path=tuner_path,
    trainable=PPORM,
    # _resume_config=ResumeConfig(
    #     finished=ResumeConfig.ResumeType.RESUME,
    #     unfinished=ResumeConfig.ResumeType.RESUME,
    #     errored=ResumeConfig.ResumeType.RESUME,
    # )
)
print(tuner.get_results())
# tuner._local_tuner._run_config.stop = {"training_iteration": 32}
# tuner._local_tuner._resume_config.finished = ResumeConfig.ResumeType.RESUME
newer_results = tuner.fit()

# BaseTrainer

print("done")