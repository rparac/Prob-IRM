"""
This file showcases how code is intended to be used in the single agent case.
Hydra configuration helps with duplicate configuration, but is not immediately clear to a user.
"""
import os
import random

import gym
import numpy as np
from gym.wrappers import RecordEpisodeStatistics

from rm_marl.agent import RewardMachineAgent, RewardMachineLearningAgent
from rm_marl.algo import QRM
from rm_marl.envs.gym_subgoal_automata_wrapper import GymSubgoalAutomataAdapter, \
    OfficeWorldDeliverCoffeeLabelingFunctionWrapper
from rm_marl.envs.wrappers import AutomataWrapper
from rm_marl.trainer import Trainer

seed = 123

trainer_run_config = {
    "training": True,
    "total_episodes": 10000,
    "log_freq": 1,
    "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    "testing_freq": 1000,
    "greedy": True,
    "synchronize": False,
    "counterfactual_update": True,
    "recording_freq": 1000,
    "seed": seed,
    "name": "office-world",
    "extra_debug_information": False,
}
np.random.seed(trainer_run_config["seed"])
random.seed(trainer_run_config["seed"])

env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"generation": "random", "environment_seed": 1, "hide_state_variables": True})
env = GymSubgoalAutomataAdapter(env, render_mode="rgb_array", max_episode_length=250)  # type: ignore

# rm = None
rm = env.get_perfect_rm()
rm.plot("rm")
env = OfficeWorldDeliverCoffeeLabelingFunctionWrapper(env)  # type: ignore

# AutomataWrapper here only provides the filter_label function (used in counter_factual update).
#  It also logs RM states
env = AutomataWrapper(
    env,
    rm=rm,
    label_mode=AutomataWrapper.LabelMode.ALL,
    termination_mode=AutomataWrapper.TerminationMode.ENV,
)
env = RecordEpisodeStatistics(env)  # type: ignore

ag = RewardMachineLearningAgent(
    # ag = RewardMachineAgent(
    # rm=rm,
    agent_id="A1",
    algo_cls=QRM,
    algo_kws={
        "action_space": env.action_space,
        "seed": 123,
        "epsilon": 0.1,
        "gamma": 0.99,
        "alpha": 0.1,
    },
)

agent_dict = {"A1": ag}
env_dict = {"E": env}
trainer = Trainer(env_dict, env_dict, agent_dict)
trainer.run(trainer_run_config)
