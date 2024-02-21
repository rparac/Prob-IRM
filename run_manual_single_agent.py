"""
This file showcases how code is intended to be used in the single agent case.
Hydra configuration helps with duplicate configuration, but is not immediately clear to a user.
"""
import os
import random

import gym
import numpy as np
from gym.wrappers import RecordEpisodeStatistics

from rm_marl.agent import RewardMachineAgent
from rm_marl.algo import QRM
from rm_marl.envs.gym_subgoal_automata.gym_subgoal_automata_wrapper import DanielGymAdapter, \
    OfficeWorldDeliverCoffeeLabelingFunctionWrapper
from rm_marl.envs.wrappers import AutomataWrapper
from rm_marl.trainer import Trainer

seed = 123

trainer_run_config = {
    "training": True,
    "total_episodes": 100,  # 100,
    "log_freq": 1,
    "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    "testing_freq": 10,
    "greedy": True,
    "synchronize": False,
    "counterfactual_update": False,
    "recording_freq": 50,
    "seed": seed,
    "name": "office-world",
    "extra_debug_information": True,
}
np.random.seed(trainer_run_config["seed"])
random.seed(trainer_run_config["seed"])

env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"generation": "random", "environment_seed": 0, "hide_state_variables": True})
env = DanielGymAdapter(env, render_mode="rgb_array")  # type: ignore

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

ag = RewardMachineAgent(
    # ag = RewardMachineAgent(
    rm=rm,
    agent_id="A1",
    algo_cls=QRM,
    algo_kws={
        "action_space": env.action_space,
        "seed": 123,
    },
)

agent_dict = {"A1": ag}
env_dict = {"E": env}
trainer = Trainer(env_dict, env_dict, agent_dict)
trainer.run(trainer_run_config)
