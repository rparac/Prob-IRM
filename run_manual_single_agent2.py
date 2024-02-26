"""
This file showcases how code is intended to be used in the single agent case.
Hydra configuration helps with duplicate configuration, but is not immediately clear to a user.
"""
import cProfile
import os
import random

import gym
import numpy as np
from gym.wrappers import RecordEpisodeStatistics

from rm_marl.agent import RewardMachineAgent, RewardMachineLearningAgent
from rm_marl.algo import QRM
from rm_marl.envs.gym_subgoal_automata_wrapper import GymSubgoalAutomataAdapter, \
    OfficeWorldOfficeLabelingFunctionWrapper, OfficeWorldPlantLabelingFunctionWrapper, \
    OfficeWorldCoffeeLabelingFunctionWrapper
from rm_marl.envs.wrappers import AutomataWrapper, NoisyLabelingFunctionComposer
from rm_marl.rm_learning.ilasp.noisy_learner.ProbFFNSLLearner import ProbFFNSLLearner
from rm_marl.rm_transition.deterministic_rm_transitioner import DeterministicRMTransitioner
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from rm_marl.trainer import Trainer


def fun():
    trainer = Trainer(env_dict, env_dict, agent_dict)
    trainer.run(trainer_run_config)


seed = 123

trainer_run_config = {
    "training": True,
    "total_episodes": 1000,  # 10000,
    "log_freq": 1,
    "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    "testing_freq": 100,  # 1000,
    "greedy": True,
    "synchronize": False,
    "counterfactual_update": False,
    "recording_freq": 100,  # 1000,
    "seed": seed,
    "name": "office-world",
    "extra_debug_information": False,
}
np.random.seed(trainer_run_config["seed"])
random.seed(trainer_run_config["seed"])

env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"generation": "random", "environment_seed": 7, "hide_state_variables": True})
env = GymSubgoalAutomataAdapter(env, render_mode="rgb_array", max_episode_length=250)  # type: ignore

office_l = OfficeWorldOfficeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
plant_l = OfficeWorldPlantLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
coffee_l = OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=0.9, sensor_false_confidence=0.9)
env = NoisyLabelingFunctionComposer([office_l, plant_l, coffee_l])

rm = None
rm_transitioner = ProbRMTransitioner(rm)
# AutomataWrapper here only provides the filter_label function (used in counter_factual update).
#  It also logs RM states
env = AutomataWrapper(
    env,
    rm_transitioner=rm_transitioner,
    label_mode=AutomataWrapper.LabelMode.ALL,
    termination_mode=AutomataWrapper.TerminationMode.ENV,
)
env = RecordEpisodeStatistics(env)  # type: ignore

ag = RewardMachineLearningAgent(
    # ag = RewardMachineAgent(
    # rm=rm,
    rm_transitioner=rm_transitioner,
    agent_id="A1",
    algo_cls=QRM,
    algo_kws={
        "action_space": env.action_space,
        "seed": 123,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 50,
        "gamma": 0.99,
        "alpha": 0.1,
    },
    rm_learner_cls=ProbFFNSLLearner,
)

agent_dict = {"A1": ag}
env_dict = {"E": env}

cProfile.run('fun()')
