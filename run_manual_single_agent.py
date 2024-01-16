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
from rm_marl.agent.rm_learning_agent import RewardMachineLearningAgent
from rm_marl.algo import QRM
from rm_marl.envs.mining import MiningLabelingFunctionWrapper, MiningNoisyLabelingFunctionWrapper
from rm_marl.envs.wrappers import AutomataWrapper
from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_transition.deterministic_rm_transitioner import DeterministicRMTransitioner
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from rm_marl.trainer import Trainer

BASE_PATH = os.path.join(os.path.dirname(__file__), "data/mining")
ENV_PATH = os.path.join(BASE_PATH, f"env.txt")

trainer_run_config = {
    "training": True,
    "total_episodes": 100,
    "log_freq": 1,
    "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    "testing_freq": 10,
    "greedy": True,
    "synchronize": False,
    "counterfactual_update": False,
    "recording_freq": 5,
    "seed": 123,
    "name": "mining-learning-rm",
}
np.random.seed(trainer_run_config["seed"])
random.seed(trainer_run_config["seed"])

env = gym.make(
    'rm-marl/Mining-v0',
    render_mode="rgb_array",
    file=ENV_PATH,
)

# rm = None
rm = RewardMachine.load_from_file("data/mining/rm_agent_1.txt")
rm_transitioner = DeterministicRMTransitioner(rm)
# rm_transitioner = ProbRMTransitioner(rm)

# env = MiningNoisyLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)  # type: ignore
env = MiningLabelingFunctionWrapper(env) # type: ignore

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
    rm_transitioner=rm_transitioner,
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
