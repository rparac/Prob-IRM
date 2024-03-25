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
    OfficeWorldOfficeLabelingFunctionWrapper, OfficeWorldCoffeeLabelingFunctionWrapper, \
    OfficeWorldPlantLabelingFunctionWrapper, OfficeWorldMailLabelingFunctionWrapper
from rm_marl.envs.wrappers import AutomataWrapper, NoisyLabelingFunctionComposer, RewardMachineWrapper
from rm_marl.rm_learning.ilasp.noisy_learner.ProbFFNSLLearner import ProbFFNSLLearner
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from rm_marl.trainer import Trainer


def get_base_env(seed, agent_id):
    env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffeeAndMail-v0",
                   params={"generation": "random", "environment_seed": seed, "hide_state_variables": True})
    env = GymSubgoalAutomataAdapter(env, agent_id, render_mode="rgb_array", max_episode_length=250)  # type: ignore
    office_l = OfficeWorldOfficeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
    plant_l = OfficeWorldPlantLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
    coffee_l = OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
    mail_l = OfficeWorldMailLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
    env = NoisyLabelingFunctionComposer([office_l, plant_l, coffee_l, mail_l])

    rm = env.get_perfect_rm()
    rm_transitioner = ProbRMTransitioner(rm)
    # AutomataWrapper here only provides the filter_label function (used in counter_factual update).
    #  It also logs RM states
    env = RewardMachineWrapper(
        env,
        rm_transitioner=rm_transitioner,
        label_mode=AutomataWrapper.LabelMode.ALL,
        termination_mode=AutomataWrapper.TerminationMode.ENV,
    )
    env = RecordEpisodeStatistics(env)  # type: ignore

    return env


seed = 123

trainer_run_config = {
    "training": True,
    "total_episodes": 2000,  # 10000,
    "log_freq": 1,
    "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
    "testing_freq": 1000,  # 1000,
    "greedy": True,
    "synchronize": False,
    "counterfactual_update": False,
    "recording_freq": 1000,  # 1000,
    "seed": seed,
    "name": "office-world-coffee-mail",
    "extra_debug_information": False,
    "no_display": False,
}
np.random.seed(trainer_run_config["seed"])
random.seed(trainer_run_config["seed"])

envs = []
rm_agents = []
num_envs = 1
for i in range(num_envs):
    agent_id = f"A{i + 1}"
    env = get_base_env(seed + i, agent_id)
    envs.append(env)
    algo = QRM(**{
        "action_space": env.action_space,
        "seed": 123,
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 50,
        "gamma": 0.99,
        "alpha": 0.1,
    })
    rm_agent = RewardMachineAgent(env.agent_id, env.rm_transitioner, algo)
    rm_agents.append(rm_agent)

# learning_ag = RewardMachineLearningAgent(
#     rm_agent=rm_agents,
#     rm_learner_cls=ProbFFNSLLearner,
# )

agent_dict = {ag.agent_id: ag for ag in rm_agents}
env_dict = {f"E{i}": env for i, env in enumerate(envs)}

trainer = Trainer(env_dict, env_dict, agent_dict)
trainer.run(trainer_run_config)
