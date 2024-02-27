"""
This file showcases how code is intended to be used in the single agent case.
Hydra configuration helps with duplicate configuration, but is not immediately clear to a user.
"""
import os
import random
import time
from typing import Tuple

import gym
import numpy as np
import torch
from gym.core import ObsType, ActType
from gym.spaces import Discrete
from gym.wrappers import RecordEpisodeStatistics

from rm_marl.agent import NoRMAgent, RewardMachineLearningAgent
from rm_marl.algo.deepqrm import DeepQRM
from rm_marl.envs.wrappers import AutomataWrapper, LabelingFunctionWrapper
from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_transition.deterministic_rm_transitioner import DeterministicRMTransitioner
from rm_marl.trainer import Trainer


class DiscreteActions(gym.ActionWrapper):
    def __init__(self, env, agent_id: str):
        super().__init__(env)
        self.action_space = Discrete(3)
        self.agent_id = agent_id

    def action(self, action):
        act = np.array([(action[self.agent_id] - 1) * 2])
        return act


class DictObservation(gym.ObservationWrapper):
    def __init__(self, env, agent_id: str):
        super().__init__(env)
        self.observation_space = gym.spaces.Dict(
            {"A1": env.observation_space},
        )
        self.agent_id = agent_id

    def observation(self, observation):
        return {self.agent_id: observation}


class InvertedPendulumLabelingFunctionWrapper(LabelingFunctionWrapper):
    def __init__(self, env):
        super().__init__(env)

    def get_labels(self, obs: dict, prev_obs: dict):
        return []


if __name__ == '__main__':
    seed = 123

    trainer_run_config = {
        "training": True,
        "total_episodes": 50000,  # 100,
        "log_freq": 1,
        "log_dir": os.path.join(os.path.dirname(__file__), "logs"),
        "testing_freq": 50,  # 10,
        "greedy": True,
        "synchronize": False,
        "counterfactual_update": False,
        "recording_freq": 50,  # 5
        "seed": seed,
        "name": "inverted-pendulum",
        "extra_debug_information": False,
    }
    np.random.seed(trainer_run_config["seed"])
    random.seed(trainer_run_config["seed"])

    env = gym.make('InvertedPendulum-v4', render_mode='rgb_array', max_episode_steps=200)
    env = DiscreteActions(env, agent_id="A1")
    env = DictObservation(env, agent_id="A1")
    env = InvertedPendulumLabelingFunctionWrapper(env)
    # env = AutomataWrapper(env, DeterministicRMTransitioner(rm=None))
    env = RecordEpisodeStatistics(env)  # type: ignore

    ag = NoRMAgent(
        agent_id="A1",
        algo_cls=DeepQRM,
        algo_kws={
            "obs_space": env.observation_space,
            "action_space": env.action_space,
            # "num_policy_layers": 3,
            # "policy_layer_size": 16,
            # "gamma": 0.99,
            "epsilon_start": 1,
            "epsilon_end": 0,
            "epsilon_decay": 10,
        },
    )

    t = time.time()

    agent_dict = {"A1": ag}
    env_dict = {"E": env}
    trainer = Trainer(env_dict, env_dict, agent_dict)
    trainer.run(trainer_run_config)

    print(f"Execution took {time.time() - t} seconds")
