"""
This file showcases how code is intended to be used in the single agent case.
Hydra configuration helps with duplicate configuration, but is not immediately clear to a user.
"""
import os
import random
import time
from typing import Tuple

import gym
import hydra
import numpy as np
import torch
from gym.core import ObsType, ActType
from gym.spaces import Discrete
from gym.wrappers import RecordEpisodeStatistics
from omegaconf import DictConfig

from rm_marl.agent import NoRMAgent, RewardMachineLearningAgent
from rm_marl.algo.deepqrm import DeepQRM
from rm_marl.envs.wrappers import AutomataWrapper, LabelingFunctionWrapper
from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_transition.deterministic_rm_transitioner import DeterministicRMTransitioner
from rm_marl.trainer import Trainer
from test_dqrm import DiscreteActions, DictObservation, InvertedPendulumLabelingFunctionWrapper


@hydra.main(version_base=None, config_path="new_conf", config_name="config")
def run(cfg: DictConfig) -> None:
    print(cfg)
    run_config = cfg['run']

    np.random.seed(run_config["seed"])
    random.seed(run_config["seed"])

    env_config = cfg["env"]

    env = gym.make(env_config["name"], render_mode=env_config["render_mode"],
                   max_episode_steps=env_config["max_episode_steps"])
    env = DiscreteActions(env, agent_id="A1")
    env = DictObservation(env, agent_id="A1")
    env = InvertedPendulumLabelingFunctionWrapper(env)

    # # env = AutomataWrapper(env, DeterministicRMTransitioner(rm=None))
    env = RecordEpisodeStatistics(env)  # type: ignore

    agent_config = env_config["agent"]
    # TODO: make it less explicit; i.e. just pass the dictionary directly
    ag = NoRMAgent(
        agent_id="A1",
        algo_cls=DeepQRM,
        algo_kws={
            "obs_space": env.observation_space,
            "action_space": env.action_space,
            "num_policy_layers": agent_config["num_policy_layers"],
            "policy_layer_size": agent_config["policy_layer_size"],
            "gamma": agent_config["gamma"],
            "epsilon_start": agent_config["epsilon_start"],
            "epsilon_end": agent_config["epsilon_end"],
            "epsilon_decay": agent_config["epsilon_decay"],
        },
    )

    agent_dict = {"A1": ag}
    env_dict = {"E": env}
    trainer = Trainer(env_dict, env_dict, agent_dict)
    return trainer.run(run_config)


if __name__ == "__main__":
    run()
