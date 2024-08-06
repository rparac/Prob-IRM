"""
This file showcases how code is intended to be used in the single agent case.
Hydra configuration helps with duplicate configuration, but is not immediately clear to a user.
"""
import random

import gym
import hydra
import numpy as np
from gym.spaces import Discrete
from gym.wrappers import RecordEpisodeStatistics
from omegaconf import DictConfig

from rm_marl.agent import NoRMAgent
from rm_marl.algo.deepqrm import DeepQRM
from rm_marl.envs.wrappers import LabelingFunctionWrapper
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

    def get_labels(self, info: dict):
        return []


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
        algo=DeepQRM,
    )

    agent_dict = {"A1": ag}
    env_dict = {"E": env}
    trainer = Trainer(env_dict, agent_dict)
    return trainer.run(run_config)


if __name__ == "__main__":
    run()
