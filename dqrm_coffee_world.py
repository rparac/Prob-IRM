import os
import random

import gym
import hydra
import numpy as np
from gym.wrappers import RecordEpisodeStatistics
from omegaconf import DictConfig

from rm_marl.agent import RewardMachineAgent
from rm_marl.algo.deepqrm import DeepQRM
from rm_marl.envs.gym_subgoal_automata_wrapper import GymSubgoalAutomataAdapter, \
    OfficeWorldOfficeLabelingFunctionWrapper, \
    OfficeWorldPlantLabelingFunctionWrapper, OfficeWorldCoffeeLabelingFunctionWrapper
from rm_marl.envs.wrappers import NoisyLabelingFunctionComposer, AutomataWrapper
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from rm_marl.trainer import Trainer


@hydra.main(version_base=None, config_path="new_conf", config_name="config")
def run(cfg: DictConfig) -> None:
    run_config = cfg['run']

    np.random.seed(run_config["seed"])
    random.seed(run_config["seed"])

    # env_config ignored for now
    env_config = cfg["env"]

    env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
                   params={"generation": "random", "environment_seed": 7, "hide_state_variables": True})
    env = GymSubgoalAutomataAdapter(env, render_mode="rgb_array", max_episode_length=250)  # type: ignore

    office_l = OfficeWorldOfficeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
    plant_l = OfficeWorldPlantLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
    coffee_l = OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
    env = NoisyLabelingFunctionComposer([office_l, plant_l, coffee_l])

    rm = env.get_perfect_rm()
    rm_transitioner = ProbRMTransitioner(rm)
    env = AutomataWrapper(
        env,
        rm_transitioner=rm_transitioner,
        label_mode=AutomataWrapper.LabelMode.ALL,
        termination_mode=AutomataWrapper.TerminationMode.ENV,
    )
    env = RecordEpisodeStatistics(env)  # type: ignore

    agent_config = env_config["agent"]

    deepqrm = DeepQRM(**{
        "obs_space": env.observation_space,
        "action_space": env.action_space,
        "use_crm": True,
        **agent_config,
    })

    ag = RewardMachineAgent(
        # rm=rm,
        rm_transitioner=rm_transitioner,
        agent_id="A1",
        algo=deepqrm,
    )
    agent_dict = {"A1": ag}
    env_dict = {"E": env}

    trainer = Trainer(env_dict, env_dict, agent_dict)
    return trainer.run(run_config)


if __name__ == '__main__':
    run()
