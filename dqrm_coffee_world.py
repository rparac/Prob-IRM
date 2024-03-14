import os
import random

import gym
import hydra
import numpy as np
from gym.wrappers import RecordEpisodeStatistics
from hydra.utils import instantiate
from omegaconf import DictConfig

from rm_marl.agent import RewardMachineAgent, RewardMachineLearningAgent
from rm_marl.algo import QRM
from rm_marl.algo.deepqrm import DeepQRM
from rm_marl.envs.gym_subgoal_automata_wrapper import GymSubgoalAutomataAdapter, \
    OfficeWorldOfficeLabelingFunctionWrapper, \
    OfficeWorldPlantLabelingFunctionWrapper, OfficeWorldCoffeeLabelingFunctionWrapper
from rm_marl.envs.wrappers import NoisyLabelingFunctionComposer, AutomataWrapper, RewardMachineWrapper
from rm_marl.rm_learning.ilasp.noisy_learner.ProbFFNSLLearner import ProbFFNSLLearner
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from rm_marl.trainer import Trainer


def _get_base_env(seed, agent_id, label_factories):
    env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
                   params={"generation": "random", "environment_seed": seed, "hide_state_variables": True})
    env = GymSubgoalAutomataAdapter(env, agent_id, render_mode="rgb_array", max_episode_length=250)  # type: ignore
    labeling_funs = []
    for label_factory in label_factories:
        labeling_funs.append(label_factory(env))
    env = NoisyLabelingFunctionComposer(labeling_funs)

    rm_transitioner = ProbRMTransitioner(rm=None)
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


# Hacky solution. In the ideal world we could just set one value and use $ interpolation for the rest
# Hydra doesn't support overriding multiple values at once with optuna
#  We override values that should be overriden with this function
def _manual_value_override(cfg):
    override_values = cfg['manual_overrides']

    for override_value in override_values:
        command = f"cfg.{override_value} = {cfg['x']}"
        exec(command)



@hydra.main(version_base=None, config_path="new_conf", config_name="config")
def run(cfg: DictConfig) -> int:
    # print(cfg['env'])
    print(cfg)
    print(cfg['manual_overrides'])

    _manual_value_override(cfg)

    print(cfg['env']['coffee_label_factory'])

    return 0


    run_config = cfg['run']

    np.random.seed(run_config["seed"])
    random.seed(run_config["seed"])

    env_config = cfg["env"]
    label_factories = [instantiate(label_factory_conf) for label_factory_conf in env_config["label_factories"]]

    envs = []
    rm_agents = []
    num_envs = 10
    for i in range(num_envs):
        agent_id = f"A{i + 1}"
        env = _get_base_env(run_config["seed"] + i, agent_id, label_factories)
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
        rm_agent = RewardMachineAgent.default_rm_agent(env.agent_id, env.rm_transitioner, algo)
        rm_agents.append(rm_agent)

    learning_ag = RewardMachineLearningAgent(
        rm_agent=rm_agents,
        rm_learner_cls=ProbFFNSLLearner,
    )

    agent_dict = {ag.agent_id: learning_ag for ag in rm_agents}
    env_dict = {f"E{i}": env for i, env in enumerate(envs)}

    trainer = Trainer(env_dict, env_dict, agent_dict)
    return trainer.run(run_config)


if __name__ == '__main__':
    run()
