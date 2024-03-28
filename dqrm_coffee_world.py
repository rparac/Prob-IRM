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
from rm_marl.envs.gym_subgoal_automata_wrapper import GymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import NoisyLabelingFunctionComposer, AutomataWrapper, RewardMachineWrapper, \
    ProbabilisticRewardShaping
from rm_marl.rm_learning.ilasp.noisy_learner.ProbFFNSLLearner import ProbFFNSLLearner
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from rm_marl.trainer import Trainer


def _get_base_env(env_name, seed, agent_id, label_factories):
    rm_transitioner = ProbRMTransitioner(rm=RewardMachineAgent.default_rm())

    # env=gym.make(
    env = gym.make(env_name,
            params={"generation": "random", "environment_seed": seed, "hide_state_variables": True})
    env = GymSubgoalAutomataAdapter(env, agent_id, render_mode="rgb_array", max_episode_length=250)  # type: ignore
    labeling_funs = []
    for label_factory in label_factories:
        labeling_funs.append(label_factory(env))
    env = NoisyLabelingFunctionComposer(labeling_funs)

    # AutomataWrapper here only provides the filter_label function (used in counter_factual update).
    #  It also logs RM states
    env = RewardMachineWrapper(
        env,
        rm_transitioner=rm_transitioner,
        label_mode=AutomataWrapper.LabelMode.ALL,
        termination_mode=AutomataWrapper.TerminationMode.ENV,
    )

    env = ProbabilisticRewardShaping(env, shaping_rm=rm_transitioner.rm, discount_factor=0.99)
    env = RecordEpisodeStatistics(env)  # type: ignore

    return env


# Hacky solution. In the ideal world we could just set one value and use $ interpolation for the rest
# Hydra doesn't support overriding multiple values at once with optuna
#  We override values that should be overriden with this function
def _manual_value_override(cfg):
    if 'manual_overrides' not in cfg:
        return

    override_values = cfg['manual_overrides']

    for override_value in override_values:
        if override_value in cfg["env"]["overridable"]:
            command = f"cfg.{override_value} = {cfg['x']}"
            exec(command)


@hydra.main(version_base=None, config_path="new_conf", config_name="config")
def run(cfg: DictConfig) -> int:
    _manual_value_override(cfg)

    run_config = cfg['run']
    print(run_config)

    np.random.seed(run_config["seed"])
    random.seed(run_config["seed"])

    env_config = cfg["env"]
    label_factories = [instantiate(label_factory_conf) for label_factory_conf in env_config["label_factories"]]

    agent_config = env_config["agent"]

    envs = []
    rm_agents = []
    num_envs = 10
    for i in range(num_envs):
        agent_id = f"A{i + 1}"
        env = _get_base_env(env_config["name"], run_config["seed"] + i, agent_id, label_factories)
        envs.append(env)
        # algo = DeepQRM(
        #     action_space=env.action_space,
        #     obs_space=env.observation_space,
        #     use_crm=True,
        #     num_policy_layers=agent_config["num_policy_layers"],
        #     policy_layer_size=agent_config["policy_layer_size"],
        #     gamma=agent_config["gamma"],
        #     epsilon_start=agent_config["epsilon_start"],
        #     epsilon_end=agent_config["epsilon_end"],
        #     epsilon_decay=agent_config["epsilon_decay"],
        # )

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
        rm_learner_kws={
            "edge_cost": run_config["edge_cost"],
            "n_phi_cost": run_config["n_phi_cost"],
            "ex_penalty_multiplier": run_config["ex_penalty_multiplier"],
        }
    )

    agent_dict = {ag.agent_id: learning_ag for ag in rm_agents}
    env_dict = {f"E{i}": env for i, env in enumerate(envs)}

    trainer = Trainer(env_dict, env_dict, agent_dict)
    return trainer.run(run_config)


if __name__ == '__main__':
    run()