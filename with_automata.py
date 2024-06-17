import json
import os
import cProfile
import random
import tracemalloc

import gym
import hydra
import numpy as np
from gym.wrappers import RecordEpisodeStatistics
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from rm_marl.agent import RewardMachineAgent, RewardMachineLearningAgent
from rm_marl.algo import QRM
from rm_marl.algo.deepqrm import DeepQRM
from rm_marl.envs.gym_subgoal_automata_wrapper import GymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import NoisyLabelingFunctionComposer, AutomataWrapper, RewardMachineWrapper, \
    ProbabilisticRewardShaping
from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_learning.ilasp.noisy_learner.ProbFFNSLLearner import ProbFFNSLLearner
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from rm_marl.trainer import Trainer


def _get_base_env(env_name, seed, agent_id, label_factories, render_mode, max_episode_length, use_rs,
                  use_restricted_observables):
    env = gym.make(env_name,
                   params={"generation": "random", "environment_seed": seed, "hide_state_variables": True})
    env = GymSubgoalAutomataAdapter(env, agent_id, render_mode=render_mode,  # type: ignore
                                    max_episode_length=max_episode_length,
                                    use_restricted_observables=use_restricted_observables)

    rm = env.get_perfect_rm()
    rm_transitioner = ProbRMTransitioner(rm=rm)

    labeling_funs = []
    for label_factory in label_factories:
        labeling_funs.append(label_factory(env))
    env = NoisyLabelingFunctionComposer(labeling_funs)

    # TODO: this is hacky but RecordEpisodeStatisics needs to be here for metrics tracking
    env = RecordEpisodeStatistics(env)  # type: ignore
    # AutomataWrapper here only provides the filter_label function (used in counter_factual update).
    #  It also logs RM states
    env = RewardMachineWrapper(
        env,
        rm_transitioner=rm_transitioner,
        label_mode=AutomataWrapper.LabelMode.ALL,
        termination_mode=AutomataWrapper.TerminationMode.ENV,
    )

    if use_rs:
        env = ProbabilisticRewardShaping(env, shaping_rm=rm_transitioner.rm, discount_factor=0.99)

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

    # agent_config = cfg["algo"]

    label_factories = [instantiate(label_factory_conf) for label_factory_conf in env_config["core_label_factories"]]

    if not env_config["use_restricted_observables"]:
        noisy_label_factories = [instantiate(label_factory_conf) for label_factory_conf in
                                 env_config["noise_label_factories"]]
        label_factories.extend(noisy_label_factories)

    # rm = RewardMachine.load_from_file(automata)
    # rm_transitioner = ProbRMTransitioner(rm=rm)

    print(env_config)
    envs = []
    rm_agents = []
    for i in range(run_config["num_envs"]):
        agent_id = f"A{i + 1}"
        env = _get_base_env(env_config["name"], run_config["seed"] + i, agent_id, label_factories,
                            env_config["render_mode"], env_config["max_episode_length"], run_config["use_rs"],
                            env_config["use_restricted_observables"])
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
            "seed": run_config["seed"],
            **cfg["algo"],
        })

        rm_agent = RewardMachineAgent(env.agent_id, env.rm_transitioner, algo)
        rm_agents.append(rm_agent)

    agent_dict = {ag.agent_id: ag for ag in rm_agents}
    env_dict = {f"E{i}": env for i, env in enumerate(envs)}

    trainer = Trainer(env_dict, agent_dict, env_config)
    result = trainer.run(run_config)
    print(f"Result for this session was {result}")
    return result


if __name__ == '__main__':
    run()
