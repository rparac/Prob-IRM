"""
Trains a PPO agent with a provided reward machine with a configurable number of agents.
It doesn't use ray tune to do so, but it looks like it works.


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --env [env name e.g. 'ALE/Pong-v5']
--wandb-key=[your WandB API key] --wandb-project=[some WandB project name]
--wandb-run-name=[optional: WandB run name within --wandb-project]`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.



There are 2 important hyperparameters that we haven't tuned:
 - entropy_coeff_schedule
 - learning_rate_schedule
They may be needed if we are not getting good enough results
"""
import random
from functools import partial

import os
import gymnasium
import hydra
import numpy as np
from ray import tune
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms import PPOConfig, PPO
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler

from rm_marl.experiments.callbacks.visualize_labelling_fn import VisualizeLabelling
from rm_marl.experiments.networks.simple import SimpleNN
from rm_marl.new_stack.callbacks.callback_composer import CallbackComposer
from rm_marl.new_stack.callbacks.env_render_callback import EnvRenderCallback
from rm_marl.new_stack.env.multi_env_with_rm import make_multi_agent_with_rm
from rm_marl.new_stack.env.visual_minecraft.env import GridWorldEnv
from rm_marl.new_stack.networks.recurrent import CustomLSTMModule
from rm_marl.new_stack.utils.env import hydra_env_creator, simple_env_creator
from rm_marl.new_stack.utils.hydra import from_hydra_config, manual_value_override
from rm_marl.new_stack.utils.run import simplified_custom_run_rllib_example_script_experiment, continue_training


def create_config():
    config = PPOConfig()

    config = (
        config.environment(
            env="env",
            env_config={
                "num_agents": 1, 
            },
        )
        .training(
            lr=1e-4,
        )
        .training(
            train_batch_size_per_learner=32768,
            grad_clip_by="global_norm",
            minibatch_size=8192,
            lambda_=0.95,
            vf_loss_coeff=0.5,
            grad_clip=0.5,
            clip_param=0.2,
            vf_clip_param=10.0,
            kl_target=0.01,
            kl_coeff=0.2,
            num_epochs=13,
            entropy_coeff=0.1,
            use_kl_loss=False
        )
        .env_runners(
            batch_mode="complete_episodes",
            # num_env_runners=0, # forces everything to be done on the local worker
            # num_env_runners= run_config["num_agents"],
            num_env_runners=40,
            num_envs_per_env_runner=1,
            # By default, environments are stepped one at a time
            # https://docs.ray.io/en/latest/rllib/rllib-env.html
            # https://docs.ray.io/en/latest/rllib/package_ref/env.html
            # Doesn't make a lot of difference; seems like it's not used tbf
            # remote_worker_envs=False,
        )
        .evaluation(
            # TODO: paramterize
            evaluation_interval=5,
            evaluation_duration=1, 
            # Important: Otherwise the evaluation runs in the main thread, which ruins environment ids
            # evaluation_num_env_runners=run_config["num_agents"],
            evaluation_num_env_runners=0,
            evaluation_duration_unit="episodes",
        )
        .callbacks(
            EnvRenderCallback
        )
        .debugging(seed=0, log_level="WARN")
    )

    def policy_mapping_fn_(aid, episode, **kwargs):
        return f"p{aid}"

    policies = {
        f"p{i}"
        for i in range(1)
    }
    config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn_,
        # Check if policy_states_are_swappable should be true
    )


    model_config = {
        "fcnet_hiddens": [64, 64, 64, 64],
        "fcnet_activation": "relu",
    }

    module_specs = {
        f"p{i}": RLModuleSpec(
        )
        for i in range(1)
    }

    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(rl_module_specs=module_specs),
        # IMPORTANT: the model config dict needs to be defined here; it gets ignored if defined for individual policies.
        #   Noticed when resetting workers
        # model_config_dict=from_hydra_config(model_config)
        model_config=model_config,
    )

    return config


@hydra.main(version_base=None, config_path="../even_newer_conf", config_name="config")
def run(cfg: DictConfig) -> int:
    manual_value_override(cfg)

    np.random.seed(0)
    random.seed(0)

    env_config = setup_env_config(cfg["env"])
    register_env("env", make_multi_agent_with_rm(simple_env_creator(env_config)))

    base_config = create_config()
    stop = {
        TRAINING_ITERATION: 25,
    }

    run_config = {
        "should_tune": True,
        "wandb": {
            "project": "No-Labelling",
            "run_name": "run",
            "key": "680ad332869d9761ae2b6bdd70cdbc068674d47b",
        },
        "name": "test-simple",
        "tune_config": {
            "num_samples": 1,
            "verbose": 2,
            "checkpoint_freq": 0,
            "checkpoint_at_end": True,
        },
    }


    simplified_custom_run_rllib_example_script_experiment(base_config, run_config, stop=stop)

    return 0


def setup_env_config(env_config):
    label_factories = [instantiate(label_factory_conf) for label_factory_conf in env_config["core_label_factories"]]

    if not env_config["use_restricted_observables"]:
        noisy_label_factories = [instantiate(label_factory_conf) for label_factory_conf in
                                 env_config["noise_label_factories"]]
        label_factories.extend(noisy_label_factories)
    print(env_config)
    env_config = OmegaConf.to_container(env_config, resolve=True)
    env_config["label_factories"] = label_factories
    env_config["seed"] = 0
    env_config["use_rs"] = False
    # env_config["rs_discount"] = run_config["rs_discount"]
    env_config["use_thresholding"] = False
    # env_config["labelling_threshold"] = False
    return env_config


if __name__ == "__main__":
    run()
