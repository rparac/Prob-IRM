import random
from functools import partial

import os
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

from rm_marl.new_stack.algos.algo import PPORMConfig, PPORMLearningConfig
from rm_marl.new_stack.callbacks.minimize_logs import MinimizeLogs
from rm_marl.new_stack.callbacks.callback_composer import CallbackComposer
from rm_marl.new_stack.callbacks.crash_after_n_iters import CrashAfterNIters
from rm_marl.new_stack.callbacks.env_render_callback import EnvRenderCallback
from rm_marl.new_stack.callbacks.heatmap_callback import HeatmapCallback
from rm_marl.new_stack.callbacks.log_original_reward import LogOriginalReward
from rm_marl.new_stack.callbacks.log_rm_learning import LogRMLearning
from rm_marl.new_stack.callbacks.store_config import StoreTracesCallback
from rm_marl.new_stack.env.multi_env_with_rm import make_multi_agent_with_rm
from rm_marl.new_stack.utils.env import hydra_env_creator
from rm_marl.new_stack.utils.hydra import from_hydra_config, manual_value_override
from rm_marl.new_stack.utils.run import (
    simplified_custom_run_rllib_example_script_experiment,
    continue_training,
)


from ray.rllib.algorithms import PPOConfig, PPO
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec
from ray.tune.schedulers import ASHAScheduler

stop_iters = 1000
run_config = {
    "should_tune": True,
    "wandb": {
        "project": "Continous-PPO",
        "run_name": "run",
        "key": "680ad332869d9761ae2b6bdd70cdbc068674d47b",
    },
    "name": "test_continous",
    "tune_config": {
        "num_samples": 1,
        "verbose": 2,
        "checkpoint_freq": 0,
        "checkpoint_at_end": False,
    },
}


def create_config():
    # n_timesteps
    # MeanStdFilter (observation filter)

    config = PPOConfig()
    config.training(
        train_batch_size=256,
        clip_param=0.1,
        entropy_coeff=0.00429,
        lambda_=0.9,
        gamma=0.9999,
        lr=7.77e-05,
        grad_clip=5,
        num_sgd_iter=10,
    )
    config.environment(
        env="MountainCarContinuous-v0",
        env_config={
            "render_mode": "rgb_array",
        }
    )
    config.env_runners(
        # num_env_runners=0,
        num_envs_per_env_runner=1,
        # rollout_fragment_length=8,
        observation_filter="MeanStdFilter",
    )
    # config.evaluation(
    #     evaluation_interval=100,
    #     evaluation_duration=1,
    #     # Important: Otherwise the evaluation runs in the main thread, which ruins environment ids
    #     # evaluation_num_env_runners=run_config["num_agents"],
    #     evaluation_num_env_runners=0,
    #     evaluation_duration_unit="episodes",
    # )
    config.debugging(logger_config={
        "type": tune.logger.NoopLogger,
    })
    config.callbacks(EnvRenderCallback)
    # config.rollouts(
    # num_envs_per_worker=1,
    # timesteps_per_iteration=20000,
    # observation_filter="MeanStdFilter",  # Normalizes observations
    # )

    # model = ({"log_std_init": -3.29, "ortho_init": False},)

    config.rl_module(
        model_config={
            "log_std_init": -3.29,
        }
    )
    return config


def run():
    print("hello world")

    config = create_config()

    scheduler = ASHAScheduler(
        # metric="env_runners/episode_return_mean",
        mode="max",
        grace_period=15,
        max_t=stop_iters,
    )
    stop = {
        TRAINING_ITERATION: stop_iters,
    }
    simplified_custom_run_rllib_example_script_experiment(
        config, run_config, stop=stop, scheduler=scheduler
    )


if __name__ == "__main__":
    run()
