import os
import random
import re
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import gymnasium as gym
import numpy as np
import optuna
import ray
from optuna.distributions import UniformDistribution, FloatDistribution
from ray import air, train, tune
from ray.air.constants import TRAINING_ITERATION
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.models.torch.misc import SlimFC, normc_initializer
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME, ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN
from ray.rllib.utils.typing import ResultDict
from ray.tune import CLIReporter, register_env

torch, nn = try_import_torch()

if TYPE_CHECKING:
    from ray.rllib.algorithms import AlgorithmConfig


@dataclass
class Args:
    experiment: str = "exp1"

    hyperparam: bool = True
    resume: bool = False

    rm: bool = False
    rm_learning: bool = False
    max_rm_num_states: int = 6
    rm_learning_freq: int = 1
    shared_hidden_layers: tuple = (1,)
    shared_policy: bool = False
    reward_shaping: bool = False
    extend_obs_space: bool = True

    num_iterations: int = 400
    num_timesteps: int = int(1e10)
    reward_threshold: float = 0.98

    env: str = "CartPole-v1"
    seed: int = 123
    debug: bool = False
    local: bool = False
    num_workers: int = 48
    max_concurrent_trials: Optional[int] = None
    num_samples: int = 50

    wandb_project: Optional[str] = "ppo-test"
    wandb_run_name: str = "some_run_name"
    # TODO: security issue; remove
    wandb_key: str = "7aa07ea83aacbc6521c091eb60561b8637a7f512"

    ###### fixed

    hidden_dim: int = 64
    hidden_count: int = 2
    activation: str = "tanh"
    vf_share_layers: bool = True

    def __post_init__(self):
        if not self.hyperparam:
            assert self.experiment, "You must select an experiment to load"
            assert not self.resume, "You cannot resume"

        assert not self.shared_hidden_layers or (
                self.shared_hidden_layers and not self.shared_policy
        ), "cannot have shared layers and shared policy simultaneously"


def hyperparams_opt(
        num_iterations=400,
        seed=123,
        points_to_evaluate=None,
        num_samples=50,
        max_concurrent_trials=None,
):
    from ray.tune.schedulers import (
        AsyncHyperBandScheduler,
        PopulationBasedTraining,
        create_scheduler,
    )
    from ray.tune.search import BasicVariantGenerator
    from ray.tune.search.optuna import OptunaSearch

    hyperparam_bounds = {
        "lambda": (0.9, 1.0),
        "clip_param": (0.01, 0.5),
        "vf_clip_param": (0.01, 0.5),
        "entropy_coeff": (0.0, 0.1),
        "kl_target": (0.0, 0.2),
    }
    hyperparam_mutations = {k: FloatDistribution(*v) for k, v in hyperparam_bounds.items()}

    scheduler_name = "asynchyperband"
    scheduler = create_scheduler(
        scheduler_name,
        time_attr="training_iteration",
        max_t=num_iterations,  # max time units per trial
        grace_period=int(num_iterations / 10 * 2),  # for early stopping
    )

    if points_to_evaluate:
        points_to_evaluate = [
            {k: v for k, v in d.items() if k in hyperparam_mutations}
            for d in points_to_evaluate
        ]

    search_alg = OptunaSearch(space=hyperparam_mutations,
                              metric=[f"episode_return_mean", f"episode_return_min", f"episode_return_max", ],
                              mode=["max", "max", "max"], seed=seed, points_to_evaluate=points_to_evaluate, )
    tune_config = tune.TuneConfig(
        metric=f"episode_return_mean",
        mode="max",
        scheduler=scheduler,
        num_samples=num_samples,
        max_concurrent_trials=max_concurrent_trials,
        reuse_actors=False,
        # search_alg=BasicVariantGenerator(random_state=seed),
        search_alg=search_alg,
    )
    return dict(
        tune_config=tune_config
    )


def new_run_rllib_experiment():
    pass


def run_rllib_experiment(
        base_config: "AlgorithmConfig",
        args: Args,
        *,
        tune_callbacks: Optional[List] = None,
) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    stop = {
        # f"{ENV_RUNNER_RESULTS}/episode_return_mean": args.reward_threshold,
        # f"{NUM_ENV_STEPS_SAMPLED_LIFETIME}": args.num_timesteps,
        TRAINING_ITERATION: args.num_iterations,
    }

    config = base_config

    # Run the experiment using Ray Tune.

    # Log results using WandB.
    tune_callbacks = tune_callbacks or []
    if hasattr(args, "wandb_key") and args.wandb_key is not None:
        project = args.wandb_project
        # tune_callbacks.append(
        #     WandbLoggerCallback(
        #         api_key=args.wandb_key,
        #         project=project,
        #         upload_checkpoints=True,
        #         **({"name": args.wandb_run_name} if args.wandb_run_name else {}),
        #     )
        # )

    # Auto-configure a CLIReporter (to log the results to the console).
    # Use better ProgressReporter for multi-agent cases: List individual policy rewards.
    progress_reporter = None
    # if args.num_agents > 0:
    # progress_reporter = CLIReporter(
    #     metric_columns={
    #         **{
    #             TRAINING_ITERATION: "iter",
    #             "time_total_s": "total time (s)",
    #             NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
    #             f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "return_mean",
    #         },
    #     },
    # )

    # Force Tuner to use old progress output as the new one silently ignores our custom
    # `CLIReporter`.
    # os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

    results = tune.Tuner(
        "PPO",
        # config.algo_class,
        # run_config=air.RunConfig(
        #     storage_path="~/ray_results/debug" if (args.debug or args.local) else None,
        #     stop=stop,
        #     verbose=1,
        #     callbacks=tune_callbacks,
        #     checkpoint_config=train.CheckpointConfig(
        #         checkpoint_frequency=1,
        #         num_to_keep=1,
        #         checkpoint_score_order="max",
        #         checkpoint_score_attribute=f"{ENV_RUNNER_RESULTS}/episode_return_mean",
        #         checkpoint_at_end=True,
        #     ),
        #     failure_config=train.FailureConfig(fail_fast=True),
        #     sync_config=train.SyncConfig(sync_artifacts=True),
        # ),
        # **hyperparams_opt(
        #     num_iterations=args.num_iterations,
        #     num_samples=args.num_samples,
        #     seed=args.seed,
        #     max_concurrent_trials=args.max_concurrent_trials,
        # ),
    ).fit()

    return results
