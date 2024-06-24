import gymnasium as gym
from dotenv import load_dotenv
from optuna.distributions import FloatDistribution
from ray.tune import register_env
from ray.tune.logger import TBXLogger

from rllib_example_test import EnvRenderCallback
from rm_marl.trainer.ray.run_args import Args
from rm_marl.envs.gym_subgoal_automata_wrapper import GymSubgoalAutomataAdapter

load_dotenv()

import logging
import os
import random
import sys
import warnings

import numpy as np
import ray
import tyro
from ray import air, train, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
)

torch, nn = try_import_torch()

warnings.filterwarnings("ignore", module="gymnasium.core")
warnings.filterwarnings("ignore", module="ray.rllib.policy")

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

SEED = 123


# ENV_NAME = "FOLMultiRoom-OneRoom-v1"


def create_new_config(
        num_workers=10,
        seed=SEED,
):
    def env_creator(env_config):
        env = gym.make('gym_subgoal_automata:OfficeWorldDeliverCoffee-v0',
                       params={"generation": "random", "environment_seed": seed, "hide_state_variables": True},
                       render_mode="rgb_array",
                       )
        env = GymSubgoalAutomataAdapter(env, agent_id="A1", max_episode_length=200, use_restricted_observables=True)
        return env

    register_env(f"my_env", env_creator)

    dummy_env = env_creator({})

    config = PPOConfig()
    config = (
        config.environment(
            env=f"my_env",
            env_config={
            },
        )
        .env_runners(
            num_env_runners=num_workers,
            observation_filter="MeanStdFilter",
            create_env_on_local_worker=True,
            # enable_connectors=False,
        )
        .framework("torch")
        .api_stack(
            enable_env_runner_and_connector_v2=True,
            enable_rl_module_and_learner=True,
        )
        # .callbacks(EnvRenderCallback)
        .training(
            model={
                "fcnet_hiddens": [64, 64],
                "fcnet_activation": "tanh",
                "vf_share_layers": True,
            },
            entropy_coeff=0.02,
            lr=2.5e-4,
            # lr_schedule=[[0, 2.5e-4], [200000, 1e-4], [500000, 5e-5], [1000000, 1e-5]],
            gamma=0.99,
            vf_loss_coeff=0.5,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=0.2,
            grad_clip=0.5,
            kl_target=0.0,
            num_sgd_iter=10,
            sgd_minibatch_size=(num_workers * (int(200) // 8)) // 4,
            train_batch_size=num_workers * (int(200) // 8),
            use_critic=True,
            use_gae=True,
        )
        .evaluation(
            evaluation_interval=10,  # Evaluate every 50 episodes
            evaluation_duration=1,  # 1 episode
            evaluation_duration_unit="episodes",
            evaluation_config=PPOConfig.overrides(
                entropy_coeff=0.0,
                explore=False,
                render_env=True,
            ),
        )
        .debugging(seed=seed, log_level="WARN")
    )
    return config


def create_config(
        num_workers=10,
        seed=SEED,
):
    config = PPOConfig()
    # dummy_env_ = make_env(f"rm-marl/{ENV_NAME}", 0, False, "test")(env_config)

    hiddens = [64, 64]
    activation = "tanh"

    max_steps = 200
    config = (
        config.environment(
            env="CartPole-v1",
            env_config={
                "render_mode": "rgb_array",
            },
            is_atari=False,
            observation_space=None,
        )
        .resources(
            # num_gpus=1,
        )
        .debugging(
            logger_creator=(lambda: TBXLogger()),
        )
        .env_runners(
            # enable_connectors=False, new api stack
        )
        # .callbacks(RewardThresholdNumIterationsCallback)
        .env_runners(
            num_env_runners=num_workers,
            observation_filter="MeanStdFilter",
        )
        .framework("torch")
        .training(
            model={
                "fcnet_hiddens": hiddens,
                "fcnet_activation": activation,
                "vf_share_layers": True,
            },
            entropy_coeff=0.02,
            lr=2.5e-4,
            # lr_schedule=[[0, 2.5e-4], [200000, 1e-4], [500000, 5e-5], [1000000, 1e-5]],
            gamma=0.99,
            vf_loss_coeff=0.5,
            lambda_=0.95,
            clip_param=0.2,
            vf_clip_param=0.2,
            grad_clip=0.5,
            kl_target=0.0,
            num_sgd_iter=10,
            sgd_minibatch_size=(num_workers * (int(max_steps) // 8)) // 4,
            train_batch_size=num_workers * (int(max_steps) // 8),
            use_critic=True,
            use_gae=True,
        )
        # .resources(
        #     num_cpus_per_learner_worker=1,
        #     num_gpus_per_learner_worker=1,
        # )
        # .resources(
        #     num_gpus=1 # turn on for GPU
        # )
        .evaluation(
            evaluation_interval=10,  # Evaluate every 50 episodes
            evaluation_duration=1,  # 1 episode
            evaluation_duration_unit="episodes",
            evaluation_config=PPOConfig.overrides(
                entropy_coeff=0.0,
                explore=False,
            ),
        )
        .debugging(seed=seed, log_level="WARN")
        # .api_stack(
        #     # enable_env_runner_and_connector_v2=True,
        #     # enable_rl_module_and_learner=True
        #     enable_rl_module_and_learner=False,
        # )

    )
    return config


def hyperparams_opt(
        num_iterations=400,
        seed=SEED,
        points_to_evaluate=None,
        num_samples=1,
        max_concurrent_trials=None,
):
    from ray.tune.schedulers import (
        create_scheduler,
    )
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
    # scheduler_name = "medianstopping"
    # scheduler_name = "hyperband"

    scheduler = create_scheduler(
        scheduler_name,
        time_attr="training_iteration",
        perturbation_interval=50,
        resample_probability=0.1,
        # Specifies the mutations of these hyperparams
        hyperparam_bounds=hyperparam_bounds,
        hyperparam_mutations=hyperparam_mutations,
        max_t=num_iterations,
        grace_period=int(num_iterations / 10 * 2),
        # hard_stop=False
    )

    if points_to_evaluate:
        points_to_evaluate = [
            {k: v for k, v in d.items() if k in hyperparam_mutations}
            for d in points_to_evaluate
        ]

    return dict(
        tune_config=tune.TuneConfig(
            metric=f"{ENV_RUNNER_RESULTS}/episode_return_mean",
            mode="max",
            scheduler=scheduler,
            num_samples=num_samples,
            max_concurrent_trials=max_concurrent_trials,
            reuse_actors=False,
            # search_alg=BasicVariantGenerator(random_state=seed),
            search_alg=OptunaSearch(
                space=hyperparam_mutations,
                metric=[
                    f"{ENV_RUNNER_RESULTS}/episode_return_mean",
                    f"{ENV_RUNNER_RESULTS}/episode_return_min",
                    f"{ENV_RUNNER_RESULTS}/episode_return_max",
                ],
                mode=["max", "max", "max"],
                seed=seed,
                points_to_evaluate=points_to_evaluate,
            ),
        )
    )


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ray.init(
        num_gpus=1,
        runtime_env={
            "env_vars": ({"RAY_DEBUG": f"{str(int(args.debug))}"} if args.debug else {}),
        }
    )

    config = create_new_config(
        seed=args.seed,
        num_workers=args.num_workers,
    )
    algo = config.build()
    algo.evaluate()

    stop = tune.stopper.CombinedStopper(
        tune.stopper.MaximumIterationStopper(max_iter=args.num_iterations),
    )

    os.environ['PYDEVD_DEBUG'] = 'True'
    print(os.environ['CUBLAS_WORKSPACE_CONFIG'])

    tuner = tune.Tuner(
        config.algo_class,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            storage_path="~/ray_results/debug" if args.debug else None,
            stop=stop,
            verbose=1,
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=1000,
                num_to_keep=1,
                checkpoint_score_order="max",
                checkpoint_score_attribute=f"{ENV_RUNNER_RESULTS}/episode_return_mean",
                checkpoint_at_end=True,
            ),
            failure_config=train.FailureConfig(fail_fast=args.debug),
            sync_config=train.SyncConfig(sync_artifacts=True),
        ),
        **hyperparams_opt(
            num_iterations=args.num_iterations,
            seed=args.seed,
            max_concurrent_trials=args.max_concurrent_trials,
        ),
    )
    tuner.fit()

    results = tuner.get_results()
    best_result = results.get_best_result(
        metric="episode_return_mean", mode="min", scope="last"
    )
    print("CONFIG:", best_result.config)

    ckpt = best_result.checkpoint
    algo = config.algo_class.from_checkpoint(ckpt)
    eval_results = algo.evaluate()
    print(eval_results)
