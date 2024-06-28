""""
Task:
 - given:
     - N environments

 - generate one env_worker per environment; execute PPO on it

 - must use v2
"""
from collections import Counter

from dotenv import load_dotenv
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env import EnvContext
from ray.rllib.env.multi_agent_env import make_multi_agent

from rm_marl.trainer.new_ray_utils.model import PPORMLearningCatalog

load_dotenv()

import logging
import random
import warnings
from dataclasses import dataclass

import gymnasium as gym
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
# warnings.filterwarnings("ignore", module="ray.rllib.policy")

# sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

logger = logging.getLogger(__name__)

SEED = 123


def make_env(env_id, idx, capture_video, run_name):
    # Keeps track the number of times the function is called
    counter = Counter()
    def thunk(_env_ctx: EnvContext):
        curr_id = counter["id"]
        counter["id"] += 1
        if (capture_video or _env_ctx.get("capture_video")) and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array",
                           params={"generation": "random", "environment_seed": SEED + curr_id,
                                   "hide_state_variables": True})
            env = gym.wrappers.RecordVideo(env, f"videos/{curr_id}/{run_name}")
        else:
            # env = gym.make("CartPole-v1")
            env = gym.make(env_id,
                           params={"generation": "random", "environment_seed": SEED + curr_id,
                                   "hide_state_variables": True})
        env = gym.wrappers.FlattenObservation(env)
        return env

    return thunk


tune.register_env(
    f"env",
    make_multi_agent(make_env('gym_subgoal_automata:OfficeWorldDeliverCoffee-v0', 0, False, "test"))
)

# dummy_env = make_multi_agent(make_env('gym_subgoal_automata:OfficeWorldDeliverCoffee-v0', 0, False, "test"))({"num_agents": 2, "seed": 0})
# dummy_env = make_env('gym_subgoal_automata:OfficeWorldDeliverCoffee-v0', 0, False, "test")


def create_config(
        seed=SEED,
        capture_video=False,
):
    env_config = {
        "num_agents": 2,
        "seed": seed,
    }

    config = PPOConfig()
    num_envs = env_config["num_agents"]
    config = (
        config.environment(
            "env",
            env_config=env_config, is_atari=False
        )
        .framework("torch")
        .multi_agent(
            policies={f"p{i}" for i in range(env_config["num_agents"])},
            policy_mapping_fn=lambda aid, *a, **kw: f"p{aid}",
        )
        .training(
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.1,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            num_sgd_iter=10,
            # Linearly adjust learning rate based on number of GPUs.
            lr=0.00015,
            grad_clip=100.0,
            grad_clip_by="global_norm",
            # train_batch_size=18, # new
        )
        .env_runners(
            num_envs_per_env_runner=1,
            # rollout_fragment_length=20, # new
        )
        .rl_module(
            rl_module_spec=MultiAgentRLModuleSpec(
                module_specs={
                    "p0": SingleAgentRLModuleSpec(catalog_class=PPORMLearningCatalog),
                    "p1": SingleAgentRLModuleSpec(catalog_class=PPORMLearningCatalog),
                }
            )
        )
        # .resources(num_cpus_per_worker=0.5)
        .evaluation(
            evaluation_interval=5, # 10,
            evaluation_duration=1,  # 5
            evaluation_duration_unit="episodes",
            evaluation_config=PPOConfig.overrides(
                entropy_coeff=0.0,
                explore=False,
                env_config={
                    "seed": seed,
                    "capture_video": True,
                },
            ),
        )
        .debugging(seed=seed, log_level="WARN")
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )

    )
    return config


def hyperparams_opt(
        num_iterations=400,
        seed=SEED,
        points_to_evaluate=None,
        num_samples=2, # Number of times to sample from the hyperparamter space
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

    hyperparam_mutations = {k: tune.uniform(*v) for k, v in hyperparam_bounds.items()}

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

    return dict(tune_config=tune.TuneConfig(metric=f"{ENV_RUNNER_RESULTS}/episode_return_mean", mode="max",
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
                                                seed=SEED,
                                                points_to_evaluate=points_to_evaluate,
                                            ),
                                            )
                )


@dataclass
class Args:
    experiment: str = ""

    hyperparam: bool = True
    resume: bool = False

    rm: bool = False
    rm_learning: bool = False
    rm_learning_freq: int = 1
    shared_layers: tuple = (0, 1)
    shared_policy: bool = False
    reward_shaping: bool = False

    seed: int = SEED
    num_iterations: int = 10 # 400
    capture_video: bool = False
    debug: bool = False
    max_concurrent_trials: int | None = None

    def __post_init__(self):
        if not self.hyperparam:
            assert self.experiment, "You must select an experiment to load"
            assert not self.resume, "You cannot resume"
        # elif self.experiment:
        #     assert self.resume, "Why do you have an experiment that you don't want to resume?"

        assert not self.shared_layers or (
                self.shared_layers and not self.shared_policy
        ), "cannot have shared layers and shared policy simultaneously"

        # if self.rm_learning:
        #     self.max_concurrent_trials = 1


if __name__ == "__main__":
    args = tyro.cli(Args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ray.init(
        num_cpus=None,
        runtime_env={
            "env_vars": ({"RAY_DEBUG": f"{str(int(args.debug))}"} if args.debug else {}),
        }
    )

    config = create_config(
        seed=args.seed,
        capture_video=args.capture_video,
    )

    stop = tune.stopper.CombinedStopper(
        tune.stopper.MaximumIterationStopper(max_iter=args.num_iterations),
        # tune.stopper.TrialPlateauStopper(
        #     metric=f"{ENV_RUNNER_RESULTS}/episode_return_mean",
        #     mode="max",
        #     std=0.01,
        #     grace_period=25,
        #     num_results=15
        # ),
    )
    tuner = tune.Tuner(
        config.algo_class,
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            storage_path="~/ray_results/debug" if args.debug else None,
            stop=stop,
            verbose=1,
            checkpoint_config=train.CheckpointConfig(
                checkpoint_frequency=100,
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
    # best_result = results.get_best_result(
    #     metric="reward_threshold_num_iterations", mode="min", scope="last"
    # )
    # print("CONFIG:", best_result.config)
