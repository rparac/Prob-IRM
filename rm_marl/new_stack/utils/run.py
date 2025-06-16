import argparse
import json
import logging
import os
import re
import time
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

import numpy as np
import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.utils.framework import try_import_jax, try_import_tf, try_import_torch
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
    EPISODE_RETURN_MEAN,
    EVALUATION_RESULTS,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
)
from ray.rllib.utils.test_utils import add_rllib_example_script_args
from ray.rllib.utils.typing import ResultDict
from ray.tune import CLIReporter
from ray.tune.logger import NoopLogger

from rm_marl.new_stack.tuner_callbacks.save_actor import SaveActorCallback

if TYPE_CHECKING:
    from ray.rllib.algorithms import Algorithm, AlgorithmConfig
from ray.rllib.algorithms import PPO

from rm_marl.new_stack.algos.algo import PPORM, PPORMLearning

jax, _ = try_import_jax()
tf1, tf, tfv = try_import_tf()
torch, _ = try_import_torch()

logger = logging.getLogger(__name__)


def continue_training(run_config):
    storage_dir = str(os.environ["RAY_RESULTS_DIR"])
    experiment_dir = f"{storage_dir}/{run_config['name']}"

    if not os.path.isdir(experiment_dir):
        return None

    previous_experiments = os.listdir(experiment_dir)    
    experiment_dir = f"{experiment_dir}/{max(previous_experiments)}"

    if run_config["no_rm"]:
        trainable = PPO
    elif run_config["use_perfect_rm"]:
        trainable = PPORM
    else:
        trainable = PPORMLearning

    start_time = time.time()
    tuner = tune.Tuner.restore(
        path=experiment_dir,
        trainable=trainable,
        resume_errored=True,
    )
    results = tuner.fit()
    time_taken = time.time() - start_time
    print(f"Time taken is {time_taken}")
    ray.shutdown()
    return results

def simplified_custom_run_rllib_example_script_experiment(
        base_config: "AlgorithmConfig",
        run_args=None,
        *,
        stop: Optional[Dict] = None,
        scheduler=None,
) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    """
    Args:
        base_config: The AlgorithmConfig object to use for this experiment. This base
            config will be automatically "extended" based on some of the provided
            `args`. For example, `args.num_env_runners` is used to set
            `config.num_env_runners`, etc..
        run_args: A hydra object containing the necessary arguments for running the program
        stop: An optional dict mapping ResultDict key strings (using "/" in case of
            nesting, e.g. "env_runners/episode_return_mean" for referring to
            `result_dict['env_runners']['episode_return_mean']` to minimum
            values, reaching of which will stop the experiment). Default is:
            {
            "env_runners/episode_return_mean": args.stop_reward,
            "training_iteration": args.stop_iters,
            "num_env_steps_sampled_lifetime": args.stop_timesteps,
            }

    Returns:
        The last ResultDict from a --no-tune run OR the tune.Tuner.fit()
        results.
    """
    # Initialize Ray.
    ray.init(
        # runtime_env={
        #     "env_vars": {"RAY_DEBUG": "1"},
        # }   
    )

    config = base_config

    config.framework("torch")
    config.api_stack(
        enable_rl_module_and_learner=True,
        enable_env_runner_and_connector_v2=True,
    )

    # Define compute resources used automatically (only using the --num-gpus arg).
    # New stack.
    # Define compute resources used.
    # TODO: maybe multiple learners would be better
    # config.resources(num_gpus=0)
    config.learners(
        num_learners=0,
        num_gpus_per_learner=(
            0
            if torch and torch.cuda.is_available()
            else 0
        ),
    )
    # config.resources(num_gpus=0)

    # Run the experiment w/o Tune (directly operate on the RLlib Algorithm object).
    # TODO: no_tune doesn't allow us to set the directory where we want the experiment to be stored. Uses ~/
    #   Also doesn't allow the use of run_config['name']
    if not run_args["should_tune"]:
        algo = config.build()
        for i in range(stop.get(TRAINING_ITERATION)):
            results = algo.train()
            if ENV_RUNNER_RESULTS in results:
                print(
                    f"iter={i} R={results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}",
                    end="",
                )
            if EVALUATION_RESULTS in results:
                Reval = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][
                    EPISODE_RETURN_MEAN
                ]
                print(f" R(eval)={Reval}", end="")
            print()
            for key, threshold in stop.items():
                val = results
                for k in key.split("/"):
                    try:
                        val = val[k]
                    except KeyError:
                        val = None
                        break
                if val is not None and not np.isnan(val) and val >= threshold:
                    print(f"Stop criterium ({key}={threshold}) fulfilled!")
                    ray.shutdown()
                    return results

        ray.shutdown()
        return results

    # Run the experiment using Ray Tune.

    # Turn off logger here; tuner turns the ones needed automatically
    config.debugging(logger_config={"type": NoopLogger})

    # Log results using WandB.
    tune_callbacks = []
    wandb_args = run_args["wandb"]
    if wandb_args["key"] is not None:
        tune_callbacks.append(
            WandbLoggerCallback(
                api_key=wandb_args["key"],
                project=wandb_args["project"],
                upload_checkpoints=True,
                **({"name": wandb_args["run_name"]}),
            )
        )
    tune_callbacks.append(
        SaveActorCallback()
    )

    # Auto-configure a CLIReporter (to log the results to the console).
    # Use better ProgressReporter for multi-agent cases: List individual policy rewards.
    progress_reporter = CLIReporter(
        metric_columns={
            **{
                TRAINING_ITERATION: "iter",
                "time_total_s": "total time (s)",
                NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "combined return",
            },
            **{
                (
                    f"{ENV_RUNNER_RESULTS}/module_episode_returns_mean/" f"{pid}"
                ): f"return {pid}"
                for pid in config.policies
            },
        },
    )

    # Force Tuner to use old progress output as the new one silently ignores our custom
    # `CLIReporter`.
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

    # Run the actual experiment (using Tune).
    start_time = time.time()
    storage_dir = str(os.environ["RAY_RESULTS_DIR"])
    # Set the storage path to understand which experiment is running
    storage_dir = f"{storage_dir}/{run_args['name']}"

    tune_args = run_args["tune_config"]

    results = tune.Tuner(
        config.algo_class,
        param_space=config,
        run_config=air.RunConfig(
            stop=stop,
            verbose=tune_args["verbose"],
            callbacks=tune_callbacks,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=tune_args["checkpoint_freq"],
                checkpoint_at_end=tune_args["checkpoint_at_end"],
                num_to_keep=2,
            ),
            progress_reporter=progress_reporter,
            storage_path=storage_dir,
        ),
        tune_config=tune.TuneConfig(
            num_samples=tune_args["num_samples"],
            scheduler=scheduler,
        ),
    ).fit()
    time_taken = time.time() - start_time
    print(f"Time taken is {time_taken}")

    ray.shutdown()

    return results


# TODO (sven): Make this the de-facto, well documented, and unified utility for most of
#  our tests:
#  - CI (label: "learning_tests")
#  - release tests (benchmarks)
#  - example scripts
def custom_run_rllib_example_script_experiment(
        base_config: "AlgorithmConfig",
        args: Optional[argparse.Namespace] = None,
        *,
        stop: Optional[Dict] = None,
        success_metric: Optional[Dict] = None,
        trainable: Optional[Type] = None,
        tune_callbacks: Optional[List] = None,
        keep_config: bool = False,
        scheduler=None,
        progress_reporter=None,
) -> Union[ResultDict, tune.result_grid.ResultGrid]:
    """Given an algorithm config and some command line args, runs an experiment.

    There are some constraints on what properties must be defined in `args`.
    It should ideally be generated via calling
    `args = add_rllib_example_script_args()`, which can be found in this very module
    here.

    The function sets up an Algorithm object from the given config (altered by the
    contents of `args`), then runs the Algorithm via Tune (or manually, if
    `args.no_tune` is set to True) using the stopping criteria in `stop`.

    At the end of the experiment, if `args.as_test` is True, checks, whether the
    Algorithm reached the `success_metric` (if None, use `env_runners/
    episode_return_mean` with a minimum value of `args.stop_reward`).

    See https://github.com/ray-project/ray/tree/master/rllib/examples for an overview
    of all supported command line options.

    Args:
        base_config: The AlgorithmConfig object to use for this experiment. This base
            config will be automatically "extended" based on some of the provided
            `args`. For example, `args.num_env_runners` is used to set
            `config.num_env_runners`, etc..
        args: A argparse.Namespace object, ideally returned by calling
            `args = add_rllib_example_script_args()`. It must have the following
            properties defined: `stop_iters`, `stop_reward`, `stop_timesteps`,
            `no_tune`, `verbose`, `checkpoint_freq`, `as_test`. Optionally, for WandB
            logging: `wandb_key`, `wandb_project`, `wandb_run_name`.
        stop: An optional dict mapping ResultDict key strings (using "/" in case of
            nesting, e.g. "env_runners/episode_return_mean" for referring to
            `result_dict['env_runners']['episode_return_mean']` to minimum
            values, reaching of which will stop the experiment). Default is:
            {
            "env_runners/episode_return_mean": args.stop_reward,
            "training_iteration": args.stop_iters,
            "num_env_steps_sampled_lifetime": args.stop_timesteps,
            }
        success_metric: Only relevant if `args.as_test` is True.
            A dict mapping a single(!) ResultDict key string (using "/" in
            case of nesting, e.g. "env_runners/episode_return_mean" for referring
            to `result_dict['env_runners']['episode_return_mean']` to a single(!)
            minimum value to be reached in order for the experiment to count as
            successful. If `args.as_test` is True AND this `success_metric` is not
            reached with the bounds defined by `stop`, will raise an Exception.
        trainable: The Trainable sub-class to run in the tune.Tuner. If None (default),
            use the registered RLlib Algorithm class specified by args.algo.
        tune_callbacks: A list of Tune callbacks to configure with the tune.Tuner.
            In case `args.wandb_key` is provided, appends a WandB logger to this
            list.
        keep_config: Set this to True, if you don't want this utility to change the
            given `base_config` in any way and leave it as-is. This is helpful
            for those example scripts which demonstrate how to set config settings
            that are taken care of automatically in this function otherwise (e.g.
            `num_env_runners`).

    Returns:
        The last ResultDict from a --no-tune run OR the tune.Tuner.fit()
        results.
    """
    if args is None:
        parser = add_rllib_example_script_args()
        args = parser.parse_args()

    # If run --as-release-test, --as-test must also be set.
    if args.as_release_test:
        args.as_test = True

    # Initialize Ray.
    ray.init(num_cpus=args.num_cpus or None, local_mode=args.local_mode)

    # Define one or more stopping criteria.
    if stop is None:
        stop = {
            f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
            f"{NUM_ENV_STEPS_SAMPLED_LIFETIME}": args.stop_timesteps,
            TRAINING_ITERATION: args.stop_iters,
        }

    config = base_config

    # Enhance the `base_config`, based on provided `args`.
    if not keep_config:
        # Set the framework.
        config.framework(args.framework)

        # Add an env specifier (only if not already set in config)?
        if args.env is not None and config.env is None:
            config.environment(args.env)

        # Enable the new API stack?
        if args.enable_new_api_stack:
            config.api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )

        # Define EnvRunner/RolloutWorker scaling and behavior.
        if args.num_env_runners is not None:
            config.env_runners(num_env_runners=args.num_env_runners)

        # Define compute resources used automatically (only using the --num-gpus arg).
        # New stack.
        if config.enable_rl_module_and_learner:
            # Define compute resources used.
            config.resources(num_gpus=0)
            config.learners(
                num_learners=args.num_gpus,
                num_gpus_per_learner=(
                    1
                    if torch and torch.cuda.is_available() and args.num_gpus > 0
                    else 0
                ),
            )
            config.resources(num_gpus=0)
        # Old stack.
        else:
            config.resources(num_gpus=args.num_gpus)

        # Evaluation setup.
        if args.evaluation_interval > 0:
            config.evaluation(
                evaluation_num_env_runners=args.evaluation_num_env_runners,
                evaluation_interval=args.evaluation_interval,
                evaluation_duration=args.evaluation_duration,
                evaluation_duration_unit=args.evaluation_duration_unit,
                evaluation_parallel_to_training=args.evaluation_parallel_to_training,
            )

        # Set the log-level (if applicable).
        if args.log_level is not None:
            config.debugging(log_level=args.log_level)

        # Set the output dir (if applicable).
        if args.output is not None:
            config.offline_data(output=args.output)

    # Run the experiment w/o Tune (directly operate on the RLlib Algorithm object).
    # TODO: no_tune doesn't allow us to set the directory where we want the experiment to be stored. Uses ~/
    if args.no_tune:
        assert not args.as_test and not args.as_release_test
        algo = config.build()
        for i in range(stop.get(TRAINING_ITERATION, args.stop_iters)):
            results = algo.train()
            if ENV_RUNNER_RESULTS in results:
                print(
                    f"iter={i} R={results[ENV_RUNNER_RESULTS][EPISODE_RETURN_MEAN]}",
                    end="",
                )
            if EVALUATION_RESULTS in results:
                Reval = results[EVALUATION_RESULTS][ENV_RUNNER_RESULTS][
                    EPISODE_RETURN_MEAN
                ]
                print(f" R(eval)={Reval}", end="")
            print()
            for key, threshold in stop.items():
                val = results
                for k in key.split("/"):
                    try:
                        val = val[k]
                    except KeyError:
                        val = None
                        break
                if val is not None and not np.isnan(val) and val >= threshold:
                    print(f"Stop criterium ({key}={threshold}) fulfilled!")
                    ray.shutdown()
                    return results

        ray.shutdown()
        return results

    # Run the experiment using Ray Tune.

    # Log results using WandB.
    tune_callbacks = tune_callbacks or []
    if hasattr(args, "wandb_key") and args.wandb_key is not None:
        project = args.wandb_project or (
                args.algo.lower() + "-" + re.sub("\\W+", "-", str(config.env).lower())
        )
        tune_callbacks.append(
            WandbLoggerCallback(
                api_key=args.wandb_key,
                project=project,
                upload_checkpoints=True,
                **({"name": args.wandb_run_name} if args.wandb_run_name else {}),
            )
        )

    # Auto-configure a CLIReporter (to log the results to the console).
    # Use better ProgressReporter for multi-agent cases: List individual policy rewards.
    if progress_reporter is None and args.num_agents > 0:
        progress_reporter = CLIReporter(
            metric_columns={
                **{
                    TRAINING_ITERATION: "iter",
                    "time_total_s": "total time (s)",
                    NUM_ENV_STEPS_SAMPLED_LIFETIME: "ts",
                    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": "combined return",
                },
                **{
                    (
                        f"{ENV_RUNNER_RESULTS}/module_episode_returns_mean/" f"{pid}"
                    ): f"return {pid}"
                    for pid in config.policies
                },
            },
        )

    # Force Tuner to use old progress output as the new one silently ignores our custom
    # `CLIReporter`.
    os.environ["RAY_AIR_NEW_OUTPUT"] = "0"

    # Run the actual experiment (using Tune).
    start_time = time.time()
    storage_dir = str(os.environ["RAY_RESULTS_DIR"])
    results = tune.Tuner(
        trainable or config.algo_class,
        param_space=config,
        run_config=air.RunConfig(
            stop=stop,
            verbose=args.verbose,
            callbacks=tune_callbacks,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=args.checkpoint_freq,
                checkpoint_at_end=args.checkpoint_at_end,
            ),
            progress_reporter=progress_reporter,
            storage_path=storage_dir,
        ),
        tune_config=tune.TuneConfig(
            num_samples=args.num_samples,
            scheduler=scheduler,
        ),
    ).fit()
    time_taken = time.time() - start_time

    ray.shutdown()

    # If run as a test, check whether we reached the specified success criteria.
    test_passed = False
    if args.as_test:
        # Success metric not provided, try extracting it from `stop`.
        if success_metric is None:
            for try_it in [
                f"{EVALUATION_RESULTS}/{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
                f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}",
            ]:
                if try_it in stop:
                    success_metric = {try_it: stop[try_it]}
                    break
            if success_metric is None:
                success_metric = {
                    f"{ENV_RUNNER_RESULTS}/{EPISODE_RETURN_MEAN}": args.stop_reward,
                }
        # TODO (sven): Make this work for more than one metric (AND-logic?).
        # Get maximum value of `metric` over all trials
        # (check if at least one trial achieved some learning, not just the final one).
        success_metric_key, success_metric_value = next(iter(success_metric.items()))
        best_value = max(
            row[success_metric_key] for _, row in results.get_dataframe().iterrows()
        )
        if best_value >= success_metric_value:
            test_passed = True
            print(f"`{success_metric_key}` of {success_metric_value} reached! ok")

        if args.as_release_test:
            trial = results._experiment_analysis.trials[0]
            stats = trial.last_result
            stats.pop("config", None)
            json_summary = {
                "time_taken": float(time_taken),
                "trial_states": [trial.status],
                "last_update": float(time.time()),
                "stats": stats,
                "passed": [test_passed],
                "not_passed": [not test_passed],
                "failures": {str(trial): 1} if not test_passed else {},
            }
            with open(
                    os.environ.get("TEST_OUTPUT_JSON", "/tmp/learning_test.json"),
                    "wt",
            ) as f:
                try:
                    json.dump(json_summary, f)
                # Something went wrong writing json. Try again w/ simplified stats.
                except Exception:
                    from ray.rllib.algorithms.algorithm import Algorithm

                    simplified_stats = {
                        k: stats[k] for k in Algorithm._progress_metrics if k in stats
                    }
                    json_summary["stats"] = simplified_stats
                    json.dump(json_summary, f)

        if not test_passed:
            raise ValueError(
                f"`{success_metric_key}` of {success_metric_value} not reached!"
            )

    return results
