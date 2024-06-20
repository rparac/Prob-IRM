from ray import tune, train, air
from ray.rllib.algorithms import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS
from ray.tune import create_scheduler
from ray.util.client import ray

from rm_marl.trainer.ray.common import run_rllib_experiment, Args, hyperparams_opt
from rm_marl.trainer.ray.ppo_trainable import PPOTrainable


class NewTrainer:
    def __init__(self):
        pass

    def run(self, run_config: dict):
        ray.init(**run_config["ray_init_config"])

        # tune.run(
        #     "PPO",
        #     config={
        #         "env": "CartPole-v1",
        #         "evaluation_duration": 2,
        #         "evaluation_duration_unit": "episodes",
        #     }
        # )

        config = self._build_config(run_config["env_config"])

        config = (
            config
            .training(
                **run_config["training_config"],
            )
            # .api_stack(
            #     enable_rl_module_and_learner=False,
            # )
            # .rollouts(num_rollout_workers=1)
            .env_runners(
                # batch_mode="complete_episodes",
                use_worker_filter_stats=False,
                observation_filter="MeanStdFilter",
                num_env_runners=run_config["num_workers"],
            )
            .evaluation(
                **run_config["evaluation"],
                evaluation_config=PPOConfig.overrides(
                    entropy_coeff=0.0,
                    explore=False,
                ),
            )
            .debugging(seed=0)
        )

        stop = tune.stopper.CombinedStopper(
            tune.stopper.MaximumIterationStopper(max_iter=100),
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
                storage_path=None,
                stop=stop,
                verbose=1,
                checkpoint_config=train.CheckpointConfig(
                    checkpoint_frequency=1,
                    num_to_keep=1,
                    checkpoint_score_order="max",
                    checkpoint_score_attribute=f"{ENV_RUNNER_RESULTS}/episode_return_mean",
                    checkpoint_at_end=True,
                ),
                failure_config=train.FailureConfig(fail_fast=False),
                sync_config=train.SyncConfig(sync_artifacts=True),
            ),
            **hyperparams_opt(
                num_iterations=100,
                seed=0,
                max_concurrent_trials=2,
            ),
        )



        # tune.run(
        #     config.algo_class,
        #     config={
        #         "env": "CartPole-v1",
        #         "evaluation_duration": 2,
        #         "evaluation_duration_unit": "episodes",
        #     }
        # )

        run_rllib_experiment(config, Args())

        ray.shutdown()

    def _build_config(self, env_config):
        config = (
            PPOConfig()
            .environment(
                env="CartPole-v1",
                env_config=env_config,
                is_atari=False,
            )
            .framework("torch")
        )
        return config
