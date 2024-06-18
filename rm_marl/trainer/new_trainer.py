from ray import tune, train
from ray.rllib.algorithms import PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.tune import create_scheduler
from ray.util.client import ray

from rm_marl.trainer.ray.common import run_rllib_experiment, Args
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
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True
            )
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

        # tune.run(
        #     config.algo_class,
        #     config={
        #         "env": "CartPole-v1",
        #         "evaluation_duration": 2,
        #         "evaluation_duration_unit": "episodes",
        #     }
        # )

        num_iterations = 100
        scheduler_name = "asynchyperband"
        scheduler = create_scheduler(
            scheduler_name,
            time_attr="training_iteration",
            max_t=num_iterations,  # max time units per trial
            grace_period=int(num_iterations / 10 * 2),  # for early stopping
        )
        tuner = tune.Tuner(
            PPOTrainable,
            tune_config=tune.TuneConfig(
                metric="episode_return_mean",
                mode="max",
                scheduler=scheduler,
            ),
            param_space={
                "env": "CartPole-v1",
                "kl_coeff": 1.0,
                "model": {"free_log_std": True},
                # These params are tuned from a fixed starting value.
                "lambda": 0.95,
                "clip_param": 0.2,
                "lr": 1e-4,
                # These params start off randomly drawn from a set.
                "num_sgd_iter": tune.choice([10, 20, 30]),
                "sgd_minibatch_size": tune.choice([128, 512, 2048]),
                "train_batch_size": tune.choice([100, 200, 300]),  # tune.choice([10000, 20000, 40000]),
            },
            run_config=train.RunConfig(),
        )
        results = tuner.fit()

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
