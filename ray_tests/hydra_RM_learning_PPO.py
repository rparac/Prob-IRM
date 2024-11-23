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

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler

from rm_marl.new_stack.algos.algo import PPORMConfig, PPORMLearningConfig
from rm_marl.new_stack.callbacks.callback_composer import CallbackComposer
from rm_marl.new_stack.callbacks.env_render_callback import EnvRenderCallback
from rm_marl.new_stack.callbacks.heatmap_callback import HeatmapCallback
from rm_marl.new_stack.callbacks.log_original_reward import LogOriginalReward
from rm_marl.new_stack.callbacks.log_rm_learning import LogRMLearning
from rm_marl.new_stack.callbacks.store_config import StoreTracesCallback
from rm_marl.new_stack.env.multi_env_with_rm import make_multi_agent_with_rm
from rm_marl.new_stack.utils.env import hydra_env_creator
from rm_marl.new_stack.utils.hydra import from_hydra_config
from rm_marl.new_stack.utils.run import simplified_custom_run_rllib_example_script_experiment


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


def create_config(
        run_config=None,
        ppo_config=None,
        algo_config=None,
        model_config=None,
        rm_learner_config=None,
):
    callbacks = [LogOriginalReward]
    use_wandb = run_config["wandb"]["key"] is not None
    if use_wandb:
        callbacks.append(EnvRenderCallback)
        callbacks.append(HeatmapCallback)

    if run_config["recurrent"]:
        config = PPOConfig()
    elif run_config["use_perfect_rm"]:
        config = PPORMConfig()
    else:
        config = PPORMLearningConfig()
        callbacks.append(StoreTracesCallback)
        callbacks.append(LogRMLearning)
        config.rm_learner(
            **rm_learner_config,
        )

    config = (
        config.environment(
            env="env",
            env_config={
                "num_agents": run_config["num_agents"],
            },
        )
        .training(
            **from_hydra_config(algo_config),
        )
        .training(
            **from_hydra_config(ppo_config),
        )
        .env_runners(
            batch_mode="complete_episodes",
            # num_env_runners=0, # forces everything to be done on the local worker
            # num_env_runners= run_config["num_agents"],
            num_env_runners=run_config["num_env_runners"],
            num_envs_per_env_runner=1,
            # By default, environments are stepped one at a time
            # https://docs.ray.io/en/latest/rllib/rllib-env.html
            # https://docs.ray.io/en/latest/rllib/package_ref/env.html
            # Doesn't make a lot of difference; seems like it's not used tbf
            # remote_worker_envs=False,
        )
        .evaluation(
            evaluation_interval=run_config["render_freq"],
            evaluation_duration=1,  # 5
            # Important: Otherwise the evaluation runs in the main thread, which ruins environment ids
            # evaluation_num_env_runners=run_config["num_agents"],
            evaluation_num_env_runners=0,
            evaluation_duration_unit="episodes",
            evaluation_config=PPORMConfig.overrides(
                # entropy_coeff=0.0,
                # explore=False,
                # env_config={
                #     "seed": run_config["seed"],
                # },
            ),
        )
        .callbacks(
            partial(CallbackComposer, callbacks, stop_iters=run_config["stop_iters"], use_wandb=use_wandb),
        )
        .debugging(seed=run_config["seed"], log_level="WARN")
    )

    def policy_mapping_fn_(aid, episode, **kwargs):
        return f"p{aid}"

    policies = {
        f"p{i}"
        for i in range(run_config["num_agents"])
    }
    config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn_,
        # Check if policy_states_are_swappable should be true
    )

    module_specs = {
        f"p{i}": RLModuleSpec()
        for i in range(run_config["num_agents"])
    }

    model_config["fcnet_hiddens"]["options"] = [[layer_size] * n_layers for n_layers in model_config["_num_layers"] for
                                                layer_size in model_config["_layer_sizes"]]
    model_config["post_fcnet_hiddens"]["options"] = [[layer_size] * n_layers for n_layers in model_config["_num_layers"] for
                                                layer_size in model_config["_layer_sizes"]]

    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(module_specs=module_specs),
        # IMPORTANT: the model config dict needs to be defined here; it gets ignored if defined for individual policies.
        #   Noticed when resetting workers
        model_config_dict=from_hydra_config(model_config)
    )

    return config


@hydra.main(version_base=None, config_path="../even_newer_conf", config_name="config")
def run(cfg: DictConfig) -> int:
    _manual_value_override(cfg)

    run_config = cfg['run']
    print(run_config)

    np.random.seed(run_config["seed"])
    random.seed(run_config["seed"])

    env_config = cfg["env"]
    label_factories = [instantiate(label_factory_conf) for label_factory_conf in env_config["core_label_factories"]]

    if not env_config["use_restricted_observables"]:
        noisy_label_factories = [instantiate(label_factory_conf) for label_factory_conf in
                                 env_config["noise_label_factories"]]
        label_factories.extend(noisy_label_factories)
    print(env_config)
    env_config = OmegaConf.to_container(env_config, resolve=True)
    env_config["label_factories"] = label_factories
    # TODO: set to 6 for now; change to run_config["seed"] later
    env_config["seed"] = run_config["seed"]
    # TODO: check if we can move this directly
    env_config["use_rs"] = run_config["use_rs"]
    register_env("env", make_multi_agent_with_rm(hydra_env_creator(env_config)))
    # register_env("env", hydra_env_creator(env_config))

    # We can only render on wandb; turn on rendering if the key exists
    ppo_config = cfg["ppo"]
    algo_config = cfg["algo"]
    model_config = cfg["model"]
    rm_learner_config = OmegaConf.to_container(cfg["rm_learner"], resolve=True)
    if rm_learner_config["base_dir"] is None:
        rm_learner_config["base_dir"] = run_config["name"]

    base_config = create_config(run_config, ppo_config, algo_config, model_config, rm_learner_config)

    stop = {
        TRAINING_ITERATION: run_config["stop_iters"],
    }

    scheuduler_conf = run_config["tune_config"]["scheduler"]
    scheduler = ASHAScheduler(metric=scheuduler_conf["metric"], mode=scheuduler_conf["mode"],
                              grace_period=min(scheuduler_conf["min_grace_period"], run_config["stop_iters"]),
                              max_t=run_config["stop_iters"])

    simplified_custom_run_rllib_example_script_experiment(base_config, run_config, stop=stop, scheduler=scheduler)

    return 0


if __name__ == "__main__":
    run()
