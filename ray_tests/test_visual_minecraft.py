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

from rm_marl.new_stack.env.visual_minecraft.env import GridWorldEnv
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
from rm_marl.new_stack.networks.recurrent import CustomLSTMModule
from rm_marl.new_stack.utils.env import hydra_env_creator, hydra_visual_minecraft_env_creator
from rm_marl.new_stack.utils.hydra import from_hydra_config, manual_value_override
from rm_marl.new_stack.utils.run import simplified_custom_run_rllib_example_script_experiment, continue_training


def create_config(
        run_config=None,
        ppo_config=None,
        algo_config=None,
        model_config=None,
):
    config = PPOConfig()
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

    model_config = OmegaConf.to_container(model_config, resolve=True)
    model_config["fcnet_hiddens"]["options"] = [[layer_size] * n_layers for n_layers in model_config["_num_layers"] for
                                                layer_size in model_config["_layer_sizes"]]
    model_config["post_fcnet_hiddens"]["options"] = [[layer_size] * n_layers for n_layers in model_config["_num_layers"] for
                                                layer_size in model_config["_layer_sizes"]]
    model_config = from_hydra_config(model_config)

    module_class = CustomLSTMModule if model_config.get("use_lstm", False) else None
    module_specs = {
        f"p{i}": RLModuleSpec(
            module_class=module_class,
        )
        for i in range(run_config["num_agents"])
    }

    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(rl_module_specs=module_specs),
        # IMPORTANT: the model config dict needs to be defined here; it gets ignored if defined for individual policies.
        #   Noticed when resetting workers
        # model_config_dict=from_hydra_config(model_config)
        model_config=model_config,
    )

    return config
    return config



@hydra.main(version_base=None, config_path="../conf", config_name="config")
def run(cfg: DictConfig) -> int:
    manual_value_override(cfg)


    run_config = cfg['run']
    print(run_config)

    np.random.seed(run_config["seed"])
    random.seed(run_config["seed"])

    env_config = setup_env_config()
    register_env("env", make_multi_agent_with_rm(hydra_visual_minecraft_env_creator(env_config)))

    ppo_config = cfg["ppo"]
    algo_config = cfg["algo"]
    model_config = cfg["model"]

    base_config = create_config(run_config, ppo_config, algo_config, model_config)

    stop = {
        TRAINING_ITERATION: run_config["stop_iters"],
    }

    scheuduler_conf = run_config["tune_config"]["scheduler"]
    scheduler = ASHAScheduler(metric=scheuduler_conf["metric"], mode=scheuduler_conf["mode"],
                              grace_period=min(scheuduler_conf["min_grace_period"], run_config["stop_iters"]),
                              max_t=run_config["stop_iters"])

    result = None
    if run_config["continue_training"]:
        result = continue_training(run_config)

    if result is None:
        simplified_custom_run_rllib_example_script_experiment(base_config, run_config, stop=stop, scheduler=scheduler)

    return 0


def setup_rm_learner_config(rm_learner_config, run_config):
    rm_learner_config = OmegaConf.to_container(rm_learner_config, resolve=True)
    if rm_learner_config["base_dir"] is None:
        rm_learner_config["base_dir"] = run_config["name"]
    if rm_learner_config["use_old_rm_learner"] is None:
        rm_learner_config["use_old_rm_learner"] = run_config["use_old_rm_learner"]
    return rm_learner_config

def setup_env_config():
    env_id="VisualMinecraft-v0"
    items = ["pickaxe", "lava", "door", "gem", "empty"]
    formula = "(F c0)", 5, "task0: visit({1})".format(*items)
    # # formula = "F(c0 & F(c1))", 5, "task3: seq_visit({0}, {1})".format(*items)

    return {
        "name": env_id,
        "render_mode": "rgb_array",
    }


if __name__ == "__main__":
    run()
