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
"""

import gymnasium as gym
import numpy as np
from ray import tune
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec
from ray.rllib.env import EnvContext
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import register_env

from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldOfficeLabelingFunctionWrapper, \
    OfficeWorldPlantLabelingFunctionWrapper, OfficeWorldCoffeeLabelingFunctionWrapper
from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import NoisyLabelingFunctionComposer
from rm_marl.new_stack.algos.algo import PPORMConfig, PPORMLearning, PPORMLearningConfig
from rm_marl.new_stack.callbacks.callback_composer import CallbackComposer
from rm_marl.new_stack.callbacks.env_render_callback import EnvRenderCallback
from rm_marl.new_stack.callbacks.store_config import StoreTracesCallback
from rm_marl.new_stack.connectors.new_rm_state_connector import NewRMStateConnector
from rm_marl.new_stack.env.multi_env_with_rm import make_multi_agent_with_rm
from rm_marl.new_stack.env.rm_wrapper import RMWrapper
from rm_marl.new_stack.modules.net import NewCustomNet
from rm_marl.new_stack.utils.env import env_creator

parser = add_rllib_example_script_args()
parser.set_defaults(env='gym_subgoal_automata:OfficeWorldDeliverCoffee-v0')


# Register our environment with tune.



def create_config(
):
    config = PPORMLearningConfig()
    # config = PPOConfig()
    actor_name = "rm_learner_actor"
    config = (
        config.environment(
            "env",
            env_config=env_config, is_atari=False,
        )
        .framework("torch")
        .training(
            lambda_=0.95,
            kl_coeff=0.5,
            clip_param=0.1,
            vf_clip_param=10.0,
            entropy_coeff=0.01,
            num_sgd_iter=10,
            lr=0.00015,
            grad_clip=100.0,
            grad_clip_by="global_norm",
        )
        .env_runners(
            batch_mode="complete_episodes",
            # num_env_runners=0, # forces everything to be done on the local worker
            # num_env_runners=23,  # env_config["num_agents"],
            num_env_runners=env_config["num_agents"],
            num_envs_per_env_runner=1,
            # By default, environments are stepped one at a time
            # https://docs.ray.io/en/latest/rllib/rllib-env.html
            # https://docs.ray.io/en/latest/rllib/package_ref/env.html
            # remote_worker_envs=True,
            # env_to_module_connector=NewRMStateConnector,
        )
        # .rl_module(
        #     rl_module_spec=RLModuleSpec(
        #         module_class=NewCustomNet,
        #         # TODO: Use this
        #         model_config_dict={
        #             "hidden_layer_dims": [16, 16],
        #             "num_rm_states": 1,
        #         }
        #     ),
        # )
        # .resources(num_cpus_per_worker=0.5)
        .evaluation(
            evaluation_interval=5,  # 10,
            evaluation_duration=1,  # 5
            # Important: Otherwise the evaluation runs in the main thread, which ruins environment ids
            evaluation_num_env_runners=env_config["num_agents"],
            evaluation_duration_unit="episodes",
            evaluation_config=PPORMConfig.overrides(
                entropy_coeff=0.0,
                explore=False,
                env_config={
                    "seed": env_config["seed"],
                },
            ),
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .actor_name(actor_name=actor_name)
        # .callbacks(EnvRenderCallback)
        .callbacks(
            lambda: CallbackComposer([
                lambda: StoreTracesCallback(actor_name),
                EnvRenderCallback,
            ])
        )
        # .callbacks(lambda: StoreTracesCallback(actor_name))
        # Switch off RLlib's logging to avoid having the large videos show up in any log
        # files.
        .debugging(seed=env_config["seed"], log_level="WARN", logger_config={"type": tune.logger.NoopLogger})
    )

    def policy_mapping_fn_(aid, worker, **kwargs):
        return f"p{aid}"

    policies = {
        f"p{i}"
        for i in range(env_config["num_agents"])
    }
    config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn_,
    )

    module_specs = {
        f"p{i}": RLModuleSpec(
            module_class=NewCustomNet,
            # TODO: Use this
            model_config_dict={
                "hidden_layer_dims": [16, 16],
                "num_rm_states": 1,
            }
        )
        for i in range(env_config["num_agents"])
    }

    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(module_specs=module_specs),
    )

    return config


env_name = 'gym_subgoal_automata:OfficeWorldDeliverCoffee-v0'
dummy_env = NewGymSubgoalAutomataAdapter(gym.make(env_name))  # type: ignore

if __name__ == "__main__":
    args = parser.parse_args()

    env_config = {
        "num_agents": 2,  # 10
        "seed": 123,
    }

    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

    register_env("env", make_multi_agent_with_rm(env_creator(env_name)))
    # register_env("env", env_creator(env_name))

    rm = dummy_env.get_perfect_rm()
    base_config = create_config()

    run_rllib_example_script_experiment(base_config, args)
