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
from ray import tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler

from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.new_stack.algos.algo import PPORMConfig, PPORMLearningConfig
from rm_marl.new_stack.callbacks.callback_composer import CallbackComposer
from rm_marl.new_stack.callbacks.env_render_callback import EnvRenderCallback
from rm_marl.new_stack.callbacks.log_original_reward import LogOriginalReward
from rm_marl.new_stack.callbacks.store_config import StoreTracesCallback
from rm_marl.new_stack.env.multi_env_with_rm import make_multi_agent_with_rm
from rm_marl.new_stack.utils.env import env_creator
from rm_marl.new_stack.utils.run import custom_run_rllib_example_script_experiment

parser = add_rllib_example_script_args()
parser.set_defaults(env='gym_subgoal_automata:OfficeWorldDeliverCoffee-v0')

parser.add_argument('--custom-num-agents', type=int, default=1,
                    help='Number of agents in our script')
parser.add_argument('--use-perfect-rm', action="store_true",
                    help="Whether to use perfect RM, as opposed to no RM at all")


# Register our environment with tune.


def create_config(
        learn_rm=True
):
    # Slows down process; add back when debugging
    # callbacks = [EnvRenderCallback]
    callbacks = [LogOriginalReward]
    if learn_rm:
        config = PPORMLearningConfig()
        actor_name = "rm_learner_actor"
        config.actor_name(actor_name)
        callbacks.append(lambda: StoreTracesCallback(actor_name))
    else:
        config = PPORMConfig()

    config = (
        config.environment(
            env="env",
            env_config=env_config, is_atari=False,
        )
        .framework("torch")
        .training(
            lr=tune.loguniform(1e-5, 1e-3),  # Learning rate on a log scale
            gamma=0.99,
        )
        .training(
            mini_batch_size_per_learner=tune.choice([4, 8, 16, 32, 64]),
            clip_param=tune.choice([0.1, 0.2, 0.3]),
            vf_clip_param=tune.uniform(5.0, 30.0),  # Value function clipping
            kl_target=tune.loguniform(0.003, 0.3),
            kl_coeff=tune.uniform(0.3, 1),
            num_sgd_iter=tune.randint(3, 30),  # Number of epochs to execute per training batch
            lambda_=tune.uniform(0.9, 1),
            vf_loss_coeff=tune.uniform(0.5, 1),
            entropy_coeff=tune.uniform(0.0, 0.1),
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
            remote_worker_envs=True,
            # env_to_module_connector=NewRMStateConnector,
        )
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
        .callbacks(
            lambda: CallbackComposer(callbacks)
        )
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        # Switch off RLlib's logging to avoid having the large videos show up in any log
        # files.
        .debugging(seed=env_config["seed"], log_level="WARN", logger_config={})  # {"type": tune.logger.NoopLogger})
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
        f"p{i}": RLModuleSpec()
        for i in range(env_config["num_agents"])
    }

    layer_sizes = [8, 16, 32]
    num_layers = [1, 2, 4]
    hidden_architectures = [[layer_size] * n_layers for n_layers in num_layers for layer_size in layer_sizes]

    config.rl_module(
        rl_module_spec=MultiRLModuleSpec(module_specs=module_specs),
        # rl_module_spec=spec,
        # IMPORTANT: the model config dict needs to be defined here; it gets ignored if defined for individual policies.
        #   Noticed when resetting workers
        model_config_dict={
            # "fcnet_hiddens": tune.choice([[16, 16, 16], [32, 32]]),
            "fcnet_hiddens": tune.choice(hidden_architectures),
            "fcnet_activation": tune.choice(["relu", "tanh", "elu"]),
        }
    )

    return config


env_name = 'gym_subgoal_automata:OfficeWorldDeliverCoffee-v0'
dummy_env = NewGymSubgoalAutomataAdapter(gym.make(env_name))  # type: ignore

if __name__ == "__main__":
    args = parser.parse_args()

    env_config = {
        "num_agents": args.custom_num_agents,
        "seed": 123,
    }

    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

    register_env("env", make_multi_agent_with_rm(env_creator(env_name)))
    # register_env("env", env_creator(env_name))

    rm = dummy_env.get_perfect_rm()
    learn_rm = not args.use_perfect_rm
    base_config = create_config(learn_rm=learn_rm)

    stop = {
        TRAINING_ITERATION: args.stop_iters,
    }

    scheduler = ASHAScheduler(metric="env_runners/episode_return_mean", mode="max",
                              grace_period=min(15, args.stop_iters),
                              max_t=args.stop_iters)

    custom_run_rllib_example_script_experiment(base_config, args, stop=stop, scheduler=scheduler)
