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
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module import RLModuleSpec, MultiRLModuleSpec
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import register_env
from ray.tune.schedulers import ASHAScheduler

from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.new_stack.callbacks.callback_composer import CallbackComposer
from rm_marl.new_stack.callbacks.env_render_callback import EnvRenderCallback
from rm_marl.new_stack.env.multi_env_with_rm import make_multi_agent_with_rm
from rm_marl.new_stack.utils.env import env_creator, NO_RM
from rm_marl.new_stack.utils.run import custom_run_rllib_example_script_experiment

parser = add_rllib_example_script_args()
parser.set_defaults(env='gym_subgoal_automata:OfficeWorldDeliverCoffee-v0')

parser.add_argument('--custom-num-agents', type=int, default=1,
                    help='number of agents in our script')
parser.add_argument('--use-perfect-rm', action="store_true",
                    help="whether to use perfect rm, as opposed to no rm at all")


# Register our environment with tune.


def create_config():
    # callbacks = [EnvRenderCallback]
    callbacks = []

    # config = PPORMLearningConfig()
    config = PPOConfig()
    # config.resources(
    #     cpu
    # )
    config = (
        config.environment(
            env="env",
            env_config=env_config, is_atari=False,
        )
        .framework("torch")
        .training(
            lr=tune.loguniform(1e-5, 1e-3),  # learning rate on a log scale
            gamma=0.99,
        )
        # entropy_coeff = 0.0340, kl_coeff = 0.5914, kl_target = 0.0069, lambda
        #     =0.9319, lr=0.0010, mini_batch_s_2024-0
        # 9 - 25_14 - 03 - 17
        .training(
            mini_batch_size_per_learner=tune.choice([300, 400, 500]),  # 8, # tune.choice([4, 8, 16, 32, 64]),
            clip_param=tune.choice([0.1, 0.2, 0.3]),
            vf_clip_param=tune.uniform(5.0, 30.0),  # value function clipping
            kl_target=tune.loguniform(0.003, 0.3),
            kl_coeff=tune.uniform(0.3, 1),
            num_sgd_iter=tune.randint(3, 30),  # number of epochs to execute per training batch
            lambda_=tune.uniform(0.9, 1),
            vf_loss_coeff=tune.uniform(0.5, 1),
            entropy_coeff=0.1,  # tune.uniform(0.0, 0.1),
            grad_clip=100.0,
            grad_clip_by="global_norm",
        )
        .env_runners(
            batch_mode="complete_episodes",
            # This should be on if environment rendering is expensive; e.g. video game (might be needed for
            #  three pillars)
            # num_gpus_per_env_runner=(1/env_config["num_agents"] * getattr(args, "num_samples", 1)),
            # num_env_runners=0, # forces everything to be done on the local worker
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
            evaluation_config=PPOConfig.overrides(
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
            "use_lstm": True,
            # "fcnet_hiddens": tune.choice([[16, 16, 16], [32, 32]]),
            "fcnet_hiddens": tune.choice(hidden_architectures),
            # encoder lstm cell size
            "lstm_cell_size": tune.choice([8, 16, 32]),
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
        "rm": NO_RM,
    }

    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

    register_env("env", make_multi_agent_with_rm(env_creator(env_name)))
    # register_env("env", env_creator(env_name))

    base_config = create_config()

    # rl_spec = base_config.get_rl_module_spec(
    #     spaces={DEFAULT_MODULE_ID: (Box(low=0, high=1, shape=(110,)), Discrete(4))})
    # raise RuntimeError(rl_spec.build())

    stop = {
        TRAINING_ITERATION: args.stop_iters,
    }

    scheduler = ASHAScheduler(metric="env_runners/episode_return_mean", mode="max", grace_period=15,
                              max_t=args.stop_iters)

    custom_run_rllib_example_script_experiment(base_config, args, stop=stop, scheduler=scheduler)
