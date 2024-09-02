"""
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
from ray.rllib.algorithms import PPOConfig
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.env import EnvContext
from ray.rllib.env.multi_agent_env import make_multi_agent
from ray.rllib.utils.test_utils import (
    add_rllib_example_script_args,
    run_rllib_example_script_experiment,
)
from ray.tune.registry import get_trainable_cls, register_env

from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldOfficeLabelingFunctionWrapper, \
    OfficeWorldPlantLabelingFunctionWrapper, OfficeWorldALabelingFunctionWrapper, OfficeWorldBLabelingFunctionWrapper, \
    OfficeWorldCLabelingFunctionWrapper, OfficeWorldDLabelingFunctionWrapper, OfficeWorldCoffeeLabelingFunctionWrapper
from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import DiscreteToBoxObservationWrapper, NoisyLabelingFunctionComposer
from rm_marl.new_stack.algos.algo import PPORMConfig
from rm_marl.new_stack.callbacks.env_render_callback import EnvRenderCallback
from rm_marl.new_stack.networks.model import PPORMLearningCatalog

parser = add_rllib_example_script_args()
parser.set_defaults(env='gym_subgoal_automata:OfficeWorldDeliverCoffee-v0')


# Register our environment with tune.

def _env_creator(env_id, num_agents):
    def thunk(_env_ctx: EnvContext):
        curr_id = _env_ctx.worker_index - 1

        # env = gym.make("CartPole-v1")
        env = gym.make(env_id, render_mode="rgb_array",
                       params={"generation": "random", "environment_seed": 5 + curr_id,
                               "hide_state_variables": True})
        env = NewGymSubgoalAutomataAdapter(env, max_episode_length=250, env_idx=curr_id,
                                           num_agents=num_agents)  # type: ignore
        # raise RuntimeError(env.observation_space.shape)

        labeling_funs = [
            OfficeWorldOfficeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            OfficeWorldPlantLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            # OfficeWorldALabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            # OfficeWorldBLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            # OfficeWorldCLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            # OfficeWorldDLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
            OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
        ]

        env = NoisyLabelingFunctionComposer(labeling_funs)
        env = gym.wrappers.FlattenObservation(env)
        # raise RuntimeError(env.observation_space.shape)
        # env = DiscreteToBoxObservationWrapper(env)
        return env

    return thunk


def create_config(
):
    rm = dummy_env.get_perfect_rm()
    config = PPORMConfig(rm=rm)
    # config = PPOConfig()
    config = (
        config.environment(
            "env",
            env_config=env_config, is_atari=False
        )
        .framework("torch")
        # .multi_agent(
        #     policies={f"p{i}" for i in range(env_config["num_agents"])},
        #     policy_mapping_fn=lambda aid, *a, **kw: f"p{aid}",
        # )
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
            # num_env_runners=0, # forces everything to be done on the local worker
            # num_env_runners=23,  # env_config["num_agents"],
            num_env_runners=env_config["num_agents"],
            num_envs_per_env_runner=1,
        )
        .rl_module(
            rl_module_spec=SingleAgentRLModuleSpec(catalog_class=PPORMLearningCatalog),
        )
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
        .callbacks(EnvRenderCallback)
        # Switch off RLlib's logging to avoid having the large videos show up in any log
        # files.
        .debugging(seed=env_config["seed"], log_level="WARN", logger_config={"type": tune.logger.NoopLogger})
    )
    return config


env_name = 'gym_subgoal_automata:OfficeWorldDeliverCoffee-v0'
dummy_env = NewGymSubgoalAutomataAdapter(gym.make(env_name))  # type: ignore

if __name__ == "__main__":
    args = parser.parse_args()

    env_config = {
        "num_agents": 1,  # 10
        "seed": 123,
    }

    assert (
        args.enable_new_api_stack
    ), "Must set --enable-new-api-stack when running this script!"

    register_env("env", _env_creator(env_name,
                                     num_agents=env_config['num_agents']))  # make_multi_agent(_env_creator(env_name)))

    rm = dummy_env.get_perfect_rm()

    base_config = create_config()

    run_rllib_example_script_experiment(base_config, args)
