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

from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldOfficeLabelingFunctionWrapper, \
    OfficeWorldPlantLabelingFunctionWrapper, OfficeWorldCoffeeLabelingFunctionWrapper
from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import NoisyLabelingFunctionComposer
from rm_marl.new_stack.env.RMWrapper import RMWrapper


# Register our environment with tune.

# env = gym.make("CartPole-v1")
def main():
    env = gym.make('gym_subgoal_automata:OfficeWorldDeliverCoffee-v0', render_mode="rgb_array",
                   params={"generation": "random", "environment_seed": 5,
                           "hide_state_variables": True})
    env = NewGymSubgoalAutomataAdapter(env, max_episode_length=250, env_idx=0,
                                       num_agents=1)  # type: ignore
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
    env = RMWrapper(env)

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

    raise RuntimeError("done")

if __name__ == "__main__":
    main()