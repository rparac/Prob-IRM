import gym_subgoal_automata
import gymnasium as gym

from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldCoffeeLabelingFunctionWrapper, \
    OfficeWorldOfficeLabelingFunctionWrapper, OfficeWorldPlantLabelingFunctionWrapper
from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import ProbabilisticRewardShaping, NoisyLabelingFunctionComposer

seed = 0


env = gym.make("OfficeWorldDeliverCoffee-v0",
               params={"generation": "random", "environment_seed": 129, "hide_state_variables": True},
               render_mode="human",
               )
env = NewGymSubgoalAutomataAdapter(env, max_episode_length=250)  # type: ignore

labeling_funs = [
    OfficeWorldOfficeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
    OfficeWorldPlantLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
    # OfficeWorldALabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
    # OfficeWorldBLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
    # OfficeWorldCLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
    # OfficeWorldDLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1),
    OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=1,
                                             sensor_false_confidence=1,)
]


rm = env.get_perfect_rm()
env = NoisyLabelingFunctionComposer(labeling_funs)
env = ProbabilisticRewardShaping(env, shaping_rm=rm, discount_factor=0.9999)

key_to_act = {
    "w": 0,
    "s": 1,
    "a": 2,
    "d": 3,
}

obs, info = env.reset()
env.render()
while True:
    x = input()

    action = key_to_act[x]
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(obs, reward, terminated, truncated, info)
    if terminated:
        print(reward)
