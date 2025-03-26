"""
Given a prior, find a posterior 
What sensor error should specify in order to have x% chance of a true detection
"""
import gymnasium as gym
import numpy as np

from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldCoffeeLabelingFunctionWrapper, WaterWorldRedLabelingFunctionWrapper


mini = 0
maxi = 1
target = 0.75

# env = gym.make("gym_subgoal_automata:WaterWorldRedGreen-v0",
env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"generation": "random", "environment_seed": 127, "hide_state_variables": True},
               render_mode="human",
               )
while mini < maxi:
    mid = (mini + maxi) / 2
    x = OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_false_confidence=mid, sensor_true_confidence=mid)
    # x = WaterWorldRedLabelingFunctionWrapper(env, sensor_false_confidence=mid, sensor_true_confidence=mid)
    conf = x.get_label_confidence(label_true_pred=True, value_true_prior=x.value_true_prior)
    if np.isclose(conf, target):
        print(f"Solution is {mid}")
        break
    elif conf > target:
        maxi = mid
    else:
        mini = mid

print(f"Solution is {mid}")
