"""
Quick and dirty script for calculating sensor confidence from posterior value
"""

import gym
import numpy as np

from rm_marl.envs.gym_subgoal_automata_wrapper import GymSubgoalAutomataAdapter, \
    OfficeWorldCoffeeLabelingFunctionWrapper, OfficeWorldALabelingFunctionWrapper

env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"generation": "random", "environment_seed": 0, "hide_state_variables": True})
env = GymSubgoalAutomataAdapter(env, render_mode="human", agent_id="A1")  # type: ignore

coffee_l = OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)

info = {"observations": {"f"}}

target_value = 0.5

for target_value in [0.5, 0.8, 0.9, 1]:
    lo = 0.5
    hi = 1
    while lo < hi:
        mid = (lo + hi) / 2
        coffee_l = OfficeWorldALabelingFunctionWrapper(env, sensor_true_confidence=mid, sensor_false_confidence=mid)
        info = {"observations": {"a"}}
        out = coffee_l.get_labels(info)
        if np.isclose(out['a'], target_value):
            print(mid)
            # exit
            lo = hi
        elif out['a'] > target_value:
            hi = mid
        else:
            lo = mid

    print(coffee_l.get_labels(info))
