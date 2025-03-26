import gym_subgoal_automata
import gymnasium as gym

from rm_marl.envs.gym_subgoal_automata_wrapper import OfficeWorldCoffeeLabelingFunctionWrapper, \
    OfficeWorldOfficeLabelingFunctionWrapper, OfficeWorldPlantLabelingFunctionWrapper
from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import ProbabilisticRewardShaping, NoisyLabelingFunctionComposer

import random
import time
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

def _get_yx_pos(curr_state):
    width = 12  
    height = 9

    idx = np.argmax(curr_state[:width * height])

    y = idx % height
    x = idx // height
    return y, x

def _create_heatmap(yx_positions, i=0):
    width = 12  
    height = 9
    # h x w
    # heatmap = np.zeros((width * 10, height * 10), dtype=np.int32)
    heatmap = np.zeros((height, width), dtype=np.int32)
    for yx_pos in yx_positions:
        heatmap[height - yx_pos[0] - 1, yx_pos[1]] += 1

    # Create the actual heatmap image.
    # Normalize the heatmap to values between 0 and 1
    norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
    # Use a colormap (e.g., 'hot') to map normalized values to RGB
    colormap = plt.get_cmap("coolwarm")  # try "hot" and "viridis" as well?
    # Returns a (64, 64, 4) array (RGBA).
    heatmap_rgb = colormap(norm(heatmap))
    # Convert RGBA to RGB by dropping the alpha channel and converting to uint8.
    heatmap_rgb = (heatmap_rgb[:, :, :3] * 255).astype(np.uint8)
    # Log the image.
    image = Image.fromarray(heatmap_rgb)
    image.save(f"test{i}.png")
    print(heatmap_rgb)

seed = 0


env = gym.make("OfficeWorldPatrolABCD-v0",
               params={"generation": "random", "environment_seed": 9, "hide_state_variables": True},
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
    OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
]


rm = env.get_perfect_rm()
env = NoisyLabelingFunctionComposer(labeling_funs)
env = ProbabilisticRewardShaping(env, shaping_rm=rm, discount_factor=0.9999)
env = gym.wrappers.FlattenObservation(env)
# env = gym.experimental.wrappers.DtypeObservationV0(env, **{"dtype": np.float32})

key_to_act = {
    "w": 0,
    "s": 1,
    "a": 2,
    "d": 3,
}


for i in range(5):
    obs, info = env.reset()
    env.render()
    yx_positions = []
    terminated, truncated = False, False
    num_steps = 0
    while not terminated and not truncated:
        # action = random.randint(0, 3)
        # num_steps += 1
        x = input()
        action = key_to_act[x]
        yx_positions.append(_get_yx_pos(obs))
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        # print(obs, reward, terminated, truncated, info)
        if terminated or truncated:
            # print(reward)
            _create_heatmap(yx_positions, i)
            print(num_steps)
