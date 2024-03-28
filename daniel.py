import gym

from rm_marl.envs.gym_subgoal_automata_wrapper import GymSubgoalAutomataAdapter, \
    OfficeWorldBLabelingFunctionWrapper, OfficeWorldCLabelingFunctionWrapper, OfficeWorldDLabelingFunctionWrapper, \
    OfficeWorldALabelingFunctionWrapper
from rm_marl.envs.wrappers import NoisyLabelingFunctionComposer

env = gym.make("gym_subgoal_automata:OfficeWorldPatrolABCD-v0",
               params={"generation": "random", "environment_seed": 0, "hide_state_variables": True})
env = GymSubgoalAutomataAdapter(env, render_mode="human", agent_id="A1")  # type: ignore
# env = OfficeWorldDeliverCoffeeLabelingFunctionWrapper(env)
# office_l = OfficeWorldOfficeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
# plant_l = OfficeWorldPlantLabelingFunctionWrapper(env, sensor_true_confidence=0.9, sensor_false_confidence=0.9)
# coffee_l = OfficeWorldCoffeeLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
# mail_l = OfficeWorldMailLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=1)
# env = NoisyLabelingFunctionComposer([office_l, plant_l, coffee_l, mail_l])
a_l = OfficeWorldALabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=0.99)
b_l = OfficeWorldBLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=0.99)
c_l = OfficeWorldCLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=0.99)
d_l = OfficeWorldDLabelingFunctionWrapper(env, sensor_true_confidence=1, sensor_false_confidence=0.99)
env = NoisyLabelingFunctionComposer([a_l, b_l, c_l, d_l])
# env = NoisyLabelingFunctionComposer([office_l, plant_l, coffee_l, mail_l])

key_to_act = {
    "w": 0,
    "s": 1,
    "a": 2,
    "d": 3,
}

env.reset()
while True:
    x = input()

    action = {"A1": key_to_act[x]}
    obs, reward, terminated, truncated, info = env.step(action)
    print(env.get_labels(obs, prev_obs=None))
    if terminated:
        print(reward)
