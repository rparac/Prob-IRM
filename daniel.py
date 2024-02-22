import gym

from rm_marl.envs.gym_subgoal_automata_wrapper import DanielGymAdapter, OfficeWorldDeliverCoffeeLabelingFunctionWrapper

env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"generation": "random", "environment_seed": 0, "hide_state_variables": True})
env = DanielGymAdapter(env, render_mode="human")  # type: ignore
env = OfficeWorldDeliverCoffeeLabelingFunctionWrapper(env)

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
    print(env.get_labels(obs))
    if terminated:
        print(reward)
