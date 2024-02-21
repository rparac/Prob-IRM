import gym

from rm_marl.envs.gym_subgoal_automata.gym_subgoal_automata_wrapper import DanielGymAdapter

env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"generation": "random", "environment_seed": 0, "hide_state_variables": True})
env = DanielGymAdapter(env)

env.reset()
env.render()
env.step(1)
env.render()
