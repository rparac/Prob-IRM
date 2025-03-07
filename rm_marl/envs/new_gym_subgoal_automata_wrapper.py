"""
Wrapper around https://github.com/ertsiger/gym-subgoal-automata/tree/master
which provides environments such as coffee world and water world.

Sample usage:
# See the link above for other options
env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"hide_state_variables": true, ...})
env = DanielGymAdapter(env)

Same as old; just doesn't require agent id
"""
import abc
import random
from typing import Optional, Union

import gymnasium as gym
import gymnasium.spaces
import numpy as np

from gym_subgoal_automata.envs.base.base_env import BaseEnv
from rm_marl.envs.wrappers import LabelingFunctionWrapper
from rm_marl.reward_machine import RewardMachine


class NewGymSubgoalAutomataAdapter(gym.Wrapper):
    def __init__(self, env: BaseEnv, max_episode_length=None,
                 use_restricted_observables: bool = True,
                 num_random_seeds: Optional[int] = None):

        # Explicitly returns observables as a part of the observation.
        # We regenerate them in this adapter using the info output.
        env.hide_state_variables = True
        super().__init__(env)

        env_name = env.spec.name
        self._avoid_magenta = "AvoidMagenta" in env_name

        self.env = env
        if use_restricted_observables:
            self.observables = self.env.unwrapped.get_restricted_observables()
        else:
            self.observables = self.env.unwrapped.get_observables()  # self.env.get_restricted_observables()
        self.max_episode_length = max_episode_length
        self.current_step = 0

        self.num_random_seeds = num_random_seeds

        # self.observation_space = gymnasium.spaces.Discrete(self.one_env_size * num_agents)
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        random.seed(0)


    def reset(self, seed=None, **kwargs):
        new_seed = seed
        if self.num_random_seeds is not None:
            new_seed = random.randint(0, self.num_random_seeds - 1)

        obs, info = self.env.reset(seed=new_seed, **kwargs)
        self.current_step = 0
        if self.render_mode == "human":
            self.env.render()

        info["is_positive_trace"] = False
        self.this_episode_infos = []
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_step += 1
        if self.render_mode == "human":
            self.env.render()

        # TODO: force negative reward when a plant is reached; this may need to be done through reward shaping or the other env
        if 'n' in info['observations']:
            reward = -1
        # force negative reward when a magenta is reached
        if self._avoid_magenta and 'm' in info['observations']:
            reward = -1

        if self.max_episode_length and self.current_step >= self.max_episode_length and not terminated:
            truncated = True

        info["is_positive_trace"] = reward > 0

        return obs, reward, terminated, truncated, info

    def render(self, **kwargs):
        if self.render_mode == "rgb_array":
            return self.env.render()

    # Converts subgoal automaton to Reward Machine
    def get_perfect_rm(self):
        subgoal_automaton = self.env.unwrapped.get_automaton()

        rm = RewardMachine()
        rm.add_states(subgoal_automaton.states)
        rm.set_u0(subgoal_automaton.initial_state)
        if subgoal_automaton.accept_state is not None:
            rm.set_uacc(subgoal_automaton.accept_state)
        if subgoal_automaton.reject_state is not None:
            rm.set_urej(subgoal_automaton.reject_state)

        for from_state, l_cond_to_state in subgoal_automaton.edges.items():
            for conditions, to_state in l_cond_to_state:
                for condition in conditions:
                    rm.add_transition(from_state, to_state, condition)
        return rm
