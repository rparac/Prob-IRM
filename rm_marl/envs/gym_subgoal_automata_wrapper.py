"""
Wrapper around https://github.com/ertsiger/gym-subgoal-automata/tree/master
which provides environments such as coffee world and water world.

Sample usage:
# See the link above for other options
env = gym.make("gym_subgoal_automata:OfficeWorldDeliverCoffee-v0",
               params={"hide_state_variables": true, ...})
env = DanielGymAdapter(env)
"""
import abc
import math
from typing import Union

import gymnasium as gym
import numpy as np

from gym_subgoal_automata.envs.base.base_env import BaseEnv

from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import LabelingFunctionWrapper
from rm_marl.reward_machine import RewardMachine


class GymSubgoalAutomataAdapter(gym.Wrapper):
    def __init__(self, env: BaseEnv, agent_id: Union[int, str], max_episode_length=None,
                 use_restricted_observables: bool = True):
        # Explicitly returns observables as a part of the observation.
        # We regenerate them in this adapter using the info output.
        env.hide_state_variables = True
        super().__init__(env)

        self.agent_id = agent_id
        self.env = env
        if use_restricted_observables:
            self.observables = self.env.get_restricted_observables()
        else:
            self.observables = self.env.get_observables()  # self.env.get_restricted_observables()
        self.max_episode_length = max_episode_length
        self.current_step = 0

        self.observation_space = gym.spaces.Dict({self.agent_id: env.observation_space})
        self.action_space = gym.spaces.Dict({self.agent_id: env.action_space})

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.current_step = 0
        if self.render_mode == "human":
            self.env.render()

        info["is_positive_trace"] = False
        return {self.agent_id: obs}, {}

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action[self.agent_id])
        self.current_step += 1
        if self.render_mode == "human":
            self.env.render()

        # TODO: force negative reward when a plant is reached; this may need to be done through reward shaping or the other env
        if 'n' in info['observations']:
            reward = -1

        if self.max_episode_length and self.current_step >= self.max_episode_length:
            truncated = True

        info["is_positive_trace"] = reward > 0
        return {self.agent_id: obs}, reward, terminated, truncated, info

    def render(self, **kwargs):
        if self.render_mode == "rgb_array":
            return self.env.render()

    # Converts subgoal automaton to Reward Machine
    def get_perfect_rm(self):
        subgoal_automaton = self.env.get_automaton()

        rm = RewardMachine()
        rm.add_states(subgoal_automaton.states)
        rm.set_u0(subgoal_automaton.initial_state)
        rm.set_uacc(subgoal_automaton.accept_state)
        rm.set_urej(subgoal_automaton.reject_state)

        for from_state, l_cond_to_state in subgoal_automaton.edges.items():
            for conditions, to_state in l_cond_to_state:
                for condition in conditions:
                    rm.add_transition(from_state, to_state, condition)
        return rm

    @staticmethod
    def _split_conditions(conditions: str):
        return conditions.split('&')


# class OfficeWorldDeliverCoffeeLabelingFunctionWrapper(LabelingFunctionWrapper):
#     """
#     Looked at https://github.com/ertsiger/gym-subgoal-automata/blob/1879a6512441cdf0758c937cc659931d49260d38/gym_subgoal_automata/envs/officeworld/officeworld_env.py#L9-L18
#     to find object ids
#     """
#
#     def __init__(self, env: GymSubgoalAutomataAdapter):
#         super().__init__(env)
#         self.env = env
#
#     def get_labels(self, info: dict = None):
#         """Returns a modified observation."""
#         labels = []
#
#         if info["observations"]["f"]:
#             labels.append('f')
#         if info["observations"]["g"]:
#             labels.append('g')
#         if info["observations"]["n"]:
#             labels.append('n')
#
#         return labels
#
#     def get_all_labels(self):
#         return [
#             'f',  # coffee
#             'g',  # office
#             'n',  # plant
#         ]


class OfficeWorldAbstractLabelingFunctionWrapper(LabelingFunctionWrapper, abc.ABC):
    def __init__(self, env: Union[GymSubgoalAutomataAdapter, NewGymSubgoalAutomataAdapter],
                 sensor_true_confidence: float,
                 sensor_false_confidence: float,
                 seed: int = 0, value_true_prior=2 / (12 * 9)):
        super().__init__(env, noisy=True, sensor_true_confidence=sensor_true_confidence,
                         sensor_false_confidence=sensor_false_confidence)
        self.env = env
        self.rng = np.random.default_rng(seed)
        self.num_steps = 0
        self.value_true_prior = value_true_prior

    def get_labels(self, info: dict):
        self.num_steps += 1
        # TODO: this may be slow; as we do it a number of times
        if self.get_label() in info.get("observations", set()):
            coffee_predicted = bool(self.rng.binomial(1, self.sensor_true_confidence))
        else:
            coffee_predicted = bool(1 - self.rng.binomial(1, self.sensor_false_confidence))
        labels = {self.get_label(): self.get_label_confidence(coffee_predicted, value_true_prior=self.value_true_prior)}
        return labels

    @abc.abstractmethod
    def get_label(self):
        raise RuntimeError("Not implemented")

    def get_all_labels(self):
        return [self.get_label()]


class OfficeWorldCoffeeLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def get_label(self):
        return 'f'  # coffee


class OfficeWorldOfficeLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def get_label(self):
        return 'g'  # office


class OfficeWorldPlantLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):

    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=1 / (12 * 9))

    def get_label(self):
        return 'n'  # plant


class OfficeWorldMailLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def get_label(self):
        return 'm'  # mail


class OfficeWorldALabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=1 / (12 * 9))

    def get_label(self):
        return "a"


class OfficeWorldBLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=1 / (12 * 9))

    def get_label(self):
        return "b"


class OfficeWorldCLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=1 / (12 * 9))

    def get_label(self):
        return "c"


class OfficeWorldDLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=1 / (12 * 9))

    def get_label(self):
        return "d"


# 2 balls * (area of circle - radius=15) / area of screen (dimx=400)
WATERWORLD_DEFAULT_PRIOR = 2 * ((15 * 15) * math.pi) / (400 * 400)

class WaterWorldRedLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "r"

class WaterWorldGreenLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "g"

class WaterWorldCyanLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "c"

class WaterWorldBlueLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "b"


class WaterWorldYellowLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "y"

class WaterWorldMagentaLabelingFunctionWrapper(OfficeWorldAbstractLabelingFunctionWrapper):
    def __init__(self, env: GymSubgoalAutomataAdapter, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(env, sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "m"
