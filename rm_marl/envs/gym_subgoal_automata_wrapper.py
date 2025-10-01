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
from typing import Set, Union

import gymnasium as gym
import numpy as np

from gym_subgoal_automata.envs.base.base_env import BaseEnv

from rm_marl.envs.new_gym_subgoal_automata_wrapper import NewGymSubgoalAutomataAdapter
from rm_marl.envs.wrappers import LabelExtractor, LabelingFunctionWrapper
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


# class OfficeWorldDeliverCoffeeLabelExtractor(LabelExtractor):
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


class OfficeWorldAbstractLabelExtractor(LabelExtractor, abc.ABC):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float,
                 seed: int = 0, value_true_prior=2 / (12 * 9)):
        super().__init__()
        self.sensor_true_confidence = sensor_true_confidence
        self.sensor_false_confidence = sensor_false_confidence

        self.rng = np.random.default_rng(seed)
        self.num_steps = 0
        self.value_true_prior = value_true_prior

    """
    We use the following binary sensor model:
        - It takes in two parameters:
            - p(true_value_predicated | true_value) = sensor_true_confidence
                - can compute p(false_value_predicted | true_value) = 1 - sensor_true_confidence
            - p(false_value_predicated | false_value) = sensor_false_confidence
                - can compute p(true_value_predicted | false_value) = 1 - sensor_false_confidence
        - Assumes the prior probabilities as 0.5, i.e p(true_value) = 0.5 (=sensor_true_prior), p(false_value) = 0.5
        - We can see a sensor prediction, but need to compute our belief that the value is true.
            - This can be done with Bayes rule
            -   p(true_value | true_value_predicted) =
                   p(true_value_predicated | true_value) * p(true_value) / 
                   p(true_value_predicated | true_value) * p(true_value) 
                      + p(true_value_predicted | false_value) * p(false_value)
           - Opposite case p(true_value | false_value_predicted)  is similar
    """

    def get_label_confidence(self, label_true_pred: bool, value_true_prior: float = 0.5):
        value_false_prior = 1 - value_true_prior
        # case: p(true_value | true_value_predicted)
        if label_true_pred:
            p_true_and_true_pred = self.sensor_true_confidence * value_true_prior
            p_true_pred = (self.sensor_true_confidence * value_true_prior +
                           (1 - self.sensor_false_confidence) * value_false_prior)
            return p_true_and_true_pred / p_true_pred
        # case p(true_value | false_value_predicted)
        else:
            p_true_and_false_pred = (1 - self.sensor_true_confidence) * value_true_prior
            p_false_pred = ((1 - self.sensor_true_confidence) * value_true_prior
                            + self.sensor_false_confidence * value_false_prior)
            return p_true_and_false_pred / p_false_pred


    def get_labels(self, info: dict):
        self.num_steps += 1
        # TODO: this may be slow; as we do it a number of times
        if self.get_label() in info.get("observations", set()):
            coffee_predicted = bool(self.rng.binomial(1, self.sensor_true_confidence))
        else:
            coffee_predicted = bool(1 - self.rng.binomial(1, self.sensor_false_confidence))
        labels = {self.get_label(): self.get_label_confidence(coffee_predicted, value_true_prior=self.value_true_prior)}
        return labels

    def get_labels_without_probability(self, info: dict) -> Set[str]:
        if self.get_label() in info.get("observations", set()):
            coffee_predicted = bool(self.rng.binomial(1, self.sensor_true_confidence))
        else:
            coffee_predicted = bool(1 - self.rng.binomial(1, self.sensor_false_confidence))
        
        if not coffee_predicted:
            return []
        return [self.get_label()]

    @abc.abstractmethod
    def get_label(self):
        raise RuntimeError("Not implemented")

    def get_all_labels(self):
        return [self.get_label()]


class OfficeWorldCoffeeLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def get_label(self):
        return 'f'  # coffee


class OfficeWorldOfficeLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def get_label(self):
        return 'g'  # office


class OfficeWorldPlantLabelExtractor(OfficeWorldAbstractLabelExtractor):

    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=1 / (12 * 9))

    def get_label(self):
        return 'n'  # plant


class OfficeWorldMailLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def get_label(self):
        return 'm'  # mail


class OfficeWorldALabelExtractor(OfficeWorldAbstractLabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=1 / (12 * 9))

    def get_label(self):
        return "a"


class OfficeWorldBLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=1 / (12 * 9))

    def get_label(self):
        return "b"


class OfficeWorldCLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=1 / (12 * 9))

    def get_label(self):
        return "c"


class OfficeWorldDLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=1 / (12 * 9))

    def get_label(self):
        return "d"


# 2 balls * (area of circle - radius=15) / area of screen (dimx=400)
WATERWORLD_DEFAULT_PRIOR = 2 * ((15 * 15) * math.pi) / (400 * 400)

class WaterWorldRedLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "r"

class WaterWorldGreenLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "g"

class WaterWorldCyanLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "c"

class WaterWorldBlueLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "b"


class WaterWorldYellowLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "y"

class WaterWorldMagentaLabelExtractor(OfficeWorldAbstractLabelExtractor):
    def __init__(self, sensor_true_confidence: float,
                 sensor_false_confidence: float, seed: int = 0):
        super().__init__(sensor_true_confidence, sensor_false_confidence, seed, value_true_prior=WATERWORLD_DEFAULT_PRIOR)

    def get_label(self):
        return "m"
