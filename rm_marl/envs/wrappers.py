import abc
import copy
import random
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, List

import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit

from ..reward_machine import RewardMachine
from ..rm_transition.rm_transitioner import RMTransitioner
from ..rm_transition.prob_rm_transitioner import ProbRMTransitioner


# Important: This class needs to be the last one used. This also includes the RecordEpisodeStatisticsWrapper
class ProbabilisticRewardShaping(gym.Wrapper):

    def __init__(self, env, shaping_rm: RewardMachine, discount_factor: float = 0.99,
                 dist_fn: str = 'max'):
        super().__init__(env)

        self._discount_factor = discount_factor

        self._rm = None
        self._rm_transitioner = None
        self._rm_state_belief = None

        self._dist_fn = dist_fn

        self.set_shaping_rm(shaping_rm)

    def reset(self, **kwargs):
        self._rm_state_belief = self._rm_transitioner.get_initial_state()
        obs, info = super().reset(**kwargs)
        info["shaping_reward"] = 0
        info["original_reward"] = 0
        return obs, info

    def step(self, action):
        assert self._rm_state_belief is not None, "Environment was not properly reset before step()"

        obs, reward, terminated, truncated, info = super().step(action)

        assert "labels" in info and type(info["labels"]) == dict, "Unsupported labeling function, list of events needed"
        # assert "rm_state" in info and type(
        #     info["rm_state"]) == np.ndarray, "Unsupported env, belief over RM states needed"

        # Determine additional reward due to reward shaping
        new_rm_state_belief = self._rm_transitioner.get_next_state(self._rm_state_belief, info["labels"])
        shaping_reward = self._compute_shaping_reward(new_rm_state_belief)
        self._rm_state_belief = new_rm_state_belief

        # Provide info relating to the shaped reward
        info["shaping_reward"] = shaping_reward
        info["original_reward"] = reward

        return obs, reward + shaping_reward, terminated, truncated, info

    def _compute_shaping_reward(self, next_belief_vector):
        current_belief_vector = self._rm_state_belief
        state_potentials = np.array([self._rm.state_potentials[u] for u in self._rm.states])

        finite_indexes = np.isfinite(state_potentials)
        finite_potentials = np.where(finite_indexes, state_potentials, 0)

        current_prob_potentials = np.dot(finite_potentials, current_belief_vector)
        next_prob_potentials = np.dot(finite_potentials, next_belief_vector)
        shaping_reward = self._discount_factor * next_prob_potentials - current_prob_potentials

        return shaping_reward

    def set_shaping_rm(self, shaping_rm):
        self._rm = shaping_rm
        self._rm.compute_state_pontentials(self._dist_fn)

        self._rm_transitioner = ProbRMTransitioner(self._rm)
        self._rm_state_belief = None


class LabelThresholding(gym.Wrapper):

    def __init__(self, env: gym.Env, threshold: float = 0.5):
        assert 0 <= threshold <= 1, f"Threshold value \"{threshold}\" is outside of valid range [0, 1]"

        super().__init__(env)
        self._threshold = threshold

    def _apply_thresholding(self, labels):
        thresholded_labels = {}
        for label, confidence in labels.items():
            thresholded_labels[label] = 1.0 if confidence >= self._threshold else 0.0

        return thresholded_labels

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        assert "labels" in info, f"Info dictionary does not contain \"labels\" key"

        labels = info["labels"]
        info["labels"] = self._apply_thresholding(labels)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        assert "labels" in info, f"Info dictionary does not contain \"labels\" key"

        labels = info["labels"]
        info["labels"] = self._apply_thresholding(labels)

        return obs, info


class LabelingFunctionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, noisy: bool = False, seed: int = 0, sensor_true_confidence: float = 1,
                 sensor_false_confidence: float = 1):
        super().__init__(env)

        self.prev_obs = None

        if noisy:
            random.seed(seed)
            # Note: When we are designing environment ourselves, we tend to know underlying true
            #   values. So, it would be desirable to simulate "sensor" errors by simply using
            #   a Bernoulli distribution with p=sensor_true(false)_confidence respectively.
            self.sensor_true_confidence = sensor_true_confidence
            self.sensor_false_confidence = sensor_false_confidence

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

    @abc.abstractmethod
    def get_labels(self, info: dict):
        raise NotImplementedError("get_labels")

    @abc.abstractmethod
    def get_all_labels(self):
        raise NotImplementedError("get_all_labels")

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["labels"] = self.get_labels(info)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with kwargs."""
        obs, info = super().reset(**kwargs)
        info["labels"] = self.get_labels(info)
        return obs, info


class NoisyLabelingFunctionComposer(LabelingFunctionWrapper):
    def __init__(self, label_funs: List[LabelingFunctionWrapper]):
        assert len(label_funs) > 0

        super().__init__(label_funs[0].env, noisy=True)
        self.label_funs = label_funs

    def get_labels(self, info):
        labels = {}
        for label_fun in self.label_funs:
            labels.update(label_fun.get_labels(info))
        return labels

    def get_all_labels(self):
        ret = []
        for label_fun in self.label_funs:
            ret.extend(label_fun.get_all_labels())
        return ret


class AutomataWrapper(gym.Wrapper):
    class LabelMode(Enum):
        ALL = 0
        RM = 1
        STATE = 2

    class TerminationMode(Enum):
        RM = 0
        ENV = 1

    def __init__(
            self,
            env: gym.Env,
            rm_transitioner: RMTransitioner,
            label_mode: LabelMode = LabelMode.ALL,
            termination_mode: TerminationMode = TerminationMode.RM
    ):
        """
        label_mode:
            - ALL: returns all the labels returned by the labeling function
            - RM: returns only the labels present in the RM
            - STATE: returns only the lables that can be observed from the current state
        termination_mode:
            - RM: ends the episode when the RM reaches accepting/rejecting state
            - ENV: ends the episode when the underlying env is returning the end signal
        """
        assert hasattr(env, "get_labels"), "The LabelingFunctionWrapper is required"

        self.label_mode = label_mode
        self.termination_mode = termination_mode

        self.rm_transitioner = rm_transitioner
        self.u = None
        super().__init__(env)

    def filter_labels(self, labels, u):
        if isinstance(labels, dict):
            return {e: v for e, v in labels.items() if self._is_valid_event(e, u)}

        return [e for e in labels if self._is_valid_event(e, u)]

    def _is_valid_event(self, event, u):
        if self.label_mode == self.LabelMode.ALL:
            return True
        elif self.label_mode == self.LabelMode.RM:
            return event in self.rm.get_valid_events()
        elif self.label_mode == self.LabelMode.STATE:
            return event in self.rm.get_valid_events(u)

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["labels"] = self.filter_labels(info["labels"], self.u)
        simulated_updates = info.pop("env_simulated_updates", {})

        info["labels"] = self._apply_simulated_updates(info["labels"], simulated_updates)

        u_next = self.rm_transitioner.get_next_state(self.u, info["labels"])
        reward = self._get_reward(reward, u_next)
        self.u = u_next

        terminated = self._get_terminated(terminated)
        info["rm_state"] = self.u

        # Assume every trace is positive unless otherwise defined
        if "is_positive_trace" not in info:
            info["is_positive_trace"] = True

        return observation, reward, terminated, truncated, info

    def _get_reward(self, reward, u_next):
        return reward

    def _get_terminated(self, terminated):
        if self.termination_mode == self.TerminationMode.ENV:
            return terminated
        else:  # should be TerminationMode.RM
            return self.rm.is_state_terminal(self.u)

    def _apply_simulated_updates(self, original_labels, simulated_updates):
        labels = copy.deepcopy(original_labels)
        for e in original_labels:
            # apply simulated updates to the environment
            if e in simulated_updates:
                labels_update = simulated_updates[e](self.unwrapped)
                if labels_update:
                    labels = labels_update(labels)
        return labels

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.u = self.rm_transitioner.get_initial_state()

        info["labels"] = self.filter_labels(info.get("labels", {}), self.u)
        simulated_updates = info.pop("env_simulated_updates", {})
        info["labels"] = self._apply_simulated_updates(info["labels"], simulated_updates)

        u_next = self.rm_transitioner.get_next_state(self.u, info["labels"])
        self.u = u_next

        info["rm_state"] = self.u
        return obs, info


class RewardMachineWrapper(AutomataWrapper):

    def __init__(
            self,
            env: gym.Env,
            rm_transitioner: RMTransitioner,
            label_mode: AutomataWrapper.LabelMode = AutomataWrapper.LabelMode.ALL,
            termination_mode: AutomataWrapper.TerminationMode = AutomataWrapper.TerminationMode.RM,
            reward_function: callable = None
    ):
        super().__init__(env, rm_transitioner, label_mode, termination_mode)

        self.reward_function = reward_function or self._simple_reward_func

    @staticmethod
    def _simple_reward_func(rm, u, u_next, reward):
        return rm.get_reward(u, u_next)

    def _get_reward(self, reward, u_next):
        return self.reward_function(self.rm_transitioner.rm, self.u, u_next, reward)


class DiscreteToBoxObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(DiscreteToBoxObservationWrapper, self).__init__(env)

        # Check if the original observation space is discrete
        assert isinstance(env.observation_space, gym.spaces.Discrete), "The observation space must be discrete"

        # Define the new observation space as a Box space with one-hot encoded vectors
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(env.observation_space.n,), dtype=np.float32,
        )

    def observation(self, obs):
        # Convert the discrete observation into a one-hot encoded vector
        one_hot_obs = np.zeros(self.observation_space.shape)
        one_hot_obs[obs] = 1.0
        return one_hot_obs
