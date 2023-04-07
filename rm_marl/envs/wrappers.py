import abc
import copy
from dataclasses import dataclass
import os
from enum import Enum
from itertools import groupby
from typing import TYPE_CHECKING

import gym
from gym.wrappers import RecordEpisodeStatistics, TimeLimit

if TYPE_CHECKING:
    from ..reward_machine import RewardMachine


class NumberStepsDiscountedRewardWrapper(gym.Wrapper):
    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)

        episode_stats = info.get("episode", {})
        if terminated and episode_stats:
            reward = episode_stats["r"] / episode_stats["l"]
        elif terminated:
            raise ValueError(info)

        return observation, reward, terminated, truncated, info


class LabelingFunctionWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.prev_agent_locations = self.unwrapped.agent_locations

    @abc.abstractmethod
    def get_labels(self, obs: dict = None, prev_obs: dict = None):
        raise NotImplementedError("get_labels")

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        info["labels"] = self.get_labels()
        self.prev_agent_locations = copy.deepcopy(self.agent_locations)
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Resets the environment with kwargs."""
        ret = super().reset(**kwargs)
        self.prev_agent_locations = self.unwrapped.agent_locations
        return ret


@dataclass
class RandomLabelingConfig:
    proba: float
    condition: callable
    env_update: callable


class RandomLabelingFunctionWrapper(gym.Wrapper):


    def __init__(self, env: gym.Env, random_events: dict):
        """
        random_event: {event: (probability, function(env))}
        """
        self.random_events = random_events or {}
        self.trace = []
        super().__init__(env)

    @property
    def flatten_trace(self):
        return tuple(e for es in self.trace for e in es)

    def get_labels(self, _obs: dict = None, _prev_obs: dict = None):
        valid_events = [e for e, c in self.random_events.items() if c.condition(self)]
        if not valid_events:
            return []

        event = self.np_random.choice(valid_events)
        config = self.random_events[event]
        if self.np_random.random() <= config.proba:
            return [event]

        return []

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        labels, simulated_env_updates = info.get("labels", []), {}
        # two events can not happen at the same time
        if not labels:
            labels = self.get_labels()
            for l in labels:
                simulated_env_updates[l] = self.random_events[l].env_update
        
        self.trace.append(labels or [])
        info["labels"] = labels
        info["env_simulated_updates"] = simulated_env_updates
        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self.trace.clear()
        return super().reset(**kwargs)

    @staticmethod
    def condition_true(e):
        return True


class RewardMachineWrapper(gym.Wrapper):
    class LabelMode(Enum):
        ALL = 0
        RM = 1
        STATE = 2

    def __init__(
        self, env: gym.Env, rm: "RewardMachine", label_mode: LabelMode = LabelMode.ALL
    ):
        """
        label_mode:
            - ALL: returns all the labels returned by the labeling function
            - RM: returns only the labels present in the RM
            - STATE: returns only the lables that can be observed from the current state
        """
        assert hasattr(env, "get_labels"), "The LabelingFunctionWrapper is required"

        self.label_mode = label_mode

        self.rm = rm
        self.u = self.rm.u0
        super().__init__(env)

    def filter_labels(self, labels, u):
        return [e for e in labels if self._is_valid_event(e, u)]

    def _is_valid_event(self, event, u):
        if self.label_mode == self.LabelMode.ALL:
            return True
        elif self.label_mode == self.LabelMode.RM:
            return event in self.rm.get_valid_events()
        elif self.label_mode == self.LabelMode.STATE:
            return event in self.rm.get_valid_events(u)

    def step(self, action):
        observation, _reward, _terminated, truncated, info = super().step(action)
        info["labels"] = self.filter_labels(info["labels"], self.u)
        simulated_updates = info.pop("env_simulated_updates", {})

        reward = 0
        for e in info["labels"]:
            # apply simulated updates to the environment
            if e in simulated_updates:
                simulated_updates[e](self.unwrapped)
        
        u_next = self.rm.get_next_state(self.u, info["labels"])
        reward += self.rm.get_reward(self.u, u_next)
        self.u = u_next

        terminated = self.rm.is_state_terminal(self.u)
        info["rm_state"] = self.u

        return observation, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.u = self.rm.u0
        info["rm_state"] = self.u
        return obs, info


class SingleAgentEnvWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, agent_id: str):
        self.agent_id = agent_id

        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        obs = {aid: apos for aid, apos in obs.items() if aid == self.agent_id}
        return obs, info

    def step(self, action):
        action = {aid: a for aid, a in action.items() if aid == self.agent_id}

        observation, reward, terminated, truncated, info = super().step(action)

        observation = {
            aid: apos for aid, apos in observation.items() if aid == self.agent_id
        }

        return observation, reward, terminated, truncated, info

