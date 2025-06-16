"""
Wrapper which is used to directly augment RM information as part of the observation
Assumes, observation is a Box and the RM info is obtained as labels
"""
from typing import SupportsFloat, Any

import gymnasium
from gymnasium.core import WrapperActType, WrapperObsType
import numpy as np

from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner


class RMWrapper(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Wrapper, rm=None):
        super().__init__(env)
        assert isinstance(env.observation_space, gymnasium.spaces.Box)

        _rm = rm if rm is not None else RewardMachine.default_rm()

        self.rm_transitioner = ProbRMTransitioner(_rm)

        self.observation_space_wo_rm = env.observation_space
        self._augment_obs_space_with_rm()

        self._curr_rm_state = None

    def _augment_obs_space_with_rm(self):
        init_state = self.rm_transitioner.get_initial_state()

        low_val = self.observation_space_wo_rm.low[0]  # type: ignore
        high_val = self.observation_space_wo_rm.high[0]  # type: ignore

        self.observation_space = gymnasium.spaces.Box(
            low=low_val, high=high_val,
            dtype=self.observation_space_wo_rm.dtype,
            shape=(self.observation_space_wo_rm.shape[0] + init_state.shape[0],),
        )

    def update_rm(self, rm: RewardMachine):
        # needs to change the observation space
        self.rm_transitioner = ProbRMTransitioner(rm)
        # TODO: interrupt episode when RM is updated
        self._curr_rm_state = self.rm_transitioner.get_initial_state()
        self._augment_obs_space_with_rm()

        # set shaping reward machine if it exits
        if hasattr(self, 'set_shaping_rm'):
            self.set_shaping_rm(rm)

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        rm_state = self.rm_transitioner.get_initial_state()
        obs, info = self.env.reset()
        rm_state = self.rm_transitioner.get_next_state(rm_state, info["labels"])
        self._curr_rm_state = rm_state
        new_obs = np.concatenate((obs, rm_state))
        return new_obs, info

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        rm_state = self.rm_transitioner.get_next_state(self._curr_rm_state, info["labels"])
        self._curr_rm_state = rm_state

        # We intentionally interrupt the episode if the RM believes the episode should be done
        if self.rm_transitioner.rm.is_state_terminal(rm_state) and not terminated:
            return np.concatenate((obs, rm_state)), reward, terminated, True, info

        return np.concatenate((obs, rm_state)), reward, terminated, truncated, info
