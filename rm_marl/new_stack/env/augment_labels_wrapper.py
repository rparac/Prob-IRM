from typing import Any, SupportsFloat

import gymnasium
import numpy as np
from gymnasium.core import WrapperObsType, WrapperActType

from rm_marl.new_stack.utils.gymnasium import gym_getattr


class AugmentLabelsWrapper(gymnasium.Wrapper):
    """
    Wrapper which augments the observation with seen labels (but without the RM)
    Important for baselines as they would otherwise have less information to learn from
    """

    def __init__(self, env: gymnasium.Wrapper):
        super().__init__(env)

        # We assume NoisyLabelingFunctionComposer is used before this component
        assert gym_getattr(env, "label_funs") is not None

        num_labels = len(gym_getattr(env, "label_funs"))

        self.observation_space = gymnasium.spaces.Box(
            low=env.observation_space.low[0], high=env.observation_space.high[0],  # type: ignore
            dtype=env.observation_space.dtype,
            shape=(env.observation_space.shape[0] + num_labels,)
        )

    @staticmethod
    def _augment_obs_with_labels(obs, info):
        sorted_keys = sorted(info["labels"].keys())
        sorted_values = np.array([info["labels"][k] for k in sorted_keys], dtype=np.float32)
        new_obs = np.concatenate((obs, sorted_values))
        return new_obs

    def reset(
            self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return self._augment_obs_with_labels(obs, info), info

    def step(
            self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._augment_obs_with_labels(obs, info), reward, terminated, truncated, info
