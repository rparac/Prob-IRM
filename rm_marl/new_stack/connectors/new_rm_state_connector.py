"""
I hope to implement env-to-module connector
"""
from typing import Optional, Any, Dict, List

import gymnasium as gym
import numpy as np
import ray

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.rl_module import RLModule
from ray.rllib.utils import override
from ray.rllib.utils.typing import EpisodeType

from rm_marl.agent import RewardMachineAgent
from rm_marl.new_stack.learner.NewProbFFNSLLearner import NewProbFFNSLLearner
from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner


class NewRMStateConnector(ConnectorV2):
    def __init__(self,
                 input_observation_space: Optional[gym.Space] = None,
                 input_action_space: Optional[gym.Space] = None,
                 *,
                 rm: RewardMachine = None,
                 **kwargs,
                 ):

        # TODO: abstract the name
        actor_name = "rm_learner_actor"

        self._rm_learner = ray.get_actor(actor_name)
        rm_ref = self._rm_learner.get_rm.remote()
        self.rm = ray.get(rm_ref)
        self.initial_rm = self.rm
        self.transitioner = ProbRMTransitioner(self.rm)
        self._current_state = self.transitioner.get_initial_state()

        super().__init__(
            input_observation_space=input_observation_space,
            input_action_space=input_action_space,
            **kwargs,
        )

    @override(ConnectorV2)
    def recompute_observation_space_from_input_spaces(self) -> gym.Space:
        # Odd bug where initially this is not a gym space
        if not isinstance(self.input_observation_space, gym.Space):
            return self.input_observation_space

        # breakpoint()
        rm_ref = self._rm_learner.get_rm.remote()
        self.rm = ray.get(rm_ref)

        if len(self.rm.states) > 1:
            raise RuntimeError(self.input_observation_space, self.rm, len(self.rm.states))

        input_observation_space = self.input_observation_space # self.input_observation_space.envs[0].observation_space

        low_val = input_observation_space.low[0]  # type: ignore
        high_val = input_observation_space.high[0]  # type: ignore
        return gym.spaces.Box(
            low=low_val, high=high_val,
            dtype=input_observation_space.dtype,
            shape=(input_observation_space.shape[0] + len(self.rm.states) - len(self.initial_rm.states),),
        )

    @override(ConnectorV2)
    def __call__(
            self,
            *,
            rl_module: RLModule,
            data: Optional[Dict[str, Any]],
            episodes: List[EpisodeType],
            explore: Optional[bool] = None,
            shared_data: Optional[dict] = None,
            **kwargs,
    ) -> Any:
        for sa_episode in self.single_agent_episode_iterator(
                episodes, agents_that_stepped_only=True
        ):
            # Episode is not finalized yet and thus still operates on lists of items.
            assert not sa_episode.is_finalized

            last_infos = sa_episode.get_infos(-1)
            self._current_state = self.transitioner.get_next_state(self._current_state, last_infos)
            last_obs = sa_episode.get_observations(-1)

            new_obs = np.concatenate((last_obs, self._current_state))
            if sa_episode.is_truncated or sa_episode.is_terminated:
                self._current_state = self.transitioner.get_initial_state()

            # Write new observation directly back into the episode.
            sa_episode.set_observations(at_indices=-1, new_data=new_obs)
            #  We set the Episode's observation space to ours so that we can safely
            #  set the last obs to the new value (without causing a space mismatch
            #  error).
            sa_episode.observation_space = self.observation_space

        return data
