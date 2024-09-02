from typing import Optional, List, Any

import gymnasium as gym
import gymnasium.spaces
import numpy as np
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core import Columns
from ray.rllib.utils.spaces.space_utils import flatten_to_single_ndarray
from ray.rllib.utils.typing import EpisodeType

from rm_marl.reward_machine import RewardMachine
from rm_marl.rm_transition.prob_rm_transitioner import ProbRMTransitioner
from src.ray.rllib.utils.annotations import override


class RMStateConnector(ConnectorV2):
    """
    Augments the observation with the current RM state.
    Also, transforms the observation state to include curr RM state
    """

    ORIG_OBS_KEY = "_orig_obs"
    RM_STATE_KEY = "rm_state"

    @override(ConnectorV2)
    def recompute_observation_space_from_input_spaces(self) -> gym.Space:
        # We assume the input_observation space is a multi-agent space
        # return self.input_observation_space

        return self._convert_individual_space(self.input_observation_space)

        # ret = {}
        # for ag_id, obs_space in self.input_observation_space.spaces.items():
        #     ret[ag_id] = self._convert_individual_space(obs_space)
        # return gymnasium.spaces.Dict(ret)

    def _convert_individual_space(self, obs_space):
        return gymnasium.spaces.Dict(
            {
                self.ORIG_OBS_KEY: obs_space,
                self.RM_STATE_KEY: gym.spaces.Box(low=0, high=1, shape=(len(self.rm.states),))
            }
        )

    def __init__(self,
                 input_observation_space: Optional[gym.Space] = None,
                 input_action_space: Optional[gym.Space] = None,
                 *,
                 rm: RewardMachine = None,
                 as_learner_connector: bool = False,
                 **kwargs,
                 ):
        self.rm = rm
        self.transitioner = ProbRMTransitioner(self.rm)

        self._as_learner_connector = as_learner_connector

        super().__init__(
            input_observation_space=input_observation_space,
            input_action_space=input_action_space,
            **kwargs,
        )

    def _call_as_learner(self,
                         *,
                         data: Optional[Any],
                         episodes: List[EpisodeType],
                         **kwargs):
        # Learner connector pipeline. Episodes have been finalized/numpy'ized
        for sa_episode in self.single_agent_episode_iterator(
            episodes, agents_that_stepped_only=False,
        ):
            old_infos = sa_episode.get_infos()

            curr_state = self.transitioner.get_initial_state()
            state_info = []
            for info in old_infos:
                curr_state = self.transitioner.get_next_state(curr_state, info["labels"])
                state_info.append(curr_state)

            # TODO: we are ignoring the last state here. Need to double check if
            #  that is fine
            self.add_n_batch_items(
                batch=data,
                column=self.RM_STATE_KEY,
                items_to_add=state_info[:-1],
                num_items=len(sa_episode),
                single_agent_episode=sa_episode,
            )
        return data

    def _call_as_env_connector(self,
                               *,
                               data: Optional[Any],
                               episodes: List[EpisodeType],
                               **kwargs):
        # Env-to-module pipeline. Episodes still operate on lists

        for i, sa_episode in enumerate(self.single_agent_episode_iterator(episodes)):
            # TODO: make more efficient; we are always computing all RM steps
            old_infos = sa_episode.get_infos()
            curr_state = self.transitioner.get_initial_state()
            # breakpoint()
            state_info = []
            for info in old_infos:
                curr_state = self.transitioner.get_next_state(curr_state, info["labels"])
                state_info.append(curr_state)

            self.add_batch_item(
                batch=data,
                column=self.RM_STATE_KEY,
                item_to_add=state_info[-1],
                single_agent_episode=sa_episode,
            )
            # sa_episode.set_observations(new_data=augmented_obs)

        return data

    def __call__(self,
                 *,
                 data: Optional[Any],
                 episodes: List[EpisodeType],
                 **kwargs):
        if self._as_learner_connector:
            return self._call_as_learner(data=data, episodes=episodes, **kwargs)

        return self._call_as_env_connector(data=data, episodes=episodes, **kwargs)


    def _generate_curr_rm_state(self, infos):
        # Given an episode worth of infos, generate corresponding rm_states
        state_info = [self.transitioner.get_initial_state()]
        for info in infos:
            new_state = self.transitioner.get_next_state(state_info[-1], info["labels"])
            state_info.append(new_state)
        return np.ndarray(state_info)
