import copy
from typing import Optional, Any, List

import gymnasium as gym
import ray

from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.utils import override
from ray.rllib.utils.typing import EpisodeType

from rm_marl.new_stack.learner.NewProbFFNSLLearner import NewProbFFNSLLearner
from rm_marl.rm_learning.ilasp.ilasp_example_representation import MultiISAExampleContainer
from rm_marl.rm_learning.trace_tracker import TraceTracker


class TraceStorage(ConnectorV2):
    """
    Augments the observation with the current RM state.
    Also, transforms the observation state to include curr RM state
    """

    # ORIG_OBS_KEY = "_orig_obs"
    # RM_STATE_KEY = "rm_state"

    def __init__(self,
                 input_observation_space: Optional[gym.Space] = None,
                 input_action_space: Optional[gym.Space] = None,
                 *,
                 learner=None,
                 **kwargs,
                 ):
        super().__init__(
            input_observation_space=input_observation_space,
            input_action_space=input_action_space,
            **kwargs,
        )

        self.learner = learner

    def __call__(self,
                 *,
                 data: Optional[Any],
                 episodes: List[EpisodeType],
                 **kwargs):
        for sa_episode in self.single_agent_episode_iterator(
                episodes, agents_that_stepped_only=False,
        ):
            t = self._create_trace(sa_episode)
            self.learner.update_examples.remote(t)

        return data

    def _create_trace(self, sa_episode):
        t = TraceTracker()

        # Agent did not run out of steps AND the batch sampling did not terminate the episode ahead of time
        is_complete = not sa_episode.is_truncated and sa_episode.is_done
        is_positive = sa_episode.get_return() > 0
        # if is_positive:
        #     breakpoint()
        # breakpoint()
        # TODO: the starting position is ignored in the orignal pipeline;
        #  need to check if that is okay
        # TODO: For unknown reason, the final observation seems duplicated in the
        #   sa_episode.get_infos(). Removed now; need to double check this. Could be an environment issue
        for info in sa_episode.get_infos()[1:]:
            t.update(info["labels"], is_positive, is_complete)

        # n_exists = any('n' in info['observations'] for info in sa_episode.get_infos())
        # if not is_complete and n_exists:
        #     breakpoint()
        #
        # f_exists = any('f' in info['observations'] for info in sa_episode.get_infos())
        # if is_complete and is_positive and not f_exists:
        #     breakpoint()

        return t
