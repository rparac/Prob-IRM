from typing import Optional, Union, Dict

import gymnasium as gym
import ray
from ray.rllib import BaseEnv, Policy

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.rl_module import RLModule
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID

from rm_marl.rm_learning.trace_tracker import TraceTracker


class StoreTracesCallback(DefaultCallbacks):
    def __init__(self, rm_learner_actor: str = None, **kwargs):
        self._rm_learner = ray.get_actor(rm_learner_actor)
        print(f"Got actor {rm_learner_actor}")
        self._traces = []


    def set_rm_learner(self, rm_learner_actor):
        print(f"New RM learner set to {rm_learner_actor}")
        self._rm_learner = ray.get_actor(rm_learner_actor)


    def on_episode_end(
            self,
            *,
            episode: Union[EpisodeType, EpisodeV2],
            env_runner: Optional["EnvRunner"] = None,
            metrics_logger: Optional[MetricsLogger] = None,
            env: Optional[gym.Env] = None,
            env_index: int,
            rl_module: Optional[RLModule] = None,
            # TODO (sven): Deprecate these args.
            worker: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            **kwargs,
    ) -> None:
        # Agent did not run out of steps AND the batch sampling did not terminate the episode ahead of time

        # We assume these are multi agent episodes
        for sa_episode in ConnectorV2.single_agent_episode_iterator([episode], agents_that_stepped_only=False):
            t = TraceTracker()
            # Should use is_truncated since is_terminated doesn't work well with a multi-agent adapter
            is_complete = not sa_episode.is_truncated and sa_episode.is_done

            last_info = sa_episode.get_infos(-1)
            # Prefer original reward if reward shaping is active
            ret = last_info["original_reward"] if "original_reward" in last_info else sa_episode.get_return()
            is_positive = ret > 0

            # TODO: the starting position is ignored in the orignal pipeline;
            #  need to check if that is okay
            # TODO: For unknown reason, the final observation seems duplicated in the
            #   episode.get_infos(). Removed now; need to double check this. Could be an environment issue
            for info in sa_episode.get_infos()[1:]:
                t.update(info["labels"], is_positive, is_complete)

            self._traces.append(t)
        #     # self._rm_learner.update_examples.remote(t)


    def on_sample_end(self, *, env_runner = None, metrics_logger = None, samples, worker = None, **kwargs):
        self._rm_learner.batch_update_examples.remote(self._traces)
        self._traces = []
