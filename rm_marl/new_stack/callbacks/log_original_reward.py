from typing import Union, Optional, Dict, List

import gymnasium as gym
from ray.rllib import BaseEnv, Policy, SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.rl_module import RLModule
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType, PolicyID

"""
For an unknown reason the value in original episode return is slightly different 
 from the return provided by rllib (the graphs have the same shape).
 Not sure why this is the case, but even in this form it's useful enough
"""


class LogOriginalReward(DefaultCallbacks):
    def __init__(self, **kwargs):
        super().__init__()

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
        if env_runner.worker_index == 0:
            for i, sa_episode in enumerate(ConnectorV2.single_agent_episode_iterator([episode], agents_that_stepped_only=False)):
                original_reward = sa_episode.get_return()
                # We only log the reward that happened in this evaluation episode
                metrics_logger.log_value(
                    f"fixed_agent_episode_return/{i}",
                    value=original_reward,
                    reduce='mean',
                    # clear_on_reduce=True,
                    window=1,
                )


        single_agent_rewards = []
        for i, sa_episode in enumerate(
                ConnectorV2.single_agent_episode_iterator([episode], agents_that_stepped_only=False)):
            original_reward = 0
            for info in sa_episode.get_infos()[1:]:
                # No need to report if reward shaping is not used
                if "original_reward" not in info:
                    return
                # Akin to previous paper we assume steping on plant gets reward 0 for logging
                rew = max(info["original_reward"], 0)
                original_reward += rew
            single_agent_rewards.append(original_reward)

            metrics_logger.log_value(
                f"original_agent_episode_return_mean/{i}",
                value=original_reward,
                reduce='mean',
                clear_on_reduce=True,
            )
            metrics_logger.log_value(
                f"original_agent_episode_return_max/{i}",
                value=original_reward,
                reduce='max',
                clear_on_reduce=True,
            )
            metrics_logger.log_value(
                f"original_agent_episode_return_min/{i}",
                value=original_reward,
                reduce='min',
                clear_on_reduce=True,
            )

        avg_single_average = sum(single_agent_rewards) / len(single_agent_rewards)
        metrics_logger.log_value(
            "original_episode_return_mean",
            value=avg_single_average,
            reduce='mean',
            clear_on_reduce=True,
        )

        metrics_logger.log_value(
            "original_episode_return_min",
            value=avg_single_average,
            reduce='min',
            clear_on_reduce=True,
        )
        metrics_logger.log_value(
            "original_episode_return_max",
            value=avg_single_average,
            reduce='max',
            clear_on_reduce=True,
        )

        multi_agent_reward = sum(single_agent_rewards)
        metrics_logger.log_value(
            "original_episode_return_mean_sum",
            value=multi_agent_reward,
            reduce='mean',
            clear_on_reduce=True,
        )

