from typing import TYPE_CHECKING, Optional

from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger

if TYPE_CHECKING:
    from ray.rllib.algorithms.algorithm import Algorithm


class MinimizeLogs(DefaultCallbacks):

    def __init__(self, **kwargs):
        super().__init__()

    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            metrics_logger: Optional[MetricsLogger] = None,
            result: dict,
            **kwargs,
    ) -> None:

        del result['timers']
        del result['fault_tolerance']
        del result['perf']
        del result['learners']
        # del result['iterations_since_restore'] -> Might be used when checkpointing
        del result['time_this_iter_s']
        # del result['time_since_restore'] -> Might be used when checkpointing
        del result['timestamp']
        # del result['num_env_steps_trained_lifetime'] -> Used by the CLIReporter

        if 'evaluation' in result:
            del result['evaluation']

        del result['env_runners']['episode_len_min']
        del result['env_runners']['episode_len_max']
        del result['env_runners']['episode_len_mean']
        del result['env_runners']['episode_duration_sec_mean']
        del result['env_runners']['num_episodes_lifetime']
        del result['env_runners']['num_env_steps_sampled_lifetime']
        del result['env_runners']['num_agent_steps_sampled_lifetime']

        # del result['env_runners']['agent_steps']
        # del result['env_runners']['num_env_steps_sampled']
        # del result['env_runners']['num_agent_steps_sampled']
        del result['env_runners']['num_module_steps_sampled']

        # del result['env_runners']['num_env_steps_sampled_lifetime']
        # del result['env_runners']['num_agent_steps_sampled_lifetime']
        del result['env_runners']['num_module_steps_sampled_lifetime']

        # del result['env_runners']['agent_episode_returns_mean']
        del result['env_runners']['module_episode_returns_mean']
