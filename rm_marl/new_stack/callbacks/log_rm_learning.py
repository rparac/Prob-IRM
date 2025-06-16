import os
from typing import Optional, Union, List

import numpy as np
import ray
from ray.rllib import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType
from ray.tune.logger.tensorboardx import TBXLogger
from pdf2image import convert_from_path as read_pdf_image


class LogRMLearning(DefaultCallbacks):
    def __init__(self, rm_learner_actor: str = None, stop_iters: int = 0, use_wandb: bool = False, **kwargs):
        self._rm_learner = ray.get_actor(rm_learner_actor)
        self._stop_iters = stop_iters
        self._iters_done = 0
        # If wandb is not used; we log RMs to tensorboard. 
        # We cannot do both at the same time unfortunately.
        self._use_wandb = use_wandb

    def set_rm_learner(self, rm_learner_actor):
        print(f"New RM learner set to {rm_learner_actor}")
        self._rm_learner = ray.get_actor(rm_learner_actor)


    def on_train_result(
            self,
            *,
            algorithm: "Algorithm",
            metrics_logger: Optional[MetricsLogger] = None,
            result: dict,
            **kwargs,
    ) -> None:
        """
        Called at the end of iteration.
        We manually track the number of iterations to log at the end of training.
        """

        self._record_rm_learning_metrics(metrics_logger)
        self._plot_reward_machines(metrics_logger)

    def _record_rm_learning_metrics(self, metrics_logger: Optional[MetricsLogger]) -> None:
        stats = ray.get(self._rm_learner.get_statistics.remote())

        for k, v in stats.items():
            metrics_logger.log_value(
                f"rm_learner/{k}",
                value=v,
                reduce='max',
                clear_on_reduce=True,
            )

    def _plot_reward_machines(self, metrics_logger: Optional[MetricsLogger]) -> None:
        self._iters_done += 1
        # Need to do it one iteration before the end as tensorboard doesn't get called otherwise
        if self._iters_done != self._stop_iters - 1:
            return

        log_dir = str(ray.get(self._rm_learner.log_folder.remote()))

        rm_plot_entries: List[os.DirEntry] = [
            f for f in os.scandir(log_dir)
            if f.name.startswith('plot_') and f.name.endswith('.pdf')
        ]

        sorted_plot_entries = sorted(
            rm_plot_entries,
            key=lambda entry: int(entry.name.removeprefix("plot_").removesuffix(".pdf"))
        )

        for i, rm_plot_entry in enumerate(sorted_plot_entries):
            print("Saving image")
            plot_image = read_pdf_image(rm_plot_entry.path)[0]
            plot_rgb_array = np.asarray(plot_image)
            if not self._use_wandb:
                plot_rgb_array = np.transpose(plot_rgb_array, axes=[2, 0, 1])
                metrics_logger.log_value(
                    key=f"rm/plot/{i}",
                    value=plot_rgb_array,
                )
            else:
                metrics_logger.log_value(
                    key=f"rm/plot/{i}",
                    value=plot_rgb_array,
                    reduce=None,
                    clear_on_reduce=True,
                )
