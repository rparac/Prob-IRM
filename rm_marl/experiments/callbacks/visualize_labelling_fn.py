from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class VisualizeLabelling(DefaultCallbacks):
    def __init__(self, **kwargs):
        super().__init__()
        # Main thread index
        self._env_runner_indices = [0]

        # Mapping from episode ID to max distance travelled thus far.
        self._episode_start_position = {}

    def on_episode_end(
            self,
            *,
            episode,
            env_runner,
            metrics_logger,
            env,
            env_index,
            rl_module,
            **kwargs,
    ) -> None:
        # Skip, if this EnvRunner's index is not in `self._env_runner_indices`.
        if (
                self._env_runner_indices is not None
                and env_runner.worker_index not in self._env_runner_indices
        ):
            return

        width = env.envs[0].unwrapped.width
        height = env.envs[0].unwrapped.height

        num_labels = 3
        heatmaps = [np.zeros((height, width), dtype=np.float32) for i in range(num_labels + 1)]

        labels = list(np.eye(num_labels))
        labels.append(np.zeros(num_labels))

        # 108 is the observation space size
        grid_pos = list(np.eye(108))

        net = rl_module['p0']

        for i in range(num_labels + 1):
            for idx, g_pos in enumerate(grid_pos):
                obs = np.concatenate((g_pos, labels[i]))
                obs = torch.Tensor(obs)
                out = net.predict_label(obs)
                y, x = idx % height, idx // height

                heatmaps[i][y,x] = out.item()
        breakpoint()

        # Create the actual heatmap image.
        # Use a colormap (e.g., 'hot') to map normalized values to RGB
        colormap = plt.get_cmap("coolwarm")  # try "hot" and "viridis" as well?
        heatmaps_rgb = [colormap(h_map) for h_map in heatmaps]
        # Convert RGBA to RGB by dropping the alpha channel and converting to uint8.
        heatmaps_rgb = [(h_map[:, :, :3] * 255).astype(np.uint8) for h_map in heatmaps_rgb] 

        for i in range(len(heatmaps)):
            metrics_logger.log_value(
                f"label_prob_map/{i}",
                heatmaps_rgb[i],
                reduce=None,
                clear_on_reduce=True,
            )



