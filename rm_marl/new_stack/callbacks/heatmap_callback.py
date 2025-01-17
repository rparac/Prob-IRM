"""Example of adding custom metrics to the results returned by `EnvRunner.sample()`.

We use the `MetricsLogger` class, which RLlib provides inside all its components (only
when using the new API stack through
`config.api_stack(_enable_rl_module_and_learner=True,
_enable_env_runner_and_connector_v2=True)`),
and which offers a unified API to log individual values per iteration, per episode
timestep, per episode (as a whole), per loss call, etc..
`MetricsLogger` objects are available in all custom API code, for example inside your
custom `Algorithm.training_step()` methods, custom loss functions, custom callbacks,
and custom EnvRunners.

This example:
    - demonstrates how to write a custom Callbacks subclass, which overrides some
    EnvRunner-bound methods, such as `on_episode_start`, `on_episode_step`, and
    `on_episode_end`.
    - shows how to temporarily store per-timestep data inside the currently running
    episode within the EnvRunner (and the callback methods).
    - shows how to extract this temporary data again when the episode is done in order
    to further process the data into a single, reportable metric.
    - explains how to use the `MetricsLogger` API to create and log different metrics
    to the final Algorithm's iteration output. These include - but are not limited to -
    a 2D heatmap (image) per episode, an average per-episode metric (over a sliding
    window of 200 episodes), a maximum per-episode metric (over a sliding window of 100
    episodes), and an EMA-smoothed metric.

In this script, we define a custom `DefaultCallbacks` class and then override some of
its methods in order to define custom behavior during episode sampling. In particular,
we add custom metrics to the Algorithm's published result dict (once per
iteration) before it is sent back to Ray Tune (and possibly a WandB logger).

For demonstration purposes only, we log the following custom metrics:
- A 2D heatmap showing the frequency of all accumulated y/x-locations of Ms Pacman
during an episode. We create and log a separate heatmap per episode and limit the number
of heatmaps reported back to the algorithm by each EnvRunner to 10 (`window=10`).
- The maximum per-episode distance travelled by Ms Pacman over a sliding window of 100
episodes.
- The average per-episode distance travelled by Ms Pacman over a sliding window of 200
episodes.
- The EMA-smoothed number of lives of Ms Pacman at each timestep (across all episodes).


How to run this script
----------------------
`python [script file name].py --enable-new-api-stack --wandb-key [your WandB key]
--wandb-project [some project name]`

For debugging, use the following additional command line options
`--no-tune --num-env-runners=0`
which should allow you to set breakpoints anywhere in the RLlib code and
have the execution stop there for inspection and debugging.

For logging to your WandB account, use:
`--wandb-key=[your WandB API key] --wandb-project=[some project name]
--wandb-run-name=[optional: WandB run name (within the defined project)]`


Results to expect
-----------------
This script has not been finetuned to actually learn the environment. Its purpose
is to show how you can create and log custom metrics during episode sampling and
have these stats be sent to WandB for further analysis.

However, you should see training proceeding over time like this:
+---------------------+----------+----------------+--------+------------------+
| Trial name          | status   | loc            |   iter |   total time (s) |
|                     |          |                |        |                  |
|---------------------+----------+----------------+--------+------------------+
| PPO_env_efd16_00000 | RUNNING  | 127.0.0.1:6181 |      4 |          72.4725 |
+---------------------+----------+----------------+--------+------------------+
+------------------------+------------------------+------------------------+
|    episode_return_mean |   num_episodes_lifetim |   num_env_steps_traine |
|                        |                      e |             d_lifetime |
|------------------------+------------------------+------------------------|
|                  76.4  |                     45 |                   8053 |
+------------------------+------------------------+------------------------+
"""
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.connectors.connector_v2 import ConnectorV2


class HeatmapCallback(DefaultCallbacks):
    """A custom callback to extract information from MsPacman and log these.

    This callback logs:
    - the positions of MsPacman over an episode to produce heatmaps from this data.
    At each episode timestep, the current pacman (y/x)-position is determined and added
    to the episode's temporary storage. At the end of an episode, a simple 2D heatmap
    is created from this data and the heatmap is logged to the MetricsLogger (to be
    viewed in WandB).
    - the max distance travelled by MsPacman per episode, then averaging these max
    values over a window of size=100.
    - the mean distance travelled by MsPacman per episode (over an infinite window).
    - the number of lifes of MsPacman EMA-smoothed over time.

    This callback can be setup to only log stats on certain EnvRunner indices through
    the `env_runner_indices` c'tor arg.
    """

    def __init__(self, **kwargs):
        """Initializes an MsPacmanHeatmapCallback instance.

        Args:
            env_runner_indices: The (optional) EnvRunner indices, for this callback
                should be active. If None, activates the heatmap for all EnvRunners.
                If a Sequence type, only logs/heatmaps, if the EnvRunner index is found
                in `env_runner_indices`.
        """
        super().__init__()
        # Main thread index
        self._env_runner_indices = [0]

        # Mapping from episode ID to max distance travelled thus far.
        self._episode_start_position = {}

    def on_episode_start(
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

        yx_pos = self._get_agents_position(episode, env)
        for i, (y, x) in yx_pos.items():
            self._episode_start_position[i] = (y, x)

    def on_episode_step(
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
        """Adds current agent y/x-position to episode's temporary data."""

        # Skip, if this EnvRunner's index is not in `self._env_runner_indices`.
        if (
                self._env_runner_indices is not None
                and env_runner.worker_index not in self._env_runner_indices
        ):
            return

        yx_positions = self._get_agents_position(episode, env)
        for i, yx_pos in yx_positions.items():
            episode.add_temporary_timestep_data(f"agent{i}_yx_pos", yx_pos)
            dist_travelled = self._compute_dist_travelled(i, yx_pos)
            episode.add_temporary_timestep_data(f"agent{i}_dist_travelled", dist_travelled)

    # Compute distance to the starting position for agent i
    def _compute_dist_travelled(self, agent_i, yx_pos):
        return np.sqrt(
            np.sum(
                np.square(
                    np.array(self._episode_start_position[agent_i])
                    - np.array(yx_pos)
                )
            )
        )

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

        # Get all pacman y/x-positions from the episode.
        for i in range(len(self._episode_start_position)):
            yx_positions = episode.get_temporary_timestep_data(f"agent{i}_yx_pos")
            dists_travelled = episode.get_temporary_timestep_data(f"agent{i}_dist_travelled")
            self._create_heatmap(i, yx_positions, height, width, metrics_logger)
            self._log_dist_travelled(i, dists_travelled, metrics_logger)

        # Erase the start position record.
        self._episode_start_position = {}

    @staticmethod
    def _get_agents_position(episode, env):
        results = {}
        for sa_episode in ConnectorV2.single_agent_episode_iterator([episode], agents_that_stepped_only=True):
            # if not sa_episode.is_done:
            results[sa_episode.agent_id] = HeatmapCallback._get_yx_pos(sa_episode, env)
        return results

    @staticmethod
    def _get_yx_pos(sa_episode, env):
        width = env.envs[0].unwrapped.width
        height = env.envs[0].unwrapped.height

        curr_state = sa_episode.get_observations(-1)
        idx = np.argmax(curr_state[:width * height])

        y = idx % height
        x = idx // height
        return y, x

    @staticmethod
    def _create_heatmap(agent_id, yx_positions, height, width, metrics_logger):
        # h x w
        # heatmap = np.zeros((width * 10, height * 10), dtype=np.int32)
        heatmap = np.zeros((height, width), dtype=np.int32)
        for yx_pos in yx_positions:
            heatmap[height - yx_pos[0] - 1, yx_pos[1]] += 1

        # Create the actual heatmap image.
        # Normalize the heatmap to values between 0 and 1
        norm = Normalize(vmin=heatmap.min(), vmax=heatmap.max())
        # Use a colormap (e.g., 'hot') to map normalized values to RGB
        colormap = plt.get_cmap("coolwarm")  # try "hot" and "viridis" as well?
        # Returns a (64, 64, 4) array (RGBA).
        heatmap_rgb = colormap(norm(heatmap))
        # Convert RGBA to RGB by dropping the alpha channel and converting to uint8.
        heatmap_rgb = (heatmap_rgb[:, :, :3] * 255).astype(np.uint8)
        # Log the image.
        metrics_logger.log_value(
            f"heatmap/{agent_id}",
            heatmap_rgb,
            reduce=None,
            clear_on_reduce=True,
        )

    @staticmethod
    def _log_dist_travelled(agent_id, dists_travelled, metrics_logger):
        # Get the max distance travelled for this episode.
        dist_travelled = np.max(
            dists_travelled
        )

        # Log the max. dist travelled in this episode (window=100).
        metrics_logger.log_value(
            f"max_dist_travelled/{agent_id}",
            dist_travelled,
            # For future reductions (e.g. over n different episodes and all the
            # data coming from other env runners), reduce by max.
            reduce="max",
            # Always keep the last 100 values and max over this window.
            window=100,
        )

        # Log the average dist travelled per episode (window=200).
        metrics_logger.log_value(
            f"mean_dist_travelled/{agent_id}",
            dist_travelled,
            reduce="mean",  # <- default
            # Always keep the last 200 values and average over this window.
            window=200,
        )
