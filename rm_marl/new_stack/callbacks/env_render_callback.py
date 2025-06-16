"""
There will be a warning from tensorboardX regarding this callback.
However, wandb should work
Use parameters:
`python [script file name].py --enable-new-api-stack --env [env name e.g. 'ALE/Pong-v5']
--wandb-key=[your WandB API key] --wandb-project=[some WandB project name]
--wandb-run-name=[optional: WandB run name within --wandb-project]`
"""

from typing import Sequence, Optional, Union, List

import gymnasium
import numpy as np
from ray.rllib import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.utils.images import resize
from ray.rllib.utils.metrics.metrics_logger import MetricsLogger
from ray.rllib.utils.typing import EpisodeType


#  Bug: Can't prevent mulitple environments appear in some steps
#     The wandb logger is coupled with the training iteration.
#     So, we log the results when training is finished to prevent an iteration having multiple videos.
#     The episodes are shorter later, as the agent works better.
#
# We only render on worker with index 0 (which is the main thread).
#  Only evaluation is set up to be exeucted in the main thread (avoiding the limitations
#    of not being able to detect when the iteration has finished)
class EnvRenderCallback(DefaultCallbacks):
    """A custom callback to render the environment.

    This can be used to create videos of the episodes for some or all EnvRunners
    and some or all env indices (in a vectorized env). These videos can then
    be sent to e.g. WandB as shown in this example script here.

    We override the `on_episode_step` method to create a single ts render image
    and temporarily store it in the Episode object.
    """

    def __init__(self, **kwargs):
        super().__init__()
        # Main thread index
        self._env_runner_indices = [0]

        # self._episodes_seen = 0
        # TODO: extract as a parameter
        # Render every 1 episodes; only use this in evaluation for now
        # self._render_freq = 500 # 1000
        self._num_agents = None

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
        """On each env.step(), we add the render image to our Episode instance.

        Note that this would work with MultiAgentEpisodes as well.
        """

        for i, sa_episode in enumerate(
                ConnectorV2.single_agent_episode_iterator([episode], agents_that_stepped_only=False)):
            pass

        if (
                self._env_runner_indices is not None
                and env_runner.worker_index not in self._env_runner_indices
        ):
            return

        # Skip recording if this episode will not be rendered
        # if self._episodes_seen % self._render_freq != 0:
        #     return

        # If we have a vector env, only render the sub-env at index 0.
        if isinstance(env.unwrapped, gymnasium.vector.VectorEnv):
            images = env.envs[0].render()
        # Render the gym.Env.
        else:
            images = env.render()

        assert isinstance(images, list)
        self._num_agents = len(images)
        for i, image in enumerate(images):
            # 512, 1024
            # 512, 512
            # Original render images for CartPole are 400x600 (hxw). We'll downsize here to
            # a very small dimension (to save space and bandwidth).
            # IMPORTANT: Not resizing images often results in OOM error
            image = resize(image, 128, 256)
            # For WandB videos, we need to put channels first.
            image = np.transpose(image, axes=[2, 0, 1])
            # Add the compiled single-step image as temp. data to our Episode object.
            # Once the episode is done, we'll compile the video from all logged images
            # and log the video with the EnvRunner's `MetricsLogger.log_...()` APIs.
            # See below:
            # `on_episode_end()`: We compile the video and maybe store it).
            # `on_sample_end()` We log the best and worst video to the `MetricsLogger`.
            episode.add_temporary_timestep_data(f"agent{i}_render_images", image)

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
        if (
                self._env_runner_indices is not None
                and env_runner.worker_index not in self._env_runner_indices
        ):
            return

        # Skip if we should not render this episode
        # if self._episodes_seen % self._render_freq != 0:
        #     self._episodes_seen += 1
        #     return
        #
        # self._episodes_seen += 1
        for i in range(self._num_agents):
            # Pull all images from the temp. data of the episode.
            images = episode.get_temporary_timestep_data(f"agent{i}_render_images")
            # `images` is now a list of 3D ndarrays

            # Create a video from the images by simply stacking them AND
            # adding an extra B=1 dimension. Note that Tune's WandB logger currently
            # knows how to log the different data types by the following rules:
            # array is shape=3D -> An image (c, h, w).
            # array is shape=4D -> A batch of images (B, c, h, w).
            # array is shape=5D -> A video (1, L, c, h, w), where L is the length of the
            # video.
            # -> Make our video ndarray a 5D one.
            video = np.expand_dims(np.stack(images, axis=0), axis=0)

            # `video` is from the best episode in this cycle (iteration).
            metrics_logger.log_value(
                f"episode_videos/{i}",
                video,
                # Do not reduce the videos (across the various parallel EnvRunners).
                # This would not make sense (mean over the pixels?). Instead, we want to
                # log all best videos of all EnvRunners per iteration.
                reduce=None,
                # # B/c we do NOT reduce over the video data (mean/min/max), we need to
                # # make sure the list of videos in our MetricsLogger does not grow
                # # infinitely and gets cleared after each `reduce()` operation, meaning
                # # every time, the EnvRunner is asked to send its logged metrics.
                clear_on_reduce=True,
                # Good for preventing multiple writes per iteration
                # window=1,
            )