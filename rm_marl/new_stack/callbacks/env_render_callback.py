"""
There will be a warning from tensorboardX regarding this callback.
However, wandb should work
Use parameters:
`python [script file name].py --enable-new-api-stack --env [env name e.g. 'ALE/Pong-v5']
--wandb-key=[your WandB API key] --wandb-project=[some WandB project name]
--wandb-run-name=[optional: WandB run name within --wandb-project]`
"""


from typing import Sequence, Optional

import gymnasium
import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.images import resize

class EnvRenderCallback(DefaultCallbacks):
    """A custom callback to render the environment.

    This can be used to create videos of the episodes for some or all EnvRunners
    and some or all env indices (in a vectorized env). These videos can then
    be sent to e.g. WandB as shown in this example script here.

    We override the `on_episode_step` method to create a single ts render image
    and temporarily store it in the Episode object.
    """

    def __init__(self, env_runner_indices: Optional[Sequence[int]] = None):
        super().__init__()
        # Only render and record on certain EnvRunner indices?
        self.env_runner_indices = env_runner_indices
        # Per sample round (on this EnvRunner), we want to only log the best- and
        # worst performing episode's videos in the custom metrics. Otherwise, too much
        # data would be sent to WandB.
        self.best_episode_and_return = (None, float("-inf"))
        self.worst_episode_and_return = (None, float("inf"))

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
        if (
                self.env_runner_indices is not None
                and env_runner.worker_index not in self.env_runner_indices
        ):
            return

        # If we have a vector env, only render the sub-env at index 0.
        if isinstance(env.unwrapped, gymnasium.vector.VectorEnv):
            image = env.envs[0].render()
        # Render the gym.Env.
        else:
            image = env.render()

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
        episode.add_temporary_timestep_data("render_images", image)

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
        """Computes episode's return and compiles a video, iff best/worst in this iter.

        Note that the actual logging to the EnvRunner's MetricsLogger only happens
        at the very env of sampling (when we know, which episode was the best and
        worst). See `on_sample_end` for the implemented logging logic.
        """
        # Get the episode's return.
        episode_return = episode.get_return()

        # Better than the best or worse than worst Episode thus far?
        if (
                episode_return > self.best_episode_and_return[1]
                or episode_return < self.worst_episode_and_return[1]
        ):
            # Pull all images from the temp. data of the episode.
            images = episode.get_temporary_timestep_data("render_images")
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
            if episode_return > self.best_episode_and_return[1]:
                self.best_episode_and_return = (video, episode_return)
            # `video` is worst in this cycle (iteration).
            else:
                self.worst_episode_and_return = (video, episode_return)

    def on_sample_end(
            self,
            *,
            env_runner,
            metrics_logger,
            samples,
            **kwargs,
    ) -> None:
        """Logs the best and worst video to this EnvRunner's MetricsLogger."""
        # Best video.
        if self.best_episode_and_return[0] is not None:
            metrics_logger.log_value(
                "episode_videos_best",
                self.best_episode_and_return[0],
                # Do not reduce the videos (across the various parallel EnvRunners).
                # This would not make sense (mean over the pixels?). Instead, we want to
                # log all best videos of all EnvRunners per iteration.
                reduce=None,
                # B/c we do NOT reduce over the video data (mean/min/max), we need to
                # make sure the list of videos in our MetricsLogger does not grow
                # infinitely and gets cleared after each `reduce()` operation, meaning
                # every time, the EnvRunner is asked to send its logged metrics.
                clear_on_reduce=True,
            )
            self.best_episode_and_return = (None, float("-inf"))
        # Worst video.
        if self.worst_episode_and_return[0] is not None:
            metrics_logger.log_value(
                "episode_videos_worst",
                self.worst_episode_and_return[0],
                # Same logging options as above.
                reduce=None,
                clear_on_reduce=True,
            )
            self.worst_episode_and_return = (None, float("inf"))
