from ray.rllib.algorithms.callbacks import DefaultCallbacks


class CallbackComposer(DefaultCallbacks):
    def __init__(self, callbacks_cls, **kwargs):
        super().__init__()
        # Ray requires passing classes instead of objects, so we keep the design consistent
        self.callbacks = [_cls(**kwargs) for _cls in callbacks_cls]

    def set_rm_learner(self, actor_name):
        for callback in self.callbacks:
            if hasattr(callback, 'set_rm_learner'):
                callback.set_rm_learner(actor_name)


    def on_algorithm_init(self, **kwargs) -> None:
        """Callback run when a new Algorithm instance has finished setup."""
        for callback in self.callbacks:
            callback.on_algorithm_init(**kwargs)

    def on_workers_recreated(self, **kwargs) -> None:
        """Callback run after one or more workers have been recreated."""
        for callback in self.callbacks:
            callback.on_workers_recreated(**kwargs)

    def on_checkpoint_loaded(self, **kwargs) -> None:
        """Callback run when an Algorithm has loaded a new state from a checkpoint."""
        for callback in self.callbacks:
            callback.on_checkpoint_loaded(**kwargs)

    def on_create_policy(self, **kwargs) -> None:
        """Callback run whenever a new policy is added to an algorithm."""
        for callback in self.callbacks:
            callback.on_create_policy(**kwargs)

    def on_environment_created(self, **kwargs) -> None:
        """Callback run when a new environment object has been created."""
        for callback in self.callbacks:
            callback.on_environment_created(**kwargs)

    def on_sub_environment_created(self, **kwargs) -> None:
        """Callback run when a new sub-environment has been created."""
        for callback in self.callbacks:
            callback.on_sub_environment_created(**kwargs)

    def on_episode_created(self, **kwargs) -> None:
        """Callback run when a new episode is created (but has not started yet!)."""
        for callback in self.callbacks:
            callback.on_episode_created(**kwargs)

    def on_episode_start(self, **kwargs) -> None:
        """Callback run right after an Episode has been started."""
        for callback in self.callbacks:
            callback.on_episode_start(**kwargs)

    def on_episode_step(self, **kwargs) -> None:
        """Called on each episode step (after the action(s) has/have been logged)."""
        for callback in self.callbacks:
            callback.on_episode_step(**kwargs)

    def on_episode_end(self, **kwargs) -> None:
        """Called when an episode is done (after terminated/truncated have been logged)."""
        for callback in self.callbacks:
            callback.on_episode_end(**kwargs)

    def on_evaluate_start(self, **kwargs) -> None:
        """Callback before evaluation starts."""
        for callback in self.callbacks:
            callback.on_evaluate_start(**kwargs)

    def on_evaluate_end(self, **kwargs) -> None:
        """Runs when the evaluation is done."""
        for callback in self.callbacks:
            callback.on_evaluate_end(**kwargs)

    def on_postprocess_trajectory(self, **kwargs) -> None:
        """Called immediately after a policy's postprocess_fn is called."""
        for callback in self.callbacks:
            callback.on_postprocess_trajectory(**kwargs)

    def on_sample_end(self, **kwargs) -> None:
        """Called at the end of `EnvRunner.sample()`."""
        for callback in self.callbacks:
            callback.on_sample_end(**kwargs)

    def on_learn_on_batch(self, **kwargs) -> None:
        """Called at the beginning of Policy.learn_on_batch()"""
        for callback in self.callbacks:
            callback.on_learn_on_batch(**kwargs)

    def on_train_result(self, **kwargs) -> None:
        """Called at the end of Algorithm.train()"""
        for callback in self.callbacks:
            callback.on_train_result(**kwargs)
