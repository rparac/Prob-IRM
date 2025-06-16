from ray.rllib.algorithms.callbacks import DefaultCallbacks

class CrashAfterNIters(DefaultCallbacks):
    """
    Crash after N iters. Resetting the iter number afterwards. 
    It is hard to extend training with checkpointing. So we do this instead, crashing the training.
    """

    def __init__(self, crash_iter: int = 0, **kwargs):
        super().__init__()

        self._crash_iter = crash_iter
        self._iters_done = 0

        # We have to delay crashing by one iteration just so the checkpoint still
        # gets created by Tune after(!) we have reached the trigger avg. return.
        self._should_crash = False

    def on_train_result(self, *, algorithm, metrics_logger, result, **kwargs):
        # We had already reached the mean-return to crash, the last checkpoint written
        # (the one from the previous iteration) should yield that exact avg. return.

        self._iters_done += 1
        if self._iters_done > self._crash_iter:
            self._iters_done = 0
            raise RuntimeError("Intended crash after reaching trigger return.")