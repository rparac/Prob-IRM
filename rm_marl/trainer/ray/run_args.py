from dataclasses import dataclass


@dataclass
class Args:
    experiment: str = ""

    hyperparam: bool = True
    resume: bool = False

    rm: bool = False
    rm_learning: bool = False
    rm_learning_freq: int = 1
    shared_layers: tuple = (0, 1)
    shared_policy: bool = False
    reward_shaping: bool = False

    seed: int = 123
    num_iterations: int = 400
    capture_video: bool = False
    debug: bool = False
    num_workers: int = 10
    max_concurrent_trials: int | None = None

    def __post_init__(self):
        if not self.hyperparam:
            assert self.experiment, "You must select an experiment to load"
            assert not self.resume, "You cannot resume"
        # elif self.experiment:
        #     assert self.resume, "Why do you have an experiment that you don't want to resume?"

        assert not self.shared_layers or (
                self.shared_layers and not self.shared_policy
        ), "cannot have shared layers and shared policy simultaneously"

        # if self.rm_learning:
        #     self.max_concurrent_trials = 1
