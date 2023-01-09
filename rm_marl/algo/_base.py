import abc

class Algo:
    def __init__(self, *args, **kwargs):
        self.n_steps = 0

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        raise NotImplementedError("learn")

    @abc.abstractmethod
    def action(self, *args, **kwargs):
        raise NotImplementedError("action")