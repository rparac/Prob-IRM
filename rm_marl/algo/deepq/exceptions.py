class NotEnoughExperiencesError(Exception):

    def __init__(self, *, requested, available):
        super().__init__(
            f'A batch of size {requested} was requested, but only {available} experiences are available in the replay memory')
