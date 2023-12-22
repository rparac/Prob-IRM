import random
import copy


from ..wrappers import LabelTampering


class RandomBlindingNoise(LabelTampering):

    def __init__(self, env, noise_quantity, *, seed=None):

        super().__init__(env)

        assert 0.0 < noise_quantity < 1.0, "Noise quantity must be in range [0, 1]"
        self._noise_quantity = noise_quantity
        self._seed = seed
        self._random = random.Random(self._seed)

    def _is_tamperable(self, events):

        return len(events) > 0

    def _tamper_events(self, events):

        # Determine if this labelling function output will be tampered
        if self._random.random() >= self._noise_quantity:
            return events, False

        tampered_events = copy.copy(events)

        # Chose a random position in the event string to tamper
        target_index = self._random.randint(0, len(events))

        # Remove the event string as a whole (compound tampering)
        if target_index == len(events):
            tampered_events.clear()
        # Remove the event at the specified index
        else:
            del tampered_events[target_index]

        return tampered_events, True
