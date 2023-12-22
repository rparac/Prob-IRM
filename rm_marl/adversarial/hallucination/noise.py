import random
import copy


from ..wrappers import LabelTampering


class RandomHallucinationNoise(LabelTampering):
    """
    Labeling function tamperer that randomly alters abstract observation with a given probability.

    This tamperer adopts the following noise model:

     - every labelling function output might be subject to noise;
     - every labelling function output tampering might consist of the removal (blinding) or substitution of any event
       in the original event string
    """

    def __init__(self, env, noise_quantity, *, seed=None):

        super().__init__(env)

        assert 0.0 < noise_quantity < 1.0, "Noise quantity must be in range [0, 1]"
        self._noise_quantity = noise_quantity
        self._seed = seed
        self._random = random.Random(self._seed)

    def _is_tamperable(self, events):

        return len(events) > 0

    def _tamper_events(self, events):

        if self._random.random() >= self._noise_quantity:
            return events, False

        tampered_events = copy.copy(events)

        # Chose a random position in the event string to tamper
        target_index = self._random.randint(0, len(events) - 1)

        # Chose a random substitute event
        substitute = self._random.choice(self._all_events)

        # If the chosen substitute is already present in the true events, simply remove the original one
        if substitute in events:
            del tampered_events[target_index]

        # If not, carry out the substitution
        else:
            tampered_events[target_index] = substitute

        return tampered_events, True


