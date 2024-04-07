from itertools import groupby

from collections import OrderedDict
from typing import Dict, List, Tuple


# TODO: delete obs; it is unused
class TraceTracker:
    def __init__(self) -> None:
        self.trace = []
        # self.obs = []

        self.is_positive = False
        self.is_complete = False

        # self._hash_state_mapping = OrderedDict()

    def reset(self):
        self.trace.clear()
        # self.obs.clear()
        self.is_positive = False
        self.is_complete = False

    # TODO: remove obs; it's unused
    def update(self, labels, obs, is_positive_trace, is_complete_trace):
        self.is_positive = self.is_positive or is_positive_trace
        self.is_complete = self.is_complete or is_complete_trace
        self.trace.append(self._process_label(labels))
        # self.obs.append(self._process_obs(obs))

    def _process_label(self, labels):
        if isinstance(labels, dict):
            # noisy trace
            return labels

        return labels or []

    # def _process_obs(self, obs):
    #     state_hash = hash(str(obs))
    #     if state_hash not in self._hash_state_mapping:
    #         self._hash_state_mapping[state_hash] = obs
    #     return list(self._hash_state_mapping.keys()).index(state_hash) + 1

    @property
    def labels_sequence(self):
        return tuple(tuple(es) for es in self.trace if es)

    @property
    def no_dups_labels_sequence(self):
        return tuple(i[0] for i in groupby(self.labels_sequence or tuple()))

    # TODO: can this be deleted?
    @property
    def flatten_labels_sequence(self):
        return tuple(e for es in self.trace for e in es)

    # TODO: can this be deleted?
    @property
    def no_dups_flatten_labels_sequence(self):
        return tuple(i[0] for i in groupby(self.flatten_labels_sequence or tuple()))
