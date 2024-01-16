from itertools import groupby

from collections import OrderedDict
from typing import Dict, List, Tuple


class TraceTracker:
    def __init__(self) -> None:
        self.trace = []
        # TODO: remove obs; it's unused
        self.obs = []

        self._hash_state_mapping = OrderedDict()

    def reset(self):
        self.trace.clear()
        self.obs.clear()

    def update(self, labels, obs):
        self.trace.append(self._process_label(labels))
        self.obs.append(self._process_obs(obs))

    def _process_label(self, labels):
        # TODO check or remove that
        # assert len(labels) < 2, f"Assumption that there is only one label at a time: [{labels}]"
        return labels or []

    def _process_obs(self, obs):
        state_hash = hash(str(obs))
        if state_hash not in self._hash_state_mapping:
            self._hash_state_mapping[state_hash] = obs
        return list(self._hash_state_mapping.keys()).index(state_hash) + 1

    @property
    def labels_sequence(self):
        return tuple(tuple(es) for es in self.trace if es)

    @property
    def no_dups_labels_sequence(self):
        return tuple(i[0] for i in groupby(self.labels_sequence or tuple()))

    @property
    def flatten_labels_sequence(self):
        return tuple(e for es in self.trace for e in es)

    @property
    def no_dups_flatten_labels_sequence(self):
        return tuple(i[0] for i in groupby(self.flatten_labels_sequence or tuple()))


class NoisyTraceTracker():
    def __init__(self):
        self.trace = []
        self.penalties = []

        self.penalty_scale_factor = 10

    def reset(self):
        self.trace.clear()
        self.penalties.clear()

    def update(self, labels: Dict[str, float]):
        tr, pen = self._process_labels(labels)
        self.trace.append(tr)
        self.penalties.append(pen)

    def get_compact_trace(self):
        pass

    def _process_labels(self, labels):
        step_used = False
        curr_step_trace = []
        curr_penalty = None
        for label, prob in labels.items():
            # How certain are we in the grounding chosen
            # The higher the value the better
            cert_value = abs(prob - 0.5) * self.penalty_scale_factor + 1
            curr_penalty = min(curr_penalty or cert_value, cert_value)
            if prob >= 0.5:
                curr_step_trace.append(label)
        return curr_step_trace, curr_penalty * self.penalty_scale_factor

    def labels_sequence(self):
        ret_labels = []
        ret_penalties = []
        for es, val in zip(self.trace, self.penalties):
            if es:
                ret_labels.append(tuple(es))
                ret_penalties.append(val)
        return ret_labels, ret_penalties
