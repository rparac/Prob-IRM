from itertools import groupby


class TraceTracker:
    def __init__(self) -> None:
        self.trace = []

        self.is_positive = False
        self.is_complete = False

        # self._hash_state_mapping = OrderedDict()

    def reset(self):
        self.trace.clear()
        self.is_positive = False
        self.is_complete = False

    def update(self, labels, is_positive_trace, is_complete_trace):
        self.is_positive = self.is_positive or is_positive_trace
        self.is_complete = self.is_complete or is_complete_trace
        self.trace.append(self._process_label(labels))

    def _process_label(self, labels):
        if isinstance(labels, dict):
            # noisy trace
            return labels

        return labels or []

    @property
    def labels_sequence(self):
        return tuple(tuple(es) for es in self.trace if es)

    @property
    def no_dups_labels_sequence(self):
        return tuple(i[0] for i in groupby(self.labels_sequence or tuple()))
