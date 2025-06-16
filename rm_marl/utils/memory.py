import random
from collections import namedtuple, deque

class FixedSizeDict:
    def __init__(self, max_size):
        self.max_size = max_size
        self.data = deque()  # Stores keys in order
        self.store = {}      # Maps keys to values

    def __setitem__(self, key, value):
        try:
            if key in self.store:
                # Remove key from deque to update its order
                self.data.remove(key)
            elif len(self.data) >= self.max_size:
                # Remove the oldest key-value pair
                oldest_key = self.data.popleft()
                del self.store[oldest_key]
            # Add the new key-value pair
            self.data.append(key)
            self.store[key] = value
        except TypeError:
            raise RuntimeError(self.data, self.max_size, type(self.data), type(self.max_size))

    def get(self, key, default=None):
        if default is not None:
            return self.store.get(key, default)
        return self.store.get(key)

    def __getitem__(self, key, default=None):
        return self.get(key, default)

    def items(self):
        return self.store.items()

    def values(self):
        return self.store.values()

    def keys(self):
        return self.store.keys()

    def __len__(self):
        return len(self.store)

    def __repr__(self):
        return repr({key: self.store[key] for key in self.data})