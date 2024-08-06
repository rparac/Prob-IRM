from collections import namedtuple
from enum import Enum

# labels - labeling function output
DQRNStep = namedtuple('DQRNStep', [
    'state',
    'labels',
    'action',
    'reward',
    'done',
    'new_state',
    'next_labels'
])


class EpsilonAnnealingTimescale(Enum):
    STEPS = 0
    EPISODES = 1
