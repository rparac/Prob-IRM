"""
Credits to: https://github.com/ertsiger/hrm-learning/blob/main/src/utils/math_utils.py#L5o
"""

import numpy as np
import random


def randargmax(input_vector):
    return random.choice(np.flatnonzero(input_vector == np.max(input_vector)))
