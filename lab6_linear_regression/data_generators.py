import numpy as np
from copy import copy

DATA_VALUES = None


def ground_truth(x):
    x = np.asarray(x)
    return 2 + 2*x


def data_values(x):
    global DATA_VALUES
    x = np.asarray(x)
    DATA_VALUES = ground_truth(x) + np.random.randn(len(x))
    return copy(DATA_VALUES)


def data_values_disturbed(x):
    global DATA_VALUES

    if DATA_VALUES is None:
        data_values(x)

    res = copy(DATA_VALUES)
    res[0] += 10
    res[-1] -= 10
    return res
