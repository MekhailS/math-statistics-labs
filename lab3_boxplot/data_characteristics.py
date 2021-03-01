import numpy as np
import pandas as pd


def num_outliers(data):
    data = np.array(data)
    Q = np.quantile(data, [1/4, 3/4])
    Q1, Q3 = Q[0], Q[1]

    whisper_start = Q1 - 1.5 * (Q3 - Q1)
    whisper_end = Q3 + 1.5 * (Q3 - Q1)
    return np.count_nonzero(np.logical_or(data < whisper_start, data > whisper_end))
