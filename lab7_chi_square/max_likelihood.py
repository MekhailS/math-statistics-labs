import numpy as np


class MaxLikelihood:

    @staticmethod
    def estimate_as_normal(sample):
        return np.mean(sample), np.var(sample)
