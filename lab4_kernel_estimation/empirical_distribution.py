import numpy as np


class EmpiricalDistribution:
    def __init__(self, data):
        self.data = data

    def evaluate(self, x):

        def eval_single_elem(x):
            return np.count_nonzero(self.data < x) / self.data.shape[0]

        return np.vectorize(eval_single_elem)(x)