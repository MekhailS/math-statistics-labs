import numpy as np


class KernelDensityEstimation:
    def __init__(self, data, bandwidth_factor=1.0):
        self.data = np.array(data)
        self.n = len(self.data)
        std = np.std(self.data)
        self.h_n = 1.06 * std * np.power(float(self.n), -1/5) * bandwidth_factor

    def evaluate(self, x):
        x = np.array(x)

        def eval_single_elem(x):
            def Kernel(u):
                return 1/np.sqrt(2*np.pi) * np.exp(-(u**2)/2)

            kernel_vec = np.vectorize(Kernel)
            kernel_input = (x - self.data) / self.h_n
            kernel_res = kernel_vec(kernel_input)

            return np.sum(kernel_res) / (self.n * self.h_n)

        return np.vectorize(eval_single_elem)(x)