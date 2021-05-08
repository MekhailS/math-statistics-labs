import numpy as np


class LinearRegressor:

    LOSSES = {'l1', 'l2'}

    def __init__(self):
        self.b_0, self.b_1 = None, None

    def fit(self, x, y, loss='l2'):
        if loss not in LinearRegressor.LOSSES:
            return

        x, y = np.asarray(x), np.asarray(y)
        if loss == 'l2':
            self.__fit_l2(x, y)
        elif loss == 'l1':
            self.__fit_l1(x, y)

        return self.b_0, self.b_1

    def eval(self, x):
        x = np.asarray(x)
        return self.b_0 + self.b_1 * x

    def __fit_l2(self, x, y):
        x_mean = np.mean(x)
        x_square_mean = np.mean(x**2)
        xy_mean = np.mean(x * y)
        y_mean = np.mean(y)

        self.b_1 = (xy_mean - x_mean * y_mean) / (x_square_mean - x_mean ** 2)
        self.b_0 = y_mean - x_mean * self.b_1

    def __fit_l1(self, x, y):
        n = len(x)

        r_Q = np.mean(np.sign(x - np.median(x)) * np.sign(y - np.median(y)))

        l_index = n // 4 + 1 if n % 4 != 0 else n // 4
        j_index = n - l_index + 1

        q_y = (y[j_index] - y[l_index])
        q_x = (x[j_index] - x[l_index])

        k_Q = x[j_index] - x[l_index]

        self.b_1 = r_Q * q_y / q_x
        self.b_0 = np.median(y) - self.b_1 * np.median(x)