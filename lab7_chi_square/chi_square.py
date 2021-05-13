import numpy as np
from scipy import stats


class ChiSquare:

    def __init__(self, hypothesis, alpha, start, end):
        self.hypothesis = hypothesis

        self.alpha = alpha
        self.start = start
        self.end = end

        self.__k = None
        self.chi_square = None
        self.quantile = None

        self.passed = None

        self.probabilities = None
        self.freq = None

        self.borders = None
        self.sample_size = None

    def fit(self, sample):
        self.sample_size = len(sample)
        sample = np.asarray(sample)

        self.__k = ChiSquare.__calc_k(sample)
        self.quantile = stats.chi2.ppf(1 - self.alpha, self.__k)

        borders = np.linspace(self.start, self.end, num=self.__k - 1)
        self.__find_probabilities(sample, borders)

        self.chi_square = np.sum((self.freq - len(sample) * self.probabilities)**2 / (self.probabilities * len(sample)))
        self.passed = self.chi_square < self.quantile
        
        self.borders = list(zip([r'-\infty'] + list(borders), list(borders) + [r'\infty']))
        return self.passed, self.chi_square, self.quantile, self.borders, self.probabilities, self.freq

    def __find_probabilities(self, sample, borders):
        p_list = [self.hypothesis(self.start)]
        n_list = [np.count_nonzero(sample < self.start)]

        for i in range(self.__k - 2):
            p_i = self.hypothesis(borders[i + 1]) - self.hypothesis(borders[i])
            p_list.append(p_i)

            n_i = np.count_nonzero(np.logical_and(sample < borders[i + 1], sample >= borders[i]))
            n_list.append(n_i)

        p_list.append(1-self.hypothesis(self.end))
        n_list.append(np.count_nonzero(sample >= self.end))

        self.probabilities = np.asarray(p_list)
        self.freq = np.asarray(n_list)

    @staticmethod
    def __calc_k(sample):
        return int(np.floor(1.72 * np.power(len(sample), 1/3)))