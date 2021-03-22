from scipy import stats
import numpy as np
import pandas as pd


class CorrelationCoefficients:

    @staticmethod
    def pearson(sample):
        x = sample[:, 0]
        y = sample[:, 1]
        return stats.pearsonr(x, y)[0]

    @staticmethod
    def spearman(sample):
        x = sample[:, 0]
        y = sample[:, 1]
        return stats.spearmanr(x, y)[0]

    @staticmethod
    def quadrant(sample):
        x = sample[:, 0]
        y = sample[:, 1]

        x_new = x - np.median(x)
        y_new = y - np.median(y)

        n = len(x_new)
        n_1 = np.count_nonzero(np.logical_and(x_new >= 0, y_new > 0))
        n_2 = np.count_nonzero(np.logical_and(x_new < 0, y_new > 0))
        n_3 = np.count_nonzero(np.logical_and(x_new < 0, y_new <= 0))
        n_4 = np.count_nonzero(np.logical_and(x_new >= 0, y_new <= 0))

        return ((n_1 + n_3) - (n_2 + n_4)) / n

    @staticmethod
    def df_correlation(generator, size, iterations):
        df_characteristics = pd.DataFrame(
            columns=['r', 'r_s', 'r_Q'],
            index=np.arange(iterations)
        )
        for i_row in range(iterations):
            data = generator(size)

            df_characteristics.at[i_row, 'r'] = CorrelationCoefficients.pearson(data)
            df_characteristics.at[i_row, 'r_s'] = CorrelationCoefficients.spearman(data)
            df_characteristics.at[i_row, 'r_Q'] = CorrelationCoefficients.quadrant(data)

        return df_characteristics