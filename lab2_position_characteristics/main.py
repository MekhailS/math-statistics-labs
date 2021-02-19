import numpy as np
import pandas as pd
import math
from scipy.special import factorial
from scipy import stats

from lab2_position_characteristics.data_position_characteristics import DataPositionCharacteristics
from lab2_position_characteristics.distribution import Distribution


PATH_LATEX_TABLES = 'tables\\'
NUM_GENERATE_SAMPLE = 1000


def correct_digits(num):
    return max(0, round(-math.log10(abs(num))))


def df_distribution_characteristics(distribution, data_size_list):
    df_summary_list = []
    for data_size in data_size_list:
        df_characteristics = distribution.pos_characteristics(data_size, NUM_GENERATE_SAMPLE)
        df_summary = pd.DataFrame()
        df_summary[f'{distribution.name} distribution'] = [
            f'$n = {data_size}$',
            '$E(z)$',
            '$D(z)$'
        ]
        for col in df_characteristics.columns:
            z = df_characteristics[col].to_numpy()
            df_summary[col] = [
                '',
                round(DataPositionCharacteristics.mean(z),
                      correct_digits(DataPositionCharacteristics.variance(z))),
                DataPositionCharacteristics.variance(z)
            ]
        df_summary_list.append(df_summary)
    df_res = pd.concat(df_summary_list, ignore_index=True)
    return df_res



if __name__ == '__main__':

    normal = Distribution(
        'normal',
        lambda x: stats.norm.pdf(x),
        lambda size: stats.norm.rvs(size=size)
    )
    cauchy = Distribution(
        'cauchy',
        lambda x: stats.cauchy.pdf(x),
        lambda size: stats.cauchy.rvs(size=size)
    )
    laplace = Distribution(
        'laplace',
        lambda x: stats.laplace.pdf(x, scale=1/math.sqrt(2)),
        lambda size: stats.laplace.rvs(size=size, scale=1/math.sqrt(2))
    )
    poisson = Distribution(
        'poisson',
        lambda x: np.exp(-10)*np.power(10, x)/factorial(x),
        lambda size: stats.poisson.rvs(size=size, mu=10)
    )
    uniform = Distribution(
        'uniform',
        lambda x: stats.uniform.pdf(x, loc=-math.sqrt(3), scale=2*math.sqrt(3)),
        lambda size: stats.uniform.rvs(size=size, loc=-math.sqrt(3), scale=2*math.sqrt(3))
    )

    distributions = [normal, cauchy, laplace, poisson, uniform]
    size_list = [10, 50, 1000]
    for dist in distributions:
        df_res = df_distribution_characteristics(dist, size_list)

        df_res.columns = [f'{dist.name} distribution', '\overline{x}', 'med\:x', '$z_R$', '$z_Q$', '$z_{tr}$']
        latex_table = df_res.to_latex(index=False, column_format='l|lllll', escape=False)

        file = open(f'{PATH_LATEX_TABLES}{dist.name}.tex', 'w')
        file.write(latex_table)
        file.close()

    print('lab2')