import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.special import factorial
from scipy import stats

from lab3_boxplot.distribution import Distribution


PATH_PLOTS = 'plots\\'
NUM_GENERATE_SAMPLE = 1000


def correct_digits(num):
    return max(0, round(-math.log10(abs(num))))


def plot_boxplot(df_data, name):
    fig, ax = plt.subplots()
    sns.boxplot(data=df_data, ax=ax)
    plt.savefig(f'{PATH_PLOTS}{name}')


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
    size_list = [20, 100]
    for dist in distributions:
        df_data = dist.df_data(size_list)
        plot_boxplot(df_data, dist.name)

    print('lab3')