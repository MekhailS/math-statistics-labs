import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.special import factorial
from scipy import stats

from lab3_boxplot.distribution import Distribution


PATH_PLOTS = 'plots\\'
PATH_TABLES = 'tables\\'
NUM_GENERATE_SAMPLE = 1000


def plot_boxplot(df_data, name):
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    sns.boxplot(data=df_data, orient='h', palette="Blues", ax=ax)
    ax.set_xlabel('values')
    ax.set_title(f'{name} distribution')

    plt.savefig(f'{PATH_PLOTS}{name}')


def df_outliners(distribution, data_size_list):
    df_summary_list = []
    for data_size in data_size_list:
        df_outliners = distribution.count_outliers(data_size, NUM_GENERATE_SAMPLE)
        df_summary = pd.DataFrame()
        df_summary[f'{distribution.name} distribution'] = [
            f'$n = {data_size}$',
        ]
        for col in df_outliners.columns:
            df_summary[col] = [
                df_outliners[col].mean()
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
    size_list = [20, 100]
    for dist in distributions:
        df_data = dist.df_data(size_list)
        plot_boxplot(df_data, dist.name)

        outliers = df_outliners(dist, size_list)
        latex_table = outliers.to_latex(index=False, column_format='l|l', escape=False)

        file = open(f'{PATH_TABLES}{dist.name}.tex', 'w')
        file.write(latex_table)
        file.close()

    print('lab3')