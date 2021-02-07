from lab1_histogram.distribution import Distribution
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import factorial
from scipy import stats


DIRECTORY_PLOTS = 'plots\\'


def plot_distribution_histogram_with_density(distribution, size_list, bins):
    DENSITY_POINTS_IN_BIN = 10

    fig_num = len(size_list)
    fig, ax = plt.subplots(1, fig_num, figsize=(10, 5))
    plt.subplots_adjust(wspace=0.4)

    plt.suptitle(f'{distribution.name} distribution')
    for i, size in enumerate(size_list):
        data = distribution.f_data_generator(size)
        density_x = np.linspace(data.min(), data.max(), num=bins*DENSITY_POINTS_IN_BIN, endpoint=True)
        density_y = np.vectorize(distribution.f_density)(density_x)

        ax[i].hist(data, density=True, histtype='bar', color='grey',
                   edgecolor='black', alpha=0.3, bins=bins)
        ax[i].plot(density_x, density_y, color='black', linewidth=1)
        ax[i].set_title(f'n = {size}')
        ax[i].set_ylabel("density")

    plt.savefig(f'{DIRECTORY_PLOTS}{distribution.name}')


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
        plot_distribution_histogram_with_density(dist, size_list, 20)

    print('lab1')