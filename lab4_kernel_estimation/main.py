import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.special import factorial
from scipy import stats

from lab4_kernel_estimation.distribution import Distribution
from lab4_kernel_estimation.kernel_density_estimation import KernelDensityEstimation
from lab4_kernel_estimation.empirical_distribution import EmpiricalDistribution

RANDOM_SEED = 166
PATH_PLOTS = 'plots\\'

X_LIM_CONTINUOUS = [-4, 4]
X_LIM_POISSON = [6, 14]
N_POINTS = 1000


def plot_kernel_density_estimation_with_density(distribution, size_list, bandwidth_list, x_lim):

    plt.suptitle(f'{distribution.name} distribution')
    for size in size_list:

        fig_num = len(size_list)
        fig, ax = plt.subplots(1, fig_num, figsize=(12, 6))
        plt.subplots_adjust(wspace=0.4)

        for i, bandwidth in enumerate(bandwidth_list):
            data = distribution.f_data_generator(size)

            density_x = np.linspace(x_lim[0], x_lim[1], N_POINTS, endpoint=True)
            density_y = np.vectorize(distribution.f_density)(density_x)

            # kde = stats.gaussian_kde(data, bw_method='silverman')
            # kde.set_bandwidth(kde.factor * bandwidth)

            kde = KernelDensityEstimation(data, bandwidth)
            kde_y = kde.evaluate(density_x)

            ax[i].plot(density_x, density_y, linewidth=1, label='density function')
            ax[i].plot(density_x, kde_y, linewidth=1, label='kernel density \n estimation')

            ax[i].set_ylim([0, 1])

            ax[i].legend(fontsize='small')
            ax[i].set_title(f'{distribution.name} n = {size}, h = h_n*{bandwidth}')
            ax[i].set_xlabel('numbers')
            ax[i].set_ylabel('density')

        plt.savefig(f'{PATH_PLOTS}{distribution.name}_density_n={size}')
        plt.close(fig)


def plot_empirical_distribution(distribution, size_list, x_lim):
    fig_num = len(size_list)
    fig, ax = plt.subplots(1, fig_num, figsize=(12, 6))
    plt.subplots_adjust(wspace=0.4)

    for i, size in enumerate(size_list):
        data = distribution.f_data_generator(size)

        x = np.linspace(x_lim[0], x_lim[1], N_POINTS, endpoint=True)
        dist_y = distribution.f_distribution(x)

        emp_dist = EmpiricalDistribution(data)
        dist_emp_y = emp_dist.evaluate(x)
        ax[i].plot(x, dist_y, linewidth=1, label='distribution function')
        ax[i].plot(x, dist_emp_y, linewidth=1, label='empirical distribution')

        ax[i].legend(loc='upper left', fontsize='small')
        ax[i].set_title(f'{distribution.name} n = {size}')
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('F(x)')

    plt.savefig(f'{PATH_PLOTS}{distribution.name}_emp_dist')
    plt.close(fig)



if __name__ == '__main__':
    # np.random.seed(RANDOM_SEED)

    stats_norm = stats.norm()
    normal = Distribution(
        'normal',
        lambda x: stats_norm.pdf(x),
        lambda x: stats_norm.cdf(x),
        lambda size: stats_norm.rvs(size=size)
    )
    stats_cauchy = stats.cauchy()
    cauchy = Distribution(
        'cauchy',
        lambda x: stats_cauchy.pdf(x),
        lambda x: stats_cauchy.cdf(x),
        lambda size: stats_cauchy.rvs(size=size)
    )
    stats_laplace = stats.laplace(scale=1/math.sqrt(2))
    laplace = Distribution(
        'laplace',
        lambda x: stats_laplace.pdf(x),
        lambda x: stats_laplace.cdf(x),
        lambda size: stats_laplace.rvs(size=size)
    )
    stats_poisson = stats.poisson(mu=10)
    poisson = Distribution(
        'poisson',
        lambda x: np.exp(-10)*np.power(10, x)/factorial(x),
        lambda x: stats_poisson.cdf(x),
        lambda size: stats_poisson.rvs(size=size)
    )
    stats_uniform = stats.uniform(loc=-math.sqrt(3), scale=2*math.sqrt(3))
    uniform = Distribution(
        'uniform',
        lambda x: stats_uniform.pdf(x),
        lambda x: stats_uniform.cdf(x),
        lambda size: stats_uniform.rvs(size=size)
    )

    distributions_cont = [normal, cauchy, laplace, uniform]
    size_list = [20, 60, 100]
    bandwidth_factor_list = [0.5, 1.0, 2.0]
    for dist in distributions_cont:
        plot_kernel_density_estimation_with_density(
            distribution=dist,
            size_list=size_list,
            bandwidth_list=bandwidth_factor_list,
            x_lim=X_LIM_CONTINUOUS
        )
        plot_empirical_distribution(
            distribution=dist,
            size_list=size_list,
            x_lim=X_LIM_CONTINUOUS
        )

    plot_kernel_density_estimation_with_density(
        distribution=poisson,
        size_list=size_list,
        bandwidth_list=bandwidth_factor_list,
        x_lim=X_LIM_POISSON
    )
    plot_empirical_distribution(
        distribution=poisson,
        size_list=size_list,
        x_lim=X_LIM_POISSON
    )

    print('lab4')
