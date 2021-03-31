from scipy import stats
from statistics import variance
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from lab5_correlation.correlation_coefficients import CorrelationCoefficients


def equiprobability_ellipse(data_generator, size, params, path):
    _, sp = plt.subplots(1, len(params), figsize=(16, 6))
    for cor_coef, subplot in zip(params, sp):
        sample = data_generator(size, cor_coef)
        x = sample[:, 0]
        y = sample[:, 1]
        vx = np.var(x)
        vy = np.var(y)
        mean_x = np.mean(x)
        mean_y = np.mean(y)

        def ellipse_equation(x, y):
            return (x-mean_x)**2/vx - 2*cor_coef*(x-mean_x)*(y-mean_y)/(np.sqrt(vx*vy)) + (y-mean_y)**2/vy
        R = np.max(ellipse_equation(x, y))

        x_grid = np.linspace(np.min(x) - 2, np.max(x) + 2, 500)
        y_grid = np.linspace(np.min(y)-2, np.max(y) + 2, 500)
        xx, yy = np.meshgrid(x_grid, y_grid)
        z = ellipse_equation(xx, yy)

        subplot.contour(xx, yy, z, [R])

        subplot.scatter(x, y)

        title = f"n = {size} rho = {cor_coef}"

        subplot.set_title(title)
        subplot.set_xlabel("X")
        subplot.set_ylabel("Y")

    plt.savefig(f"{path}ellipse_{size}.png")


def df_correlation(data_generator, params, num_iterations, name, size):
    df_summary_list = []
    for param in params:
        data_generator_w_param = lambda size: data_generator(size, param)
        df_corr = CorrelationCoefficients.df_correlation(
            data_generator_w_param,
            size,
            num_iterations
        )
        df_summary = pd.DataFrame()
        df_summary[f'{name} n = {size}'] = [
            f'$\\rho = {param}$',
            '$E(z)$',
            '$E(z^2)$',
            '$D(z)$',
        ]
        for col in df_corr.columns:
            DIGITS_TO_KEEP = 3
            z = df_corr[col].to_numpy()

            mean = round(np.mean(z), DIGITS_TO_KEEP)
            mean_pow2 = round(np.mean(z ** 2), DIGITS_TO_KEEP)
            variance = round(mean_pow2 - mean**2, DIGITS_TO_KEEP)
            df_summary[f'${col}$'] = [
                '',
                mean,
                mean_pow2,
                variance,
            ]
        df_summary_list.append(df_summary)
    df_res = pd.concat(df_summary_list, ignore_index=True)
    return df_res


if __name__ == '__main__':

    PATH_LATEX_TABLES = 'tables\\'
    PATH_PLOTS = 'plots\\'
    # correlation coef
    DIST_PARAMETERS = [0, 0.5, 0.9]
    # samples
    SAMPLE_SIZE = [20, 60, 100]

    NUM_ITERATIONS = 1000

    dist_normal = lambda size, cor_coef: \
        stats.multivariate_normal.rvs([0, 0], [[1, cor_coef], [cor_coef, 1]], size)
    for size in SAMPLE_SIZE:
        '''
        df_corr = df_correlation(
            dist_normal,
            DIST_PARAMETERS,
            NUM_ITERATIONS,
            'normal 2d',
            size
        )
        '''
        #latex_table = df_corr.to_latex(index=False, column_format='l|lll', escape=False)
        #file = open(f'{PATH_LATEX_TABLES}{df_corr.columns[0]}.tex', 'w')
        #file.write(latex_table)
        #file.close()
        equiprobability_ellipse(dist_normal, size, DIST_PARAMETERS, PATH_PLOTS)

    dist_mixture_normal = lambda size, _: \
        0.9 * stats.multivariate_normal.rvs([0, 0], [[1, 0.9], [0.9, 1]], size) + \
        0.1 * stats.multivariate_normal.rvs([0, 0], [[10, -0.9], [-0.9, 10]], size)
    for size in SAMPLE_SIZE:
        df_corr = df_correlation(
            dist_mixture_normal,
            [''],
            NUM_ITERATIONS,
            'mixture of normal 2D',
            size
        )
        latex_table = df_corr.to_latex(index=False, column_format='l|lll', escape=False)
        file = open(f'{PATH_LATEX_TABLES}{df_corr.columns[0]}.tex', 'w')
        file.write(latex_table)
        file.close()
