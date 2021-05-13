import numpy as np
import pandas as pd
from scipy import stats

from chi_square import ChiSquare
from max_likelihood import MaxLikelihood


PATH_TABLES = 'tables\\'


def build_df_chi_square(chi_square: ChiSquare, format_for_report=False):
    REPORT_COLUMNS_DICT = {
        'delta': r'Interval $\delta_i$',
        'n': r'$n_i$',
        'p': r'$p_i$',
        'np': r'$np_i$',
        'n-np': r'$n_i - np_i$',
        r'{n-np}^2/{np}': r'$\frac{(n_i - np_i)^2}{np_i}$'
    }
    df = pd.DataFrame()
    df['delta'] = chi_square.borders
    df['n'] = chi_square.freq
    df['p'] = chi_square.probabilities
    df['np'] = chi_square.sample_size * chi_square.probabilities
    df['n-np'] = chi_square.freq - chi_square.sample_size * chi_square.probabilities
    df[r'{n-np}^2/{np}'] = (chi_square.freq - chi_square.sample_size * chi_square.probabilities)**2 / (chi_square.sample_size * chi_square.probabilities)

    # add summary
    df = df.append(df.sum(numeric_only=True), ignore_index=True)

    if format_for_report:
        df.fillna('', inplace=True)
        df['delta'] = df['delta'].astype(str)
        df['delta'] = df['delta'].apply(lambda x: f'${x}$')

        df[r'{n-np}^2/{np}'] = df[r'{n-np}^2/{np}'].astype(str)
        df.at[len(df)-1, r'{n-np}^2/{np}'] = fr'${df.at[len(df)-1, r"{n-np}^2/{np}"]} = \chi^2_B$'
        df.index += 1

        df.rename(columns=REPORT_COLUMNS_DICT, index={len(df): r'$\sum$'}, inplace=True)

    return df


def main():
    sample_generators_dict = {
        'normal_100': lambda: stats.norm().rvs(100),
        'uniform_20': lambda: stats.uniform(loc=-np.sqrt(3), scale=2*np.sqrt(3)).rvs(20),
        'laplace_20': lambda: stats.laplace(loc=0, scale=1/np.sqrt(2)).rvs(20)
    }
    alpha = 0.05
    start, end = -1.1, 1.1
    hypothesis_normal = stats.norm().cdf

    chi_square = ChiSquare(hypothesis_normal, alpha, start, end)
    for name, generator in sample_generators_dict.items():
        sample = generator()
        print(name)
        print(f'Max-likelihood estimation: {MaxLikelihood.estimate_as_normal(sample)}')
        chi_square.fit(sample)

        print(f'passed: {chi_square.passed} ; quantile: {chi_square.quantile}')

        df_report = build_df_chi_square(chi_square, format_for_report=True)
        latex_table = df_report.to_latex(index=True, column_format='l|llllll', escape=False)
        with open(f'{PATH_TABLES}{name}.tex', 'w') as file:
            file.write(latex_table)


if __name__ == '__main__':
    main()
