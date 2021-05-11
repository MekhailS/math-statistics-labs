import numpy as np
import pandas as pd

from chi_square import ChiSquare
from max_likelihood import MaxLikelihood


def build_df_chi_square(chi_square: ChiSquare, format_for_report=False):
    REPORT_COLUMNS_DICT = {
        'delta': r'Границы $\delta_i$',
        'n': r'$n_i$',
        'p': r'$p_i$',
        'np': r'$np_i$',
        'n-np': r'$n_i - np_i$',
        '{n-np}^2/{np}': r'$\frac{(n_i - np_i)^2}{np_i}$'
    }
    df = pd.DataFrame()
    df['delta'] = chi_square.borders
    df['n'] = chi_square.freq
    df['p'] = chi_square.probabilities
    df['np'] = chi_square.freq * chi_square.probabilities
    df['n-np'] = chi_square.freq - chi_square.freq * chi_square.probabilities
    df['{n-np}^2/{np}'] = (chi_square.freq - chi_square.freq * chi_square.probabilities)**2 / chi_square.freq * chi_square.probabilities

    df = df.append(df.sum(numeric_only=True), ignore_index=True)

    df.iloc[-1]['np'] = pd.NA
    df.iloc[-1]['n-np'] = pd.NA

    if format_for_report:
        df.fillna('—', inplace=True)

        df.index += 1
        df.iloc[-1]['{n-np}^2/{np}'] = fr'${df.iloc[-1]["{n-np}^2/{np}"]} = chi^2_B$'

        df.rename(columns=REPORT_COLUMNS_DICT, index={len(df): '$\sum$'}, inplace=True)

    return df


