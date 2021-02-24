import numpy as np
import pandas as pd

from lab3_boxplot.data_characteristics import num_outliners

class Distribution:
    def __init__(self, name, f_density, f_data_generator):
        self.name = name
        self.f_density = f_density
        self.f_data_generator = f_data_generator

    def df_data(self, data_size_list):
        df_data = []
        for size in data_size_list:
            df_data.append(pd.DataFrame(self.f_data_generator(size), columns=[f'n = {size}']))

        return pd.concat(df_data, ignore_index=False, axis=1)

    def count_outliners(self, data_size, num_samples_generate):
        df = pd.DataFrame(columns=['num_outliners'])
        for i_row in range(num_samples_generate):
            data = self.f_data_generator(data_size)
            df.loc[i_row] = num_outliners(data)/data_size

        return df