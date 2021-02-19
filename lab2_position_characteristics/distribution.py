import numpy as np
import pandas as pd

from lab2_position_characteristics.data_position_characteristics import DataPositionCharacteristics

class Distribution:
    def __init__(self, name, f_density, f_data_generator):
        self.name = name
        self.f_density = f_density
        self.f_data_generator = f_data_generator

    def pos_characteristics(self, data_size, num_samples_generate):
        df_characteristics = pd.DataFrame(
            columns=['mean x', 'med x', 'z_R', 'z_Q', 'z_tr'],
            index=np.arange(num_samples_generate)
        )
        for i_row in range(num_samples_generate):
            data = np.sort(np.array(self.f_data_generator(data_size)))

            df_characteristics.at[i_row, 'mean x'] = DataPositionCharacteristics.mean(data)
            df_characteristics.at[i_row, 'med x'] = DataPositionCharacteristics.median(data)
            df_characteristics.at[i_row, 'z_R'] = DataPositionCharacteristics.half_sum(data)
            df_characteristics.at[i_row, 'z_Q'] = DataPositionCharacteristics.quartile_half_sum(data)
            df_characteristics.at[i_row, 'z_tr'] = DataPositionCharacteristics.truncated_mean(data)

        return df_characteristics
