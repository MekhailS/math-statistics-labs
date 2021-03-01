import numpy as np
import pandas as pd

from lab3_boxplot.data_characteristics import num_outliers

class Distribution:
    def __init__(self, name, f_density, f_distribution, f_data_generator):
        self.name = name
        self.f_density = f_density
        self.f_distribution = f_distribution
        self.f_data_generator = f_data_generator
