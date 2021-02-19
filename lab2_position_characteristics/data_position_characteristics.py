import numpy as np


class DataPositionCharacteristics:
    @staticmethod
    def mean(data):
        return data.sum() / len(data)

    @staticmethod
    def median(data):
        data_size = len(data)
        median = data[(data_size + 1) // 2] if data_size % 2 == 1 \
            else (data[data_size // 2] + data[data_size // 2 + 1]) / 2
        return median

    @staticmethod
    def half_sum(data):
        return (data[0] + data[-1]) // 2

    @staticmethod
    def quartile_half_sum(data):
        return (np.quantile(data, 1 / 4) + np.quantile(data, 3 / 4)) / 2

    @staticmethod
    def truncated_mean(data):
        data_size = len(data)
        r = data_size // 4
        return data[r: -r].sum() / (data_size - 2 * r)

    @staticmethod
    def variance(data):
        data = np.array(data)
        mean = DataPositionCharacteristics.mean(data)
        return DataPositionCharacteristics.mean(data * data) - mean * mean