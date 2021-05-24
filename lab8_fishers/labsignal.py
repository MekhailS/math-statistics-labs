import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
from scipy.ndimage import median_filter
from copy import deepcopy

class SectionType(Enum):
    background = (0, 'y')
    transition = (1, 'g')
    signal = (2, 'r')

    def __init__(self, id, title):
        self.id = id
        self.title = title


def define_color(section_type: SectionType):
    if section_type.id == 0:
        return 'фон'
    elif section_type.id == 1:
        return 'переход'
    else:
        return 'сигнал'


class Section:
    def __init__(self, left: int, right: int, section_type: SectionType):
        self.left = left
        self.right = right
        self.section_type = section_type

    def get_edges(self):
        return self.left, self.right

    def get_linspace(self):
        return np.linspace(self.left, self.right, self.right - self.left + 1)

    def get_section_type(self):
        return self.section_type


class LabSignal:
    def __init__(self, signal_data: []):
        self.signal_data = signal_data
        self.size = len(signal_data)
        self.filtered_signal = []
        self.hist_data = []
        self.sections = []

    def _plot(self, signal_data):
        x = np.linspace(0, self.size - 1, self.size)
        plt.plot(x, signal_data)

    def plot_signal_data(self):
        self._plot(self.signal_data)
        plt.savefig('images/SignalData.png')
        plt.show()

    def plot_filtered_signal_data(self):
        assert self.filtered_signal
        self._plot(self.filtered_signal)
        plt.savefig('images/Filtered.png')
        plt.show()

    def plot_signal_hist(self):
        number_of_intervals = int(1.72 * (self.size ** (1 / 3)))
        self.hist_data = plt.hist(self.signal_data, bins=number_of_intervals)
        plt.savefig('images/Histogram.png')
        plt.show()

    def apply_median_filter(self):
        self.filtered_signal = list(median_filter(deepcopy(self.signal_data), 5))

    def define_sections(self):
        assert self.hist_data
        bins = [int(x) for x in self.hist_data[0]]

        sections = [float(x) for x in self.hist_data[1]]
        background_section_index = bins.index(max(bins))
        signal_section_index = bins.index(max(bins[:background_section_index] + bins[background_section_index + 1:]))

        signal_values = [sections[signal_section_index], sections[signal_section_index + 1]]
        background_values = [sections[background_section_index], sections[background_section_index + 1]]

        self.sections = self._define_sections_types(signal_values, background_values)

    def _define_sections_types(self, signal_values: [], backgrounds_values: []):
        assert self.filtered_signal
        assert len(signal_values) == len(backgrounds_values) == 2

        signal_types = [SectionType.transition for _ in range(self.size)]

        for i in range(self.size):
            if signal_values[0] <= self.filtered_signal[i] <= signal_values[1]:
                signal_types[i] = SectionType.signal
            if backgrounds_values[0] <= self.filtered_signal[i] <= backgrounds_values[1]:
                signal_types[i] = SectionType.background

        sections = []
        cur_type = signal_types[0]
        last_edge = 0
        for i in range(self.size):
            if cur_type.id != signal_types[i].id:
                sections.append(Section(last_edge, i - 1, cur_type))
                last_edge = i
                cur_type = signal_types[i]
        sections.append(Section(last_edge, self.size - 1, cur_type))

        return sections

    def plot_data_signal_with_regions(self):
        assert self.sections
        assert self.filtered_signal
        for section in self.sections:
            left, right = section.get_edges()
            section_type = section.get_section_type()
            plt.plot(section.get_linspace(), self.filtered_signal[left:right + 1],
                     color=section_type.title, label=define_color(section_type))
        plt.legend()
        plt.savefig('images/Regions.png')
        plt.show()

    K_LIST = [7, 4, 4, 4, 7]
    idx_cur_k = 0

    @staticmethod
    def number_of_splitting(size: int):
        """
        k = s = size
        while k > 8:
            k = 4
            while size % k != 0:
                k += 1
                if k > 10:
                    break
            size -= 1
        return k, size if s == size else size + 1
        """
        res = LabSignal.K_LIST[LabSignal.idx_cur_k]
        LabSignal.idx_cur_k = (LabSignal.idx_cur_k + 1) % len(LabSignal.K_LIST)
        return res, size

    @staticmethod
    def _intar_group(sample: [], k: int):
        size = len(sample)
        delta = int(size / k)

        inter = 0

        left = 0
        for i in range(0, k):
            sub_sample = sample[left:left + delta]
            inter += LabSignal.variance(sub_sample) * (len(sub_sample) - 1) / (k - 1)
            left += delta

        return inter / k

    @staticmethod
    def _inter_group(sample: [], k: int):
        size = len(sample)
        delta = int(size / k)
        means = [0.0 for _ in range(k)]
        left = 0
        for i in range(0, k):
            means[i] = (LabSignal.mean(sample[left:left + delta]))
            left += delta

        return LabSignal.variance(means) * k

    def _fisher(self, left, right):
        size = right - left + 1
        k, size = LabSignal.number_of_splitting(size)
        print(k)
        sample = self.filtered_signal[left:left + size]
        intar = LabSignal._intar_group(sample, k)
        inter = LabSignal._inter_group(sample, k)
        return inter / intar

    def fisher_rule(self):
        assert self.sections
        assert self.filtered_signal
        fisher_coefficients = []
        for section in self.sections:
            left, right = section.get_edges()
            fisher_coefficients.append(self._fisher(left, right))
        print(fisher_coefficients)
        return

    @staticmethod
    def mean(sample):
        s = 0
        for elem in sample:
            s += elem
        return s / len(sample)

    @staticmethod
    def variance(sample):
        mean = LabSignal.mean(sample)
        s = 0
        for elem in sample:
            s += (elem - mean) ** 2
        return s / (len(sample) - 1)