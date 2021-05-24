from labsignal import LabSignal
import matplotlib.pyplot as plt


def get_signal(string: str, size: int, index: int):
    signals = string.replace(',', '').replace('[', '').replace(']', '')
    signal_data = signals.split(' ')
    return [float(x) for x in signal_data[size * index: size * (index + 1)]]


def read_signal(filename: str, size: int):
    with open(filename, 'r') as fin:
        line = fin.readline()
        return get_signal(line, size, 10)


def main(argv: []):
    plt.style.use("seaborn")
    if len(argv) == 2:
        signal_data = read_signal(argv[0], int(argv[1]))

        signal = LabSignal(signal_data)

        signal.plot_signal_data()
        signal.plot_signal_hist()

        signal.apply_median_filter()

        signal.plot_filtered_signal_data()

        signal.define_sections()
        signal.plot_data_signal_with_regions()

        signal.fisher_rule()
    return


if __name__ == '__main__':
    main(['data\\wave_ampl.txt', 1024])