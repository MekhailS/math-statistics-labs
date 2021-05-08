import numpy as np
import matplotlib.pyplot as plt

import data_generators
from linear_regressor import LinearRegressor


PATH_PLOTS = 'plots\\'


def main():
    start, end, num_points = -1.8, 2.0, 20

    x = np.linspace(start, end, num_points)
    y_truth = data_generators.ground_truth(x)
    y_generators_dict = {
        "not disturbed": data_generators.data_values,
        "disturbed": data_generators.data_values_disturbed
    }
    model = LinearRegressor()

    for name, generator in y_generators_dict.items():
        y_target = generator(x)

        b_0_l2, b_1_l2 = model.fit(x, y_target, 'l2')
        y_pred_l2 = model.eval(x)

        b_0_l1, b_1_l1 = model.fit(x, y_target, 'l1')
        y_pred_l1 = model.eval(x)

        print(name)
        print(f"b_0_l2 = {b_0_l2} ; b_1_l2 = {b_1_l2} \n "
              f"b_0_l1 = {b_1_l1} ; b_1_l1 = {b_1_l1}")

        plt.plot(x, y_target, 'ko', mfc='none')
        plt.plot(x, y_truth, )
        plt.plot(x, y_pred_l2,)
        plt.plot(x, y_pred_l1,)
        plt.legend(('Выборка', 'Модель', 'МНК', 'МНМ'))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.savefig(f"{PATH_PLOTS}{name}.png")
        plt.show()

    pass


if __name__ == '__main__':
    main()
