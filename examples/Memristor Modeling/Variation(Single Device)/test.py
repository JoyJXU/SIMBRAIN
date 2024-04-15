import json
import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simbrain.Fitting_Functions.conductance_fitting import Conductance


def main():
    # Fit
    with open("../../../memristordata/my_memristor.json") as f:
        dict = json.load(f)
    dict.update(
        {
            'G_off': 1.1496016e-09,
            'G_on': 2.889198e-10,
            'alpha_off': 8,
            'alpha_on': 9,
        }
    )
    file = "../../../memristordata/Conductance.xlsx"

    exp = Conductance(file, dict)
    exp.k_off = 160.70528182616417
    exp.k_on = -48.241087041653735
    exp.P_off = 1.438449888287663
    exp.P_on = 0.33598182862837817
    sigma_relative, sigma_absolute = exp.c2c_fitting()

    # Output
    df = pd.DataFrame(
        {'value': [sigma_relative, sigma_absolute]},
        index=['sigma_relative', 'sigma_absolute']
    )
    print(df)
    with open("fitting_record.json", "w") as f:
        json.dump(dict, f, indent=2)

    # Plot
    z = np.array([sigma_relative ** 2 * 2 / np.pi, sigma_absolute ** 2 * 2 / np.pi])
    p = np.poly1d(z)
    plot_x = exp.x_mean
    plot_y = p(plot_x)

    fig = plt.figure(figsize=(24, 10.8))
    ax1 = fig.add_subplot(121)
    ax1.scatter(exp.memx_total, exp.variation_x, c='r')
    ax1.set_xlabel('x')
    ax1.set_ylabel('Variation')
    ax1.set_title('C2C Variation')
    ax2 = fig.add_subplot(122)
    ax2.scatter(plot_x, exp.var_x_average, c='r')
    ax2.plot(plot_x, plot_y, c='b')
    ax2.set_xlabel('x')
    ax2.set_ylabel('Variation_mean')
    ax2.set_title('C2C Variation fitting')

    plt.tight_layout()
    plt.savefig("Variation.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
