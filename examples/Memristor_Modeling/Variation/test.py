import json
import sys
sys.path.append('../../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from simbrain.Fitting_Functions.conductance_fitting import Conductance
from simbrain.Fitting_Functions.variation_fitting import Variation


def main():
    # Fit
    with open("../../../memristordata/my_memristor.json") as f:
        dict = json.load(f)
    dict.update(
        {
            'v_off': 2,
            'v_on': -2,
            'G_off': None,
            'G_on': None,
            'alpha_off': None,
            'alpha_on': None,
            'k_off': None,
            'k_on': None,
            'P_off': None,
            'P_on': None,
            'delta_t': 30 * 1e-3,
        }
    )
    file = "../../../memristordata/conductance.xlsx"

    _, Goff_sigma, _, Gon_sigma = Variation(
        file,
        dict
    ).d2d_G_fitting()
    dict.update(
        {
            "Goff_sigma": Goff_sigma,
            "Gon_sigma": Gon_sigma,
        }
    )

    data = pd.DataFrame(pd.read_excel(
        file,
        sheet_name=0,
        header=None,
        index_col=None,
    ))
    data.columns = ['Pulse Voltage(V)', 'Read Voltage(V)', 'Current(A)'] + list(data.columns[3:])

    conductance = np.array(data['Current(A)']) / np.array(data['Read Voltage(V)'][0])
    conductance_r = conductance[:np.sum(np.array(data['Pulse Voltage(V)']) > 0)]
    conductance_d = conductance[np.sum(np.array(data['Pulse Voltage(V)']) > 0):]
    G_off = np.average(conductance_r[conductance_r.shape[0] - 10:])
    G_on = np.average(conductance_d[conductance_d.shape[0] - 10:])
    dict.update(
        {
            'G_off': G_off,
            'G_on': G_on
        }
    )
    alpha_off, alpha_on = 5, 5
    dict.update(
        {
            "alpha_off": alpha_off,
            "alpha_on": alpha_on
        }
    )
    P_off, P_on, k_off, k_on, _ = Conductance(file, dict).fitting()
    dict.update(
        {
            'k_off': k_off,
            'k_on': k_on,
            'P_off': P_off,
            'P_on': P_on,
            'G_off': None,
            'G_on': None
        }
    )

    exp = Variation(
        "../../../memristordata/conductance.xlsx",
        dict
    )
    _, Poff_sigma, _, Pon_sigma = exp.d2d_P_fitting()
    dict.update(
        {
            "Poff_sigma": Pon_sigma,
            "Pon_sigma": Poff_sigma
        }
    )
    print(exp.P_off, exp.P_on)
    sigma_relative, sigma_absolute = exp.c2c_fitting()
    dict.update(
        {
            "sigma_relative": sigma_relative,
            "sigma_absolute": sigma_absolute
        }
    )
    # Output
    df = pd.DataFrame(
        {'value': [Goff_sigma, Gon_sigma, Poff_sigma, Pon_sigma, sigma_relative, sigma_absolute]},
        index=['Goff_sigma', 'Gon_sigma', 'Poff_sigma', 'Pon_sigma', 'sigma_relative', 'sigma_absolute']
    )
    print(df)
    with open("fitting_record.json", "w") as f:
        json.dump(dict, f, indent=2)

    # Plot
    plot_x_1 = np.linspace(exp.G_off * (1 - 3 * Goff_sigma), exp.G_off * (1 + 3 * Goff_sigma))
    plot_x_2 = np.linspace(exp.G_on * (1 - 3 * Gon_sigma), exp.G_on * (1 + 3 * Gon_sigma))
    plot_y_1 = norm.pdf(plot_x_1, exp.G_off, exp.G_off * Goff_sigma)
    plot_y_2 = norm.pdf(plot_x_2, exp.G_on, exp.G_on * Gon_sigma)

    fig = plt.figure(figsize=(12, 16))
    ax1 = fig.add_subplot(321)
    ax1.hist(exp.G_off_variation, bins=10, density=True)
    ax1.plot(plot_x_1, plot_y_1, c='r')
    ax1.set_xlabel('G_off')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('D2D Variation (G_off)')
    ax2 = fig.add_subplot(322)
    ax2.hist(exp.G_on_variation, bins=10, density=True)
    ax2.plot(plot_x_2, plot_y_2, c='r')
    ax2.set_xlabel('G_on')
    ax2.set_ylabel('Probability Density')
    ax2.set_title('D2D Variation (G_on)')

    plot_x_1 = np.linspace(exp.P_off * (1 - 3 * Poff_sigma), exp.P_off * (1 + 3 * Poff_sigma))
    plot_x_2 = np.linspace(exp.P_on * (1 - 3 * Pon_sigma), exp.P_on * (1 + 3 * Pon_sigma))
    plot_y_1 = norm.pdf(plot_x_1, exp.P_off, exp.P_off * Poff_sigma)
    plot_y_2 = norm.pdf(plot_x_2, exp.P_on, exp.P_on * Pon_sigma)

    ax3 = fig.add_subplot(323)
    ax3.hist(exp.P_off_variation, bins=10, density=True)
    ax3.plot(plot_x_1, plot_y_1, c='r')
    ax3.set_xlabel('P_off')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('D2D Variation (P_off)')
    ax4 = fig.add_subplot(324)
    ax4.hist(exp.P_on_variation, bins=10, density=True)
    ax4.plot(plot_x_2, plot_y_2, c='r')
    ax4.set_xlabel('P_on')
    ax4.set_ylabel('Probability Density')
    ax4.set_title('D2D Variation (P_on)')

    z = np.array([sigma_relative ** 2 * 2 / np.pi, sigma_absolute ** 2 * 2 / np.pi])
    p = np.poly1d(z)
    plot_x = exp.x_mean
    plot_y = p(plot_x)

    ax5 = fig.add_subplot(325)
    ax5.scatter(exp.memx_total, exp.variation_x, c='r')
    ax5.set_xlabel('x')
    ax5.set_ylabel('Variation')
    ax5.set_title('C2C Variation')
    ax6 = fig.add_subplot(326)
    ax6.scatter(plot_x, exp.var_x_average, c='r')
    ax6.plot(plot_x, plot_y, c='b')
    ax6.set_xlabel('x')
    ax6.set_ylabel('Variation_mean')
    ax6.set_title('C2C Variation fitting')

    plt.tight_layout()
    plt.savefig("Variation.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
