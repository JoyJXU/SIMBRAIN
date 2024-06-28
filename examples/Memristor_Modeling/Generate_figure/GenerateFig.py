import os
import json
import sys
sys.path.append('../../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from simbrain.Fitting_Functions.conductance_fitting import Conductance
from simbrain.Fitting_Functions.variation_fitting import Variation


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [2]))
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    # Fit
    with open("../../../memristor_data/my_memristor.json") as f:
        dict = json.load(f)
    dict.update(
        {
            'v_off': 1.5,
            'v_on': -1.5,
            'G_off': None,
            'G_on': None,
            'alpha_off': None,
            'alpha_on': None,
            'k_off': None,
            'k_on': None,
            'P_off': None,
            'P_on': None,
            'delta_t': 100 * 1e-3,
        }
    )
    file = "../../../memristor_data/conductance_deletehead.xlsx"

    data = pd.DataFrame(pd.read_excel(
        file,
        sheet_name=0,
        header=None,
        index_col=None,
    ))
    data.columns = ['Pulse Voltage(V)', 'Read Voltage(V)'] + list(data.columns[2:] - 2)

    V_write = np.array(data['Pulse Voltage(V)'])
    points_r = np.sum(V_write > 0)
    points_d = np.sum(V_write < 0)
    read_voltage = np.array(data['Read Voltage(V)'])[0]

    device_nums = data.shape[1] - 2
    G_off_list = np.zeros(device_nums)
    G_on_list = np.zeros(device_nums)

    for i in range(device_nums):
        G_off_list[i] = np.average(
            data[i][points_r - 10:points_r] / read_voltage
        )
        G_on_list[i] = np.average(
            data[i][points_r + points_d - 10:] / read_voltage
        )

    G_off = np.mean(G_off_list)
    G_on = np.mean(G_on_list)
    dict.update(
        {
            'G_off': G_off,
            'G_on': G_on
        }
    )

    P_off_list = np.zeros(device_nums)
    P_on_list = np.zeros(device_nums)

    _, Goff_sigma, _, Gon_sigma = Variation(
        file,
        G_off_list,
        G_on_list,
        P_off_list,
        P_on_list,
        dict
    ).d2d_G_fitting()
    dict.update(
        {
            "Goff_sigma": Goff_sigma,
            "Gon_sigma": Gon_sigma,
        }
    )

    alpha_off, alpha_on = 5, 5
    dict.update(
        {
            "alpha_off": alpha_off,
            "alpha_on": alpha_on
        }
    )
    exp_0 = Conductance(file, dict)
    P_off, P_on, k_off, k_on, _ = exp_0.fitting()
    RRMSE = np.min(exp_0.loss.cpu().numpy())
    dict.update(
        {
            'k_off': k_off,
            'k_on': k_on,
            'P_off': P_off,
            'P_on': P_on,
        }
    )

    P_off_list, P_on_list = exp_0.mult_P_fitting(G_off_list, G_on_list)

    exp = Variation(
        file,
        G_off_list,
        G_on_list,
        P_off_list,
        P_on_list,
        dict
    )
    _, Poff_sigma, _, Pon_sigma = exp.d2d_P_fitting()
    dict.update(
        {
            "Poff_sigma": Pon_sigma,
            "Pon_sigma": Poff_sigma
        }
    )

    # Output
    df = pd.DataFrame(
        {'value': [k_off, k_on, P_off, P_on, RRMSE, Goff_sigma, Gon_sigma, Poff_sigma, Pon_sigma]},
        index=['k_off', 'k_on', 'P_off', 'P_on', 'RRMSE', 'Goff_sigma', 'Gon_sigma', 'Poff_sigma', 'Pon_sigma']
    )
    print(df)

    # Plot
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2)

    plot_x_1 = np.linspace(exp.G_off * (1 - 3 * Goff_sigma), exp.G_off * (1 + 3 * Goff_sigma))
    plot_x_2 = np.linspace(exp.G_on * (1 - 3 * Gon_sigma), exp.G_on * (1 + 3 * Gon_sigma))
    plot_y_1 = norm.pdf(plot_x_1, exp.G_off, exp.G_off * Goff_sigma)
    plot_y_2 = norm.pdf(plot_x_2, exp.G_on, exp.G_on * Gon_sigma)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(exp.G_off_variation, bins=20, density=True)
    ax1.plot(plot_x_1, plot_y_1, color='red', label='Goff')
    ax1.set_xlabel('G_off/G_on')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('D2D Variation')
    ax2 = ax1.twinx()
    ax2.hist(exp.G_on_variation, bins=20, density=True, color='orange')
    ax2.plot(plot_x_2, plot_y_2, color='green', label='G_on')
    ax2.set_ylabel('Probability Density')
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1))

    plot_x_1 = np.linspace(exp.P_off * (1 - 3 * Poff_sigma), exp.P_off * (1 + 3 * Poff_sigma))
    plot_x_2 = np.linspace(exp.P_on * (1 - 3 * Pon_sigma), exp.P_on * (1 + 3 * Pon_sigma))
    plot_y_1 = norm.pdf(plot_x_1, exp.P_off, exp.P_off * Poff_sigma)
    plot_y_2 = norm.pdf(plot_x_2, exp.P_on, exp.P_on * Pon_sigma)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(exp.P_off_variation, bins=20, density=True)
    ax3.plot(plot_x_1, plot_y_1, color='red', label='P_off')
    ax3.set_xlabel('P_off/P_on')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('D2D Variation')
    ax4 = ax3.twinx()
    ax4.hist(exp.P_on_variation, bins=20, density=True, color='orange')
    ax4.plot(plot_x_2, plot_y_2, color='green', label='P_on')
    ax4.set_ylabel('Probability Density')
    lines = ax3.get_lines() + ax4.get_lines()
    labels = [line.get_label() for line in lines]
    ax3.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1))

    current_r = np.array(exp_0.data[:])[exp_0.start_point_r: exp_0.start_point_r + exp_0.points_r, 2:]
    current_d = np.array(exp_0.data[:])[exp_0.start_point_d: exp_0.start_point_d + exp_0.points_d, 2:]
    conductance_r = current_r / exp_0.read_voltage
    conductance_d = current_d / exp_0.read_voltage
    x_r = (conductance_r - exp_0.G_on) / (exp_0.G_off - exp_0.G_on)
    x_d = (conductance_d - exp_0.G_on) / (exp_0.G_off - exp_0.G_on)
    V_write_r = exp_0.V_write[exp_0.start_point_r: exp_0.start_point_r + exp_0.points_r]
    V_write_d = exp_0.V_write[exp_0.start_point_d: exp_0.start_point_d + exp_0.points_d]
    x_init_r = x_r[0]
    x_init_d = x_d[0]
    mem_x_r = np.zeros(exp_0.points_r)
    mem_x_d = np.zeros(exp_0.points_d)
    mem_x_r[0] = np.average(x_init_r)
    mem_x_d[0] = np.average(x_init_d)
    J1 = 1
    # k_off = 0.45
    # P_off = 0.25
    # k_on = -400
    # P_on = 1.3

    for i in range(exp_0.points_r - 1):
        mem_x_r[i + 1] = (
                k_off
                * ((V_write_r[i + 1] / exp_0.v_off - 1) ** exp_0.alpha_off)
                * J1
                * (1 - mem_x_r[i]) ** P_off
                * exp_0.delta_t
                * exp_0.duty_ratio
                + mem_x_r[i]
        )
        mem_x_r[i + 1] = np.where(mem_x_r[i + 1] < 0, 0, mem_x_r[i + 1])
        mem_x_r[i + 1] = np.where(mem_x_r[i + 1] > 1, 1, mem_x_r[i + 1])
    for i in range(exp_0.points_d - 1):
        mem_x_d[i + 1] = (
                k_on
                * ((V_write_d[i + 1] / exp_0.v_on - 1) ** exp_0.alpha_on)
                * J1
                * mem_x_d[i] ** P_on
                * exp_0.delta_t
                * exp_0.duty_ratio
                + mem_x_d[i]
        )
        mem_x_d[i + 1] = np.where(mem_x_d[i + 1] < 0, 0, mem_x_d[i + 1])
        mem_x_d[i + 1] = np.where(mem_x_d[i + 1] > 1, 1, mem_x_d[i + 1])
    memx_total = np.concatenate((mem_x_r, mem_x_d))
    x_total = np.concatenate((x_r, x_d))
    plot_x = np.arange(exp_0.points_r)

    colors = plt.cm.viridis(np.linspace(0, 1, exp_0.device_nums))

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.set_title('Potential Curve')
    ax5.plot(plot_x, mem_x_r, linewidth=3, c='r')
    for i in range(exp_0.device_nums):
        ax5.scatter(plot_x, x_r.T[i], color=colors[i], s=0.5)
    ax5.set_xlabel('points')
    ax5.set_ylabel('x')
    ax5.set_title('Potential Curve')

    plt.tight_layout()
    plt.savefig("Fig3.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
