import json
import sys

sys.path.append('../../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from simbrain.Fitting_Functions.iv_curve_fitting import IVCurve
from simbrain.Fitting_Functions.conductance_fitting import Conductance


def main():
    # Fit
    with open("../../../memristordata/my_memristor.json") as f:
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
    data = pd.DataFrame(pd.read_excel(
        "../../../memristordata/conductance_.xlsx",
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

    G_off_temp = G_off  # 1.85e-9
    G_on_temp = G_on  # 2.8e-10
    dict.update(
        {
            'G_off': G_off_temp,
            'G_on': G_on_temp
        }
    )

    dict.update(
        {
            'v_off': 2,
            'v_on': -2,
        }
    )

    file = "../../../memristordata/iv_curve.xlsx"
    exp_1 = IVCurve(file, dict)
    alpha_off, alpha_on = exp_1.fitting()
    # alpha_off_temp, alpha_on_temp = alpha_off, alpha_on  # 5, 5
    dict.update(
        {
            'G_off': G_off,
            'G_on': G_on,
            "alpha_off": alpha_off,
            "alpha_on": alpha_on,
            'v_off': 1.5,
            'v_on': -1.5,
            'delta_t': 100 * 1e-3,
        }
    )

    file = "../../../memristordata/conductance_.xlsx"
    # for alpha_off_temp in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #     for alpha_on_temp in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    #         dict.update(
    #             {
    #                 'alpha_off': alpha_off_temp,
    #                 'alpha_on': alpha_on_temp,
    #             }
    #         )
    #         exp_2 = Conductance(file, dict)
    #         P_off, P_on, k_off, k_on, _ = exp_2.fitting()
    exp_2 = Conductance(file, dict)
    P_off, P_on, k_off, k_on, _ = exp_2.fitting()
    dict.update(
        {
            # 'G_off': G_off_temp,
            # 'G_on': G_on_temp,
            "P_off": P_off,
            "P_on": P_on,
            "k_off": k_off,
            "k_on": k_on
        }
    )

    # Output
    df = pd.DataFrame(
        {
            'value': [alpha_off, alpha_on, P_off, P_on, k_off, k_on]
        },
        index=['alpha_off', 'alpha_on', 'P_off', 'P_on', 'k_off', 'k_on']
    )
    print(df)
    with open("fitting_record.json", "w") as f:
        json.dump(dict, f, indent=2)
    # print(exp_2.loss_r, exp_2.loss_d, torch.min(exp_2.loss_r), torch.min(exp_2.loss_d))

    # Plot
    fig = plt.figure(figsize=(12, 5.4))

    x_init = (exp_1.current[0] / exp_1.voltage[0] - exp_1.G_on) / (exp_1.G_off - exp_1.G_on)
    x_init = x_init if x_init > 0 else 0
    x_init = x_init if x_init < 1 else 1

    mem_x, mem_c = exp_1.Memristor_conductance_model(6, 4, x_init, exp_1.voltage)
    current_fit = np.array(mem_c) * np.array(exp_1.voltage)
    ax1 = fig.add_subplot(121)
    ax1.set_title('I-V Curve')
    ax1.plot(exp_1.voltage, current_fit, c='b')
    ax1.scatter(exp_1.voltage, exp_1.current, c='r')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current (A)')
    ax1.set_title('I-V Curve')

    dict.update(
        {
            'G_off': G_off,
            'G_on': G_on
        }
    )
    current_r = np.array(exp_2.data[:])[exp_2.start_point_r: exp_2.start_point_r + exp_2.points_r, 2:]
    current_d = np.array(exp_2.data[:])[exp_2.start_point_d: exp_2.start_point_d + exp_2.points_d, 2:]
    conductance_r = current_r / exp_2.read_voltage
    conductance_d = current_d / exp_2.read_voltage
    x_r = (conductance_r - exp_2.G_on) / (exp_2.G_off - exp_2.G_on)
    x_d = (conductance_d - exp_2.G_on) / (exp_2.G_off - exp_2.G_on)
    V_write_r = exp_2.V_write[exp_2.start_point_r: exp_2.start_point_r + exp_2.points_r]
    V_write_d = exp_2.V_write[exp_2.start_point_d: exp_2.start_point_d + exp_2.points_d]
    x_init_r = x_r[0]
    x_init_d = x_d[0]
    mem_x_r = np.zeros(exp_2.points_r)
    mem_x_d = np.zeros(exp_2.points_d)
    mem_x_r[0] = np.average(x_init_r)
    mem_x_d[0] = np.average(x_init_d)
    J1 = 1
    # k_off = 0.45
    # P_off = 0.25
    # k_on = -400
    # P_on = 1.3

    for i in range(exp_2.points_r - 1):
        mem_x_r[i + 1] = (
                k_off
                * ((V_write_r[i + 1] / exp_2.v_off - 1) ** exp_2.alpha_off)
                * J1
                * (1 - mem_x_r[i]) ** P_off
                * exp_2.delta_t
                * exp_2.duty_ratio
                + mem_x_r[i]
        )
        mem_x_r[i + 1] = np.where(mem_x_r[i + 1] < 0, 0, mem_x_r[i + 1])
        mem_x_r[i + 1] = np.where(mem_x_r[i + 1] > 1, 1, mem_x_r[i + 1])
    for i in range(exp_2.points_d - 1):
        mem_x_d[i + 1] = (
                k_on
                * ((V_write_d[i + 1] / exp_2.v_on - 1) ** exp_2.alpha_on)
                * J1
                * mem_x_d[i] ** P_on
                * exp_2.delta_t
                * exp_2.duty_ratio
                + mem_x_d[i]
        )
        mem_x_d[i + 1] = np.where(mem_x_d[i + 1] < 0, 0, mem_x_d[i + 1])
        mem_x_d[i + 1] = np.where(mem_x_d[i + 1] > 1, 1, mem_x_d[i + 1])
    memx_total = np.concatenate((mem_x_r, mem_x_d))
    x_total = np.concatenate((x_r, x_d))
    plot_x = np.arange(exp_2.points_r)

    ax2 = fig.add_subplot(122)
    ax2.set_title('Conductance Curve')
    ax2.plot(plot_x, mem_x_r, c='b')
    for i in range(exp_2.device_nums):
        ax2.scatter(plot_x, x_r.T[i], c='r', s=0.1)
    ax2.set_xlabel('points')
    ax2.set_ylabel('x')
    ax2.set_title('Conductance Curve')

    plt.tight_layout()
    plt.savefig("Baseline Model.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
