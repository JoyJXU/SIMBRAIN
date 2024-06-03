import json
import sys

sys.path.append('../../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from simbrain.Fitting_Functions.iv_curve_fitting import IVCurve
from simbrain.Fitting_Functions.conductance_fitting import Conductance
import os
import math
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1]))

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
    alpha_off, alpha_on = 5, 5
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

    file = "../../../memristordata/conductance_deletehead.xlsx"
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
    fig = plt.figure(figsize=(20, 12))

    x_init = (exp_1.current[0] / exp_1.voltage[0] - exp_1.G_on) / (exp_1.G_off - exp_1.G_on)
    x_init = x_init if x_init > 0 else 0
    x_init = x_init if x_init < 1 else 1

    mem_x, mem_c = exp_1.Memristor_conductance_model(6, 4, x_init, exp_1.voltage)
    current_fit = np.array(mem_c) * np.array(exp_1.voltage)
    # ax1 = fig.add_subplot(221)
    # ax1.set_title('I-V Curve')
    # ax1.plot(exp_1.voltage, current_fit, c='b')
    # ax1.scatter(exp_1.voltage, exp_1.current, c='r')
    # ax1.set_xlabel('Voltage (V)')
    # ax1.set_ylabel('Current (A)')
    # ax1.set_title('I-V Curve')

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
    # mem_x_d[0] = 0.7
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
    plot_x = np.arange(exp_2.points_r + exp_2.points_d)

    ax2 = fig.add_subplot(231)
    ax2.set_title('Conductance Curve')
    ax2.plot(plot_x, memx_total, c='b')
    for i in range(exp_2.device_nums):
        ax2.scatter(plot_x, x_total.T[i], c='r', s=0.1, alpha=0.3)
    ax2.set_xlabel('points')
    ax2.set_ylabel('x')
    ax2.set_title('Conductance Curve')

    # Get C2C information
    cr_tensor = torch.from_numpy(conductance_r)
    cd_tensor = torch.from_numpy(conductance_d)
    Goff_tensor = torch.from_numpy(G_off_list)
    Gon_tensor = torch.from_numpy(G_on_list)
    xr_tensor = (cr_tensor - Gon_tensor) / (Goff_tensor - Gon_tensor)
    xd_tensor = (cd_tensor - Gon_tensor) / (Goff_tensor - Gon_tensor)
    x_tensor = torch.cat((xr_tensor, xd_tensor), dim=0)

    plot_x = np.arange(x_r.shape[0] + x_d.shape[0])
    ax3 = fig.add_subplot(232)
    ax3.set_title('X Curve')
    for i in range(exp_2.device_nums):
        ax3.scatter(plot_x, x_tensor[:, i], c='r', s=0.1, alpha=0.3)

    xt_tensor = torch.from_numpy(x_total)
    memx_tensor = torch.from_numpy(memx_total)
    vx_tensor = torch.abs(xt_tensor - memx_tensor.unsqueeze(1))

    def takeFirst(ele):
        return ele[0]

    memxt_tensor = torch.from_numpy(memx_total)
    memxt_tensor = memxt_tensor.unsqueeze(1)
    memxt_tensor = memxt_tensor.expand(470, 53)
    complex_tensor = torch.cat((memxt_tensor.flatten().unsqueeze(0), vx_tensor.flatten().unsqueeze(0)), dim=0)
    # complex_tensor.sort(key=takeFirst)

    ax4 = fig.add_subplot(233)
    ax4.set_title('X Variation')
    for i in range(exp_2.device_nums):
        ax4.scatter(plot_x, vx_tensor[:, i], c='r', s=0.1, alpha=0.3)

    ax5 = fig.add_subplot(234)
    ax5.set_title('X Variation')
    ax5.scatter(complex_tensor[0], complex_tensor[1], c='r', s=0.1, alpha=0.3)

    model_x = complex_tensor[0]
    variation_raw = complex_tensor[1]
    group = 10
    segments = torch.linspace(0, 1, group + 1)
    x_mean = torch.zeros(group)
    variation_mean = torch.zeros(group)
    for i in range(group):
        if i == group - 1:
            mask = (model_x >= segments[i]) & (model_x <= segments[i+1])
        else:
            mask = (model_x >= segments[i]) & (model_x < segments[i+1])

        x_mean[i] = model_x[mask].float().mean()
        variation_mean[i] = variation_raw[mask].float().mean()

    z1 = np.polyfit(torch.square(x_mean), torch.square(variation_mean), 1)
    p1 = np.poly1d(z1)
    print(p1)

    sigma_relative = math.sqrt(z1[0] * math.pi / 2)
    sigma_absolute = math.sqrt(z1[1] * math.pi / 2)
    print(sigma_relative)
    print(sigma_absolute)

    ax6 = fig.add_subplot(235)
    ax6.set_title('Clustering')
    ax6.scatter(torch.square(x_mean), torch.square(variation_mean), c='r', s=5)

    plt.tight_layout()
    plt.savefig("Baseline Model.png", dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    main()
