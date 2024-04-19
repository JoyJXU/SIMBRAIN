import json
import sys
sys.path.append('../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simbrain.Fitting_Functions.IV_curve_fitting import IVCurve
from simbrain.Fitting_Functions.conductance_fitting import Conductance


def main():
    # Fit
    with open("../../../memristordata/my_memristor.json") as f:
        dict = json.load(f)
    data = pd.DataFrame(pd.read_excel(
        "../../../memristordata/conductance.xlsx",
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
            "G_off": G_off,
            "G_on": G_on
        }
    )

    file = "../../../memristordata/IV_curve.xlsx"
    exp_1 = IVCurve(file, dict)
    alpha_off, alpha_on = exp_1.fitting()
    alpha_off, alpha_on = 5, 5
    dict.update(
        {
            "alpha_off": alpha_off,
            "alpha_on": alpha_on
        }
    )

    file = "../../../memristordata/conductance.xlsx"
    exp_2 = Conductance(file, dict)
    P_off, P_on, k_off, k_on = exp_2.fitting()
    dict.update(
        {
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

    # Plot
    fig = plt.figure(figsize=(12, 5.4))

    x_init = (exp_1.current[0] / exp_1.voltage[0] - exp_1.G_on) / (exp_1.G_off - exp_1.G_on)
    mem_x, mem_c = exp_1.Memristor_conductance_model(alpha_off, alpha_on, x_init, exp_1.voltage)
    current_fit = np.array(mem_c) * np.array(exp_1.voltage)
    ax1 = fig.add_subplot(121)
    ax1.set_title('I-V Curve')
    ax1.plot(exp_1.voltage, exp_1.current, c='b')
    ax1.scatter(exp_1.voltage, exp_1.current, c='r')
    ax1.set_xlabel('Voltage (V)')
    ax1.set_ylabel('Current (A)')
    ax1.set_title('I-V Curve')

    V_write_r = exp_2.V_write[exp_2.start_point_r: exp_2.start_point_r + exp_2.points_r]
    x_init_r = exp_2.x_r[0]
    V_write_d = exp_2.V_write[exp_2.start_point_d: exp_2.start_point_d + exp_2.points_d]
    x_init_d = exp_2.x_d[0]

    mem_x_r, _ = exp_2.Memristor_conductance_model(
        exp_2.k_off, exp_2.k_on, exp_2.P_off, exp_2.P_on, x_init_r, V_write_r
    )
    mem_x_d, _ = exp_2.Memristor_conductance_model(
        exp_2.k_off, exp_2.k_on, exp_2.P_off, exp_2.P_on, x_init_d, V_write_d
    )
    memx_total = np.concatenate((mem_x_r, mem_x_d))
    x_total = np.concatenate((exp_2.x_r, exp_2.x_d))
    plot_x = np.arange(exp_2.points_r + exp_2.points_d)

    ax2 = fig.add_subplot(122)
    ax2.set_title('Conductance Curve')
    ax2.plot(plot_x, memx_total, c='b')
    ax2.scatter(plot_x, x_total, c='r')
    ax2.set_xlabel('points')
    ax2.set_ylabel('x')
    ax2.set_title('Conductance Curve')

    plt.tight_layout()
    plt.savefig("Baseline Model.png", dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    main()
