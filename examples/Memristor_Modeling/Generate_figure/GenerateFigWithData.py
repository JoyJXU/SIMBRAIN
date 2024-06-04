import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm


def main():
    # read data
    file_json = "dict.json"
    with open(file_json) as f:
        dict = json.load(f)
    G_off = dict['G_off']
    G_on = dict['G_on']
    P_off = dict['P_off']
    P_on = dict['P_on']

    file_G = "G_variation.xlsx"
    file_P = "P_variation.xlsx"
    file_scatter = "x_scatter.xlsx"
    file_curve = "conductance_curve.xlsx"
    best_curve = "best_curve.xlsx"

    data_G = pd.DataFrame(pd.read_excel(
        file_G,
        sheet_name=0,
        header=None,
        index_col=None,
    ))
    data_G.columns = ['G_off', 'G_on']
    G_off_variation = np.array(data_G['G_off'])
    G_on_variation = np.array(data_G['G_on'])

    data_P = pd.DataFrame(pd.read_excel(
        file_P,
        sheet_name=0,
        header=None,
        index_col=None,
    ))
    data_P.columns = ['P_off', 'P_on']
    P_off_variation = np.array(data_P['P_off'])
    P_on_variation = np.array(data_P['P_on'])

    data_scatter = pd.DataFrame(pd.read_excel(
        file_scatter,
        sheet_name=0,
        header=None,
        index_col=None,
    ))

    data_curve = pd.DataFrame(pd.read_excel(
        file_curve,
        sheet_name=0,
        header=None,
        index_col=None,
    ))

    best_curve = pd.DataFrame(pd.read_excel(
        best_curve,
        sheet_name=0,
        header=None,
        index_col=None,
    ))

    # pre-process
    G_off_cal = (G_off_variation - G_off) / G_off
    G_on_cal = (G_on_variation - G_on) / G_on
    _, Goff_sigma = norm.fit(G_off_cal)
    _, Gon_sigma = norm.fit(G_on_cal)

    P_off_cal = (P_off_variation - P_off) / P_off
    P_on_cal = (P_on_variation - P_on) / P_on
    _, Poff_sigma = norm.fit(P_off_cal)
    _, Pon_sigma = norm.fit(P_on_cal)

    # plot
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(2, 2)

    plot_x_1 = np.linspace(G_off * (1 - 3 * Goff_sigma), G_off * (1 + 3 * Goff_sigma))
    plot_x_2 = np.linspace(G_on * (1 - 3 * Gon_sigma), G_on * (1 + 3 * Gon_sigma))
    plot_y_1 = norm.pdf(plot_x_1, G_off, G_off * Goff_sigma)
    plot_y_2 = norm.pdf(plot_x_2, G_on, G_on * Gon_sigma)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(G_off_variation, bins=20, density=True)
    ax1.plot(plot_x_1, plot_y_1, color='red', label='Goff')
    ax1.set_xlabel('G_off/G_on')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('D2D Variation')
    ax2 = ax1.twinx()
    ax2.hist(G_on_variation, bins=20, density=True, color='orange')
    ax2.plot(plot_x_2, plot_y_2, color='green', label='G_on')
    ax2.set_ylabel('Probability Density')
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1))

    plot_x_1 = np.linspace(P_off * (1 - 3 * Poff_sigma), P_off * (1 + 3 * Poff_sigma))
    plot_x_2 = np.linspace(P_on * (1 - 3 * Pon_sigma), P_on * (1 + 3 * Pon_sigma))
    plot_y_1 = norm.pdf(plot_x_1, P_off, P_off * Poff_sigma)
    plot_y_2 = norm.pdf(plot_x_2, P_on, P_on * Pon_sigma)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.hist(P_off_variation, bins=20, density=True)
    ax3.plot(plot_x_1, plot_y_1, color='red', label='P_off')
    ax3.set_xlabel('P_off/P_on')
    ax3.set_ylabel('Probability Density')
    ax3.set_title('D2D Variation')
    ax4 = ax3.twinx()
    ax4.hist(P_on_variation, bins=20, density=True, color='orange')
    ax4.plot(plot_x_2, plot_y_2, color='green', label='P_on')
    ax4.set_ylabel('Probability Density')
    lines = ax3.get_lines() + ax4.get_lines()
    labels = [line.get_label() for line in lines]
    ax3.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, 1))

    plot_x = np.arange(data_curve.shape[0])
    colors = plt.cm.viridis(np.linspace(0, 1, data_curve.shape[1]))

    ax5 = fig.add_subplot(gs[1, 0])
    ax5.set_title('Potential Curve')
    ax5.plot(plot_x, best_curve, linewidth=10, c='r')
    for i in range(data_curve.shape[1]):
        ax5.scatter(plot_x, data_scatter[i], color='orange', s=0.5, alpha=0.5)
        # ax5.plot(plot_x, data_curve[i], color=colors[i], linewidth=0.5)
        ax5.plot(plot_x, data_curve[i], color='blue', linewidth=0.4, alpha=0.5)
    ax5.set_xlabel('points')
    ax5.set_ylabel('x')
    ax5.set_title('Potential Curve')

    plt.tight_layout()
    plt.savefig("Fig.png", dpi=300)
    plt.show()


if __name__ == '__main__':
    main()
