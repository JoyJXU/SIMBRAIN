import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.optimize import curve_fit
from scipy.stats import norm


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time
        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print('%s cost time: %.3f s' % (func.__name__, time_spend))
        return result

    return func_wrapper


def gaussian(x, mu, sigma):
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


class Variation(object):
    def __init__(
            self,
            file,
            G_off_list: np.array,
            G_on_list: np.array,
            P_off_list: np.array,
            P_on_list: np.array,
            dictionary: dict = {},
            **kwargs,
    ) -> None:
        self.data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name='Sheet1',
            header=None,
            index_col=None
        ))
        self.data.columns = ['Pulse Voltage(V)', 'Read Voltage(V)'] + list(self.data.columns[2:] - 2)

        self.J1 = 1
        self.v_off = dictionary['v_off']
        self.v_on = dictionary['v_on']
        self.G_off = dictionary['G_off']
        self.G_on = dictionary['G_on']
        self.alpha_off = dictionary['alpha_off']
        self.alpha_on = dictionary['alpha_on']
        self.k_off = dictionary['k_off']
        self.k_on = dictionary['k_on']
        self.P_off = dictionary['P_off']
        self.P_on = dictionary['P_on']
        self.delta_t = dictionary['delta_t']

        self.V_write = np.array(self.data['Pulse Voltage(V)'])
        self.points_r = np.sum(self.V_write > 0)
        self.points_d = np.sum(self.V_write < 0)

        self.V_write_r = self.V_write[:self.points_r]
        self.V_write_d = self.V_write[self.points_r:]
        self.read_voltage = np.array(self.data['Read Voltage(V)'])[0]

        self.device_num = self.data.shape[1] - 2
        self.G_off_variation = G_off_list
        self.G_on_variation = G_on_list
        if self.G_off is None:
            self.G_off = np.mean(self.G_off_variation)
        if self.G_on is None:
            self.G_on = np.mean(self.G_on_variation)

        self.P_off_variation = P_off_list
        self.P_on_variation = P_on_list
        if None in [self.P_off, self.P_on]:
            self.P_off = np.mean(self.P_off_variation)
            self.P_on = np.mean(self.P_on_variation)

    def Memristor_conductance_model(self, P_off, P_on, initial, V_write):
        points = len(V_write)
        # initialization
        internal_state = [0 for i in range(points)]
        internal_state[0] = initial
        # conductance change
        for i in range(points - 1):
            if V_write[i + 1] > self.v_off and V_write[i + 1] > 0:
                delta_x = (
                        self.k_off
                        * ((V_write[i + 1] / self.v_off - 1) ** self.alpha_off)
                        * self.J1
                        * ((1 - internal_state[i]) ** P_off)
                )
                internal_state[i + 1] = internal_state[i] + self.delta_t * delta_x

            elif V_write[i + 1] < 0 and V_write[i + 1] < self.v_on:
                delta_x = (
                        self.k_on
                        * ((V_write[i + 1] / self.v_on - 1) ** self.alpha_on)
                        * self.J1
                        * (internal_state[i] ** P_on)
                )
                internal_state[i + 1] = internal_state[i] + self.delta_t * delta_x

            else:
                delta_x = 0
                internal_state[i + 1] = internal_state[i]

            if internal_state[i + 1] < 0:
                internal_state[i + 1] = 0
            elif internal_state[i + 1] > 1:
                internal_state[i + 1] = 1

        return internal_state

    def RRMSE_PERCENT(
            self,
            conductance_fit,
            conductance
    ):
        c_diff = list(map(lambda x: x[0] - x[1], zip(conductance_fit, conductance)))
        c_diff_percent = []
        for i in range(len(c_diff)):
            c_diff_percent.append(c_diff[i] / conductance[i])

        square_sum = np.dot(c_diff_percent, c_diff_percent)
        mean_square_sum = square_sum / len(c_diff_percent)
        RRMSE_percent = np.sqrt(mean_square_sum)
        return RRMSE_percent

    @timer
    def d2d_G_fitting(self):
        G_off_cal = (self.G_off_variation - self.G_off) / self.G_off
        G_on_cal = (self.G_on_variation - self.G_on) / self.G_on
        Goff_mu_noise, Goff_sigma = norm.fit(G_off_cal)
        Goff_mu = self.G_off * (1 + Goff_mu_noise)
        Gon_mu_noise, Gon_sigma = norm.fit(G_on_cal)
        Gon_mu = self.G_on * (1 + Gon_mu_noise)

        # bins_num = 10
        # Goff_hist, bin_edge_off = np.histogram(G_off_cal, bins=bins_num)
        # edge_diff_off = bin_edge_off[1] - bin_edge_off[0]
        # params, _ = curve_fit(
        #     gaussian,
        #     bin_edge_off[:bins_num],
        #     Goff_hist / Goff_hist.sum() / edge_diff_off
        # )
        # Goff_mu = self.G_off * (1 + params[0])
        # Goff_sigma = params[1]
        # # print(self.G_off * (1 + params[0]), Goff_sigma)


        # Gon_hist, bin_edge_on = np.histogram(G_on_cal, bins=bins_num)
        # edge_diff_on = bin_edge_on[1] - bin_edge_on[0]
        # params, _ = curve_fit(
        #     gaussian,
        #     bin_edge_on[:bins_num],
        #     Gon_hist / Gon_hist.sum() / edge_diff_on
        # )
        # Gon_mu = self.G_on * (1 + params[0])
        # Gon_sigma = params[1]
        # # print(self.G_on * (1 + params[0]), Gon_sigma)

        return Goff_mu, Goff_sigma, Gon_mu, Gon_sigma

    @timer
    def d2d_P_fitting(self):
        P_off_cal = (self.P_off_variation - self.P_off) / self.P_off
        P_on_cal = (self.P_on_variation - self.P_on) / self.P_on
        Poff_mu_noise, Poff_sigma = norm.fit(P_off_cal)
        Poff_mu = self.P_off * (1 + Poff_mu_noise)
        Pon_mu_noise, Pon_sigma = norm.fit(P_on_cal)
        Pon_mu = self.P_on * (1 + Pon_mu_noise)
        print(Poff_mu, Pon_mu)

        # bins_num = 10
        # Poff_hist, bin_edge_off = np.histogram(P_off_cal, bins=bins_num)
        # edge_diff_off = bin_edge_off[1] - bin_edge_off[0]
        # params, _ = curve_fit(
        #     gaussian,
        #     bin_edge_off[:bins_num],
        #     Poff_hist / Poff_hist.sum() / edge_diff_off
        # )
        # Poff_mu = self.P_off * (1 + params[0])
        # Poff_sigma = params[1]
        #
        # Pon_hist, bin_edge_on = np.histogram(P_on_cal, bins=bins_num)
        # edge_diff_on = bin_edge_on[1] - bin_edge_on[0]
        # params, _ = curve_fit(
        #     gaussian,
        #     bin_edge_on[:bins_num],
        #     Pon_hist / Pon_hist.sum() / edge_diff_on
        # )
        # Pon_mu = self.P_on * (1 + params[0])
        # Pon_sigma = params[1]

        return Poff_mu, Poff_sigma, Pon_mu, Pon_sigma

    @timer
    def c2c_fitting(self):
        best_mem_x_r = []
        best_mem_x_d = []
        for i in range(self.device_num):
            best_mem_x_r.append(np.array(self.Memristor_conductance_model(
                self.P_off_variation[i],
                self.P_on_variation[i],
                (self.data[i][0] / self.read_voltage - self.G_on_variation[i])
                / (self.G_off_variation[i] - self.G_on_variation[i]),
                self.V_write_r
            )))
            best_mem_x_d.append(np.array(self.Memristor_conductance_model(
                self.P_off_variation[i],
                self.P_on_variation[i],
                (self.data[i][self.points_r] / self.read_voltage - self.G_on_variation[i])
                / (self.G_off_variation[i] - self.G_on_variation[i]),
                self.V_write_d
            )))
            # best_mem_x_d.append(np.array(self.Memristor_conductance_model(
            #     self.P_off_variation[i],
            #     self.P_on_variation[i],
            #     (self.data[i][self.points_r - 1] / self.read_voltage - self.G_on)
            #     / (self.G_off - self.G_on),
            #     self.V_write_d
            # )))
        best_memx_total = np.concatenate((np.array(best_mem_x_r), np.array(best_mem_x_d)), axis=1).flatten()

        x_r = []
        x_d = []
        for i in range(self.device_num):
            conductance_r = np.array(self.data[i][:self.points_r] / self.read_voltage)
            x_r.append(
                (conductance_r - self.G_on_variation[i])
                / (self.G_off_variation[i] - self.G_on_variation[i])
            )
            # x_r.append((conductance_r - self.G_on) / (self.G_off - self.G_on))
            conductance_d = np.array(self.data[i][self.points_r:] / self.read_voltage)
            x_d.append(
                (conductance_d - self.G_on_variation[i])
                / (self.G_off_variation[i] - self.G_on_variation[i])
            )
            # x_d.append((conductance_d - self.G_on) / (self.G_off - self.G_on))
        x_total = np.concatenate((np.array(x_r), np.array(x_d)), axis=1).flatten()

        variation_x = abs(best_memx_total - x_total)

        # %% Variation Analysis
        var_x_complex = []
        for i in range(len(x_total)):
            var_x_complex.append((x_total[i], variation_x[i]))

        def takeFirst(ele):
            return ele[0]

        # Sort
        var_x_complex.sort(key=takeFirst)

        # %% Variation Clustering
        x_sorted = np.array([i[0] for i in var_x_complex])
        var_x_sorted = np.array([i[1] for i in var_x_complex])

        group_no = max(10, int((len(x_total)/5)**0.5))
        print('Group number:{}'.format(group_no))

        # Group with pulse number
        total_points = (self.points_r + self.points_d) * self.device_num
        every_points = math.ceil(total_points / group_no)

        x_mean = []
        var_x_median = []
        var_x_average = []
        for i in range(group_no):
            temp_x_mean = np.mean(x_sorted[every_points * i:(every_points * (i + 1))])
            temp_var_median = np.median(var_x_sorted[every_points * i:(every_points * (i + 1))])
            temp_var_average = np.mean(var_x_sorted[every_points * i:(every_points * (i + 1))])
            x_mean.append(temp_x_mean ** 2)
            var_x_median.append(temp_var_median ** 2)
            var_x_average.append(temp_var_average ** 2)

        z1 = np.polyfit(x_mean, var_x_median, 1)
        p1 = np.poly1d(z1)
        print(p1)

        z2 = np.polyfit(x_mean, var_x_average, 1)
        p2 = np.poly1d(z2)
        print(p2)

        self.x_mean = x_mean
        self.var_x_average = var_x_average
        self.memx_total = best_memx_total
        self.variation_x = variation_x

        sigma_relative = math.sqrt(z2[0] * math.pi / 2)
        sigma_absolute = math.sqrt(z2[1] * math.pi / 2)
        # sigma_relative = 1
        # sigma_absolute = 1

        return sigma_relative, sigma_absolute
