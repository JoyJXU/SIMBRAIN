import math
import numpy as np
import pandas as pd


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


class Conductance(object):
    def __init__(
            self,
            file,
            dictionary: dict = {},
            **kwargs,
    ):
        # Read excel
        data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name=0,
            header=None,
            names=[
                'Pulse Voltage(V)',
                'Current(A)',
                'Read Voltage(V)'
            ]
        ))

        # Initialize parameters
        self.P_off = 1
        self.P_on = 1
        self.k_off = 1
        self.k_on = -1

        # Read parameters
        self.v_off = dictionary['v_off']
        self.v_on = dictionary['v_on']
        self.G_off = dictionary['G_off']
        self.G_on = dictionary['G_on']
        self.alpha_off = dictionary['alpha_off']
        self.alpha_on = dictionary['alpha_on']

        # Read data
        self.delta_t = dictionary['delta_t']
        self.V_write = np.array(data['Pulse Voltage(V)'])
        self.read_voltage = np.array(data['Read Voltage(V)'][0])

        self.start_point_r = 0
        self.points_r = np.sum(self.V_write>0)
        self.start_point_d = self.start_point_r + np.sum(self.V_write>0)
        self.points_d = np.sum(self.V_write<0)

        self.current_r = np.array(data['Current(A)'])[self.start_point_r: self.start_point_r + self.points_r]
        self.conductance_r = self.current_r / self.read_voltage
        self.x_r = (self.conductance_r - self.G_on) / (self.G_off - self.G_on)
        self.current_d = np.array(data['Current(A)'])[self.start_point_d: self.start_point_d + self.points_d]
        self.conductance_d = self.current_d / self.read_voltage
        self.x_d = (self.conductance_d - self.G_on) / (self.G_off - self.G_on)

    def RRMSE_MEAN(
            self,
            internal_state,
            x
    ):
        points = len(internal_state)
        x_diff = list(map(lambda x: x[0] - x[1], zip(internal_state, x)))
        square_sum = np.dot(x_diff, x_diff)
        mean_square_sum = square_sum / points
        RMSE = np.sqrt(mean_square_sum)
        RRMSE_mean = RMSE / np.mean(x)
        return RRMSE_mean

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

    def Memristor_conductance_model(
            self,
            k_off,
            k_on,
            P_off,
            P_on,
            x_init,
            V_write
    ):
        J1 = 1
        points = len(V_write)

        # initialization
        internal_state = [0 for i in range(points)]
        conductance_fit = [0 for i in range(points)]

        # conductance change
        internal_state[0] = x_init
        for i in range(points - 1):
            if V_write[i + 1] > self.v_off and V_write[i + 1] > 0:
                delta_x = k_off * ((V_write[i + 1] / self.v_off - 1) ** self.alpha_off) * J1 * (
                        (1 - internal_state[i]) ** P_off)
                internal_state[i + 1] = internal_state[i] + self.delta_t * delta_x

            elif V_write[i + 1] < 0 and V_write[i + 1] < self.v_on:
                delta_x = k_on * ((V_write[i + 1] / self.v_on - 1) ** self.alpha_on) * J1 * (
                        internal_state[i] ** P_on)
                internal_state[i + 1] = internal_state[i] + self.delta_t * delta_x

            else:
                delta_x = 0
                internal_state[i + 1] = internal_state[i]

            if internal_state[i + 1] < 0:
                internal_state[i + 1] = 0
            elif internal_state[i + 1] > 1:
                internal_state[i + 1] = 1

        # conductance calculation
        for i in range(points):
            conductance_fit[i] = self.G_off * internal_state[i] + self.G_on * (1 - internal_state[i])

        return internal_state, conductance_fit

    @timer
    def fitting(self):
        P_off_num = 20
        P_on_num = 20
        P_off_list = np.logspace(-1, 1, P_off_num, base=10)
        P_on_list = np.logspace(-1, 1, P_on_num, base=10)

        k_off_num = 200
        k_on_num = 200
        k_off_list = np.logspace(-4, 9, k_off_num, base=10)
        k_on_list = -np.logspace(-4, 9, k_on_num, base=10)

        # rise
        V_write_r = self.V_write[self.start_point_r: self.start_point_r + self.points_r]
        x_init_r = self.x_r[0]
        mem_x_r = np.zeros(self.points_r)
        mem_c_r = np.zeros(self.points_r)
        INDICATOR_r = np.ones([k_off_num, P_off_num])
        indicator_temp_r = 90
        min_x_r = 0
        min_y_r = 0

        for i in range(k_off_num):
            for j in range(P_off_num):
                mem_x_r, mem_c_r = self.Memristor_conductance_model(
                    k_off_list[i],
                    k_on_list[0],
                    P_off_list[j],
                    P_on_list[0],
                    x_init_r,
                    V_write_r
                )

                INDICATOR_r[i][j] = self.RRMSE_PERCENT(mem_c_r, self.conductance_r)
                if INDICATOR_r[i][j] <= indicator_temp_r:
                    min_x_r = i
                    min_y_r = j
                    indicator_temp_r = INDICATOR_r[i][j]

        self.k_off = k_off_list[min_x_r]
        self.P_off = P_off_list[min_y_r]

        # decline
        V_write_d = self.V_write[self.start_point_d: self.start_point_d + self.points_d]
        x_init_d = self.x_d[0]
        mem_x_d = np.zeros(self.points_d)
        mem_c_d = np.zeros(self.points_d)
        INDICATOR_d = np.ones([k_on_num, P_on_num])
        indicator_temp_d = 90
        min_x_d = 0
        min_y_d = 0

        for i in range(k_on_num):
            for j in range(P_on_num):
                mem_x_d, mem_c_d = self.Memristor_conductance_model(
                    k_off_list[0],
                    k_on_list[i],
                    P_off_list[0],
                    P_on_list[j],
                    x_init_d,
                    V_write_d
                )

                INDICATOR_d[i][j] = self.RRMSE_PERCENT(mem_c_d, self.conductance_d)
                if INDICATOR_d[i][j] <= indicator_temp_d:
                    min_x_d = i
                    min_y_d = j
                    indicator_temp_d = INDICATOR_d[i][j]

        self.k_on = k_on_list[min_x_d]
        self.P_on = P_on_list[min_y_d]

        return self.P_off, self.P_on, self.k_off, self.k_on

    @timer
    def c2c_fitting(self):
        V_write_r = self.V_write[self.start_point_r: self.start_point_r + self.points_r]
        x_init_r = self.x_r[0]
        V_write_d = self.V_write[self.start_point_d: self.start_point_d + self.points_d]
        x_init_d = self.x_d[0]
        
        best_mem_x_r, _ = self.Memristor_conductance_model(
            self.k_off, self.k_on, self.P_off, self.P_on, x_init_r, V_write_r
        )
        best_mem_x_d, _ = self.Memristor_conductance_model(
            self.k_off, self.k_on, self.P_off, self.P_on, x_init_d, V_write_d
        )

        best_memx_total = np.concatenate((best_mem_x_r, best_mem_x_d))
        x_total = np.concatenate((self.x_r, self.x_d))

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

        group_no = 10

        # Group with pulse number
        total_points = self.points_r + self.points_d
        every_points = math.ceil(total_points / group_no)

        x_mean = []
        var_x_median = []
        var_x_average = []
        for i in range(group_no):
            temp_x_mean = np.mean(x_sorted[every_points * i:(every_points * (i + 1))])
            temp_var_median = np.median(var_x_sorted[every_points * i:(every_points * (i + 1))])
            # TODO: Fit with average
            temp_var_average = np.mean(var_x_sorted[every_points * i:(every_points * (i + 1))])
            x_mean.append(temp_x_mean ** 2)
            var_x_median.append(temp_var_median ** 2)
            var_x_average.append(temp_var_average ** 2)

        z1 = np.polyfit(x_mean, var_x_median, 1)
        p1 = np.poly1d(z1)
        # print(p1)

        z2 = np.polyfit(x_mean, var_x_average, 1)
        p2 = np.poly1d(z2)
        # print(p2)

        self.x_mean = x_mean
        self.var_x_average = var_x_average
        self.memx_total = best_memx_total
        self.variation_x = variation_x

        sigma_relative = math.sqrt(z2[0] * math.pi / 2)
        sigma_absolute = math.sqrt(z2[1] * math.pi / 2)

        return sigma_relative, sigma_absolute
