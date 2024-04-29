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


class IVCurve(object):
    def __init__(
            self,
            file,
            dictionary: dict = {},
            **kwargs,
    ):
        """
            Fitting alpha with IV curve.
            :param file: IV curve data file
            :param dictionary: Memristor device parameters
        """
        # Read excel
        data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name=0,
            header=None,
            names=[
                'Time(s)',
                'Excitation Voltage(V)',
                'Current Response(A)'
            ]
        ))

        # Initialize parameters
        self.alpha_off = 1
        self.alpha_on = 1

        # Read parameters
        self.v_off = dictionary['v_off']
        self.v_on = dictionary['v_on']

        self.G_off = dictionary['G_off_fit']
        self.G_on = dictionary['G_on_fit']

        self.k_off = dictionary['k_off']
        self.k_on = dictionary['k_on']
        self.P_off = dictionary['P_off']
        self.P_on = dictionary['P_on']

        # Default parameters
        if None in [self.P_off, self.P_on]:
            self.P_off = 1
            self.P_on = 1

        # Read data
        self.voltage = np.array(data['Excitation Voltage(V)'])
        self.current = np.array(data['Current Response(A)'])
        self.delta_t = data['Time(s)'][1] - data['Time(s)'][0]

    def Memristor_conductance_model(
            self,
            alpha_off,
            alpha_on,
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
                delta_x = self.k_off * ((V_write[i + 1] / self.v_off - 1) ** alpha_off) * J1 * (
                        (1 - internal_state[i]) ** self.P_off)
                internal_state[i + 1] = internal_state[i] + self.delta_t * delta_x

            elif V_write[i + 1] < 0 and V_write[i + 1] < self.v_on:
                delta_x = self.k_on * ((V_write[i + 1] / self.v_on - 1) ** alpha_on) * J1 * (
                        internal_state[i] ** self.P_on)
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
        alpha_off_num = 10
        alpha_on_num = 10
        alpha_off_list = [i + 1 for i in range(alpha_off_num)]
        alpha_on_list = [i + 1 for i in range(alpha_on_num)]

        if None in [self.k_off, self.k_on]:
            k_off_num = 1000
            k_on_num = 1000
            k_off_list = np.logspace(-4, 9, k_off_num, base=10)
            k_on_list = -np.logspace(-4, 9, k_on_num, base=10)
        else:
            k_off_num = 1
            k_on_num = 1
            k_off_list = np.array(self.k_off)
            k_on_list = np.array(self.k_on)

        V_write = self.voltage
        points_r = np.sum(V_write > 0)
        V_write_r = V_write[:points_r]
        V_write_d = V_write[points_r:]
        current_r = self.current[:points_r]
        current_d = self.current[points_r:]

        x_init_r = (current_r[0] / self.voltage[0] - self.G_on) / (self.G_off - self.G_on)
        if x_init_r < 0:
            x_init_r = 0
        elif x_init_r > 1:
            x_init_r = 1
        x_init_d = (current_d[0] / self.voltage[points_r] - self.G_on) / (self.G_off - self.G_on)
        if x_init_d < 0:
            x_init_d = 0
        elif x_init_d > 1:
            x_init_d = 1

        INDICATOR_r = np.ones([k_off_num, alpha_off_num])
        INDICATOR_d = np.ones([k_on_num, alpha_on_num])
        indicator_temp_r = 90
        indicator_temp_d = 90
        min_x = 0
        min_y = 0

        # positive
        self.k_on = k_on_list[0]
        for i in range(k_off_num):
            for j in range(alpha_off_num):
                self.k_off = k_off_list[i]
                mem_x_r, mem_c_r = self.Memristor_conductance_model(
                    alpha_off_list[j],
                    alpha_on_list[0],
                    x_init_r,
                    V_write_r
                )
                current_fit_r = np.array(mem_c_r) * np.array(V_write_r)
                # RRMSE calculation
                i_diff = np.array(list(map(lambda x: x[0] - x[1], zip(current_fit_r, current_r))))
                INDICATOR_r[i][j] = np.sqrt(np.dot(i_diff, i_diff) / np.dot(current_r, current_r) / points_r)

                if INDICATOR_r[i][j] <= indicator_temp_r:
                    min_x = j
                    indicator_temp_r = INDICATOR_r[i][j]

        # negative
        self.k_off = k_off_list[0]
        for i in range(k_on_num):
            for j in range(alpha_on_num):
                self.k_on = k_on_list[i]
                mem_x_d, mem_c_d = self.Memristor_conductance_model(
                    alpha_off_list[0],
                    alpha_on_list[j],
                    x_init_d,
                    V_write_d
                )
                current_fit_d = np.array(mem_c_d) * np.array(V_write_d)
                # RRMSE calculation
                i_diff = np.array(list(map(lambda x: x[0] - x[1], zip(current_fit_d, current_d))))
                INDICATOR_d[i][j] = np.sqrt(np.dot(i_diff, i_diff) / np.dot(current_d, current_d) / points_r)

                if INDICATOR_d[i][j] <= indicator_temp_d:
                    min_y = j
                    indicator_temp_d = INDICATOR_d[i][j]

        self.alpha_off = alpha_off_list[min_x]
        self.alpha_on = alpha_on_list[min_y]

        return self.alpha_off, self.alpha_on
