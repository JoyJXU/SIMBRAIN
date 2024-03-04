import numpy as np
import pandas as pd


class IVCurve(object):
    def __init__(
            self,
            file,
            dictionary: dict = {}
    ):
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
        self.G_off = dictionary['G_off']
        self.G_on = dictionary['G_on']
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

    def fitting(self):
        alpha_off_num = 10
        alpha_on_num = 10
        alpha_off_list = [i+1 for i in range(10)]
        alpha_on_list = [i+1 for i in range(10)]

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
        points = len(V_write)
        V_write_p = V_write[:int(points / 2)]
        V_write_n = V_write[int(points / 2):]
        current_p = self.current[:int(points / 2)]
        current_n = self.current[int(points / 2):]

        x_init_p = (current_p[0] / self.voltage[0] - self.G_on) / (self.G_off - self.G_on)
        if x_init_p < 0:
            x_init_p = 0
        elif x_init_p > 1:
            x_init_p = 1
        x_init_n = (current_n[0] / self.voltage[int(points / 2)] - self.G_on) / (self.G_off - self.G_on)
        if x_init_n < 0:
            x_init_n = 0
        elif x_init_n > 1:
            x_init_n = 1

        INDICATOR_p = np.ones([k_off_num, alpha_off_num])
        INDICATOR_n = np.ones([k_on_num, alpha_on_num])
        indicator_temp_p = 90
        indicator_temp_n = 90
        min_x = 0
        min_y = 0

        # positive
        self.k_on = k_on_list[0]
        for i in range(k_off_num):
            for j in range(alpha_off_num):
                self.k_off = k_off_list[i]
                mem_x_p, mem_c_p = self.Memristor_conductance_model(
                    alpha_off_list[j],
                    alpha_on_list[0],
                    x_init_p,
                    V_write_p
                )
                current_fit_p = np.array(mem_c_p) * np.array(V_write_p)
                # RRMSE calculation
                i_diff = np.array(list(map(lambda x: x[0] - x[1], zip(current_fit_p, current_p))))
                INDICATOR_p[i][j] = np.sqrt(np.dot(i_diff, i_diff) / np.dot(current_p, current_p) / int(points / 2))

                if INDICATOR_p[i][j] <= indicator_temp_p:
                    min_x = j
                    # k_off_best = self.k_off
                    indicator_temp_p = INDICATOR_p[i][j]
                    print(indicator_temp_p)

        # negative
        self.k_off = k_off_list[0]
        for i in range(k_on_num):
            for j in range(alpha_on_num):
                self.k_on = k_on_list[i]
                mem_x_n, mem_c_n = self.Memristor_conductance_model(
                    alpha_off_list[0],
                    alpha_on_list[j],
                    x_init_n,
                    V_write_n
                )
                current_fit_n = np.array(mem_c_n) * np.array(V_write_n)
                # RRMSE calculation
                i_diff = np.array(list(map(lambda x: x[0] - x[1], zip(current_fit_n, current_n))))
                INDICATOR_n[i][j] = np.sqrt(np.dot(i_diff, i_diff) / np.dot(current_n, current_n) / int(points / 2))

                if INDICATOR_n[i][j] <= indicator_temp_n:
                    min_y = j
                    # k_on_best = self.k_on
                    indicator_temp_n = INDICATOR_n[i][j]
                    print(indicator_temp_n)

        self.alpha_off = alpha_off_list[min_x]
        self.alpha_on = alpha_on_list[min_y]

        return self.alpha_off, self.alpha_on
