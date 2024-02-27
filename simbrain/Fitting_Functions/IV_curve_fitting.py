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
        # print(data)

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
        if None in [self.k_off, self.k_on]:
            self.k_off = 1  # 1000
            self.k_on = -1  # -10
        if None in [self.P_off, self.P_on]:
            self.P_off = 1
            self.P_on = 1

        # Read data
        self.read_voltage = dictionary['read_voltage']
        self.delta_t = data['Time(s)'][1] - data['Time(s)'][0]
        self.voltage = np.array(data['Excitation Voltage(V)'])
        self.current = np.array(data['Current Response(A)'])

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
        points = len(conductance_fit)
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
        alpha_off_list = [i+1 for i in range(10)]
        alpha_on_list = [i+1 for i in range(10)]
        alpha_off_num = 10
        alpha_on_num = 10

        V_write = self.voltage
        points = len(V_write)
        x_init = (self.current[0] / self.read_voltage - self.G_on) / (self.G_off - self.G_on)
        if x_init < 0:
            x_init = 0
        elif x_init > 1:
            x_init = 1

        INDICATOR = [[1 for col in range(alpha_on_num)] for row in range(alpha_off_num)]
        indicator_temp = 90
        min_x = 0
        min_y = 0

        for i in range(alpha_off_num):
            for j in range(alpha_on_num):
                mem_x, mem_c = self.Memristor_conductance_model(
                    alpha_off_list[i],
                    alpha_on_list[j],
                    x_init,
                    self.voltage
                )
                current_fit = np.array(mem_c) * np.array(self.voltage)
                # RRMSE calculation
                i_diff = list(map(lambda x: x[0] - x[1], zip(current_fit, self.current)))
                INDICATOR[i][j] = np.sqrt(np.dot(i_diff, i_diff) / np.dot(self.current, self.current) / points)

                if INDICATOR[i][j] <= indicator_temp:
                    min_x = i
                    min_y = j
                    indicator_temp = INDICATOR[i][j]

        self.alpha_off = alpha_off_list[min_x]
        self.alpha_on = alpha_on_list[min_y]

        return self.alpha_off, self.alpha_on


def main():
    dict_f = {
        'G_on': 7.0e-8, 'G_off': 9.0e-6,
        'v_on': -2, 'v_off': 1.4,
        'k_off': 0.8, 'k_on': -42,
        'P_off': None, 'P_on': None,
        'read_voltage': 0.1
    }
    dict_h = {
        'G_on': 2.5e-10, 'G_off': 1.9e-9,
        'v_on': -2, 'v_off': 2,
        'k_off': None, 'k_on': None,
        'P_off': None, 'P_on': None,
        'read_voltage': 0.5
    }
    exp = IVCurve(
        "../../memristordata/IV_curve_ferro.xlsx",
        dict_f
    )
    alpha_off, alpha_on = exp.fitting()
    print('alpha_off:{}\nalpha_on:{}'.format(alpha_off, alpha_on))
    return 0


if __name__ == '__main__':
    main()
