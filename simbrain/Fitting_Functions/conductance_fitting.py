import numpy as np
import pandas as pd


class Conductance(object):
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
                'Pulse Voltage(V)',
                'Current(A)',
                'Read Voltage(V)'
            ]
        ))
        print(data)

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
        self.read_voltage = data['Read Voltage(V)'][0]

        self.start_point_r = 0
        self.points_r = np.where(self.V_write < 0)[0][0]
        self.start_point_d = self.points_r
        self.points_d = np.size(self.V_write) - self.points_r

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

    def fitting(self):
        P_off_num = 50
        P_on_num = 50
        P_off_list = np.logspace(-1, 1, P_off_num, base=10)
        P_on_list = np.logspace(-1, 1, P_on_num, base=10)

        k_off_num = 1000
        k_on_num = 1000
        k_off_list = np.logspace(-2, 10, k_off_num, base=10)
        k_on_list = -np.logspace(-2, 10, k_on_num, base=10)

        # rise
        V_write_r_temp = self.V_write[self.start_point_r: self.start_point_r + self.points_r]
        # V_write_r = V_write_r_temp
        V_write_r = V_write_r_temp[0:500:5]
        x_init_r = self.x_r[0]
        # mem_x_r = np.zeros(self.points_r)
        # mem_c_r = np.zeros(self.points_r)
        mem_x_r = np.zeros(self.points_r // 5)
        mem_c_r = np.zeros(self.points_r // 5)
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
                # mem_x_r_repeat = mem_x_r
                # mem_c_r_repeat = mem_c_r
                mem_x_r_repeat = [val for val in mem_x_r for i in range(5)]
                mem_c_r_repeat = [val for val in mem_c_r for i in range(5)]

                INDICATOR_r[i][j] = self.RRMSE_PERCENT(mem_c_r_repeat, self.conductance_r)
                if INDICATOR_r[i][j] <= indicator_temp_r:
                    min_x_r = i
                    min_y_r = j
                    indicator_temp_r = INDICATOR_r[i][j]

        self.k_off = k_off_list[min_x_r]
        self.P_off = P_off_list[min_y_r]

        # decline
        V_write_d_temp = self.V_write[self.start_point_d: self.start_point_d + self.points_d]
        # V_write_d = V_write_d_temp
        V_write_d = V_write_d_temp[0:500:5]
        x_init_d = self.x_d[0]
        # mem_x_d = np.zeros(self.points_d)
        # mem_c_d = np.zeros(self.points_d)
        mem_x_d = np.zeros(self.points_d // 5)
        mem_c_d = np.zeros(self.points_d // 5)
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
                # mem_x_d_repeat = mem_x_d
                # mem_c_d_repeat = mem_c_d
                mem_x_d_repeat = [val for val in mem_x_d for i in range(5)]
                mem_c_d_repeat = [val for val in mem_c_d for i in range(5)]

                INDICATOR_d[i][j] = self.RRMSE_PERCENT(mem_c_d_repeat, self.conductance_d)
                if INDICATOR_d[i][j] <= indicator_temp_d:
                    min_x_d = i
                    min_y_d = j
                    indicator_temp_d = INDICATOR_d[i][j]
                # print(INDICATOR_d[i][j])
                # print(indicator_temp_d)

        self.k_on = k_on_list[min_x_d]
        self.P_on = P_on_list[min_y_d]

        return self.P_off, self.P_on, self.k_off, self.k_on


def main():
    exp = Conductance(
        "../../memristordata/conductance_ferro.xlsx",
        {
            'G_on': 7e-8, 'G_off': 9e-6,
            'v_on': -2, 'v_off': 1.4,
            'k_off': None, 'k_on': None,
            'P_off': None, 'P_on': None,
            'alpha_off': 3, 'alpha_on': 4,
            'delta_t': 1e-7, 'read_voltage': 0.1
        }
    )
    P_off, P_on, k_off, k_on = exp.fitting()
    print('P_off:{}\nP_on:{}\nk_off:{}\nk_on:{}'.format(P_off, P_on, k_off, k_on))

    return 0


if __name__ == '__main__':
    main()
