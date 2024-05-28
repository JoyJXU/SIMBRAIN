import numpy as np
import pandas as pd
import torch


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

        self.G_off = dictionary['G_off']
        self.G_on = dictionary['G_on']

        self.k_off = dictionary['k_off']
        self.k_on = dictionary['k_on']
        self.P_off = dictionary['P_off']
        self.P_on = dictionary['P_on']

        self.duty_ratio = dictionary['duty_ratio']

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
                internal_state[i + 1] = internal_state[i] + self.delta_t * self.duty_ratio * delta_x

            elif V_write[i + 1] < 0 and V_write[i + 1] < self.v_on:
                delta_x = self.k_on * ((V_write[i + 1] / self.v_on - 1) ** alpha_on) * J1 * (
                        internal_state[i] ** self.P_on)
                internal_state[i + 1] = internal_state[i] + self.delta_t * self.duty_ratio * delta_x

            else:
                delta_x = 0
                internal_state[i + 1] = internal_state[i]

            if internal_state[i + 1] < 0:
                internal_state[i + 1] = 0
            elif internal_state[i + 1] > 1:
                internal_state[i + 1] = 1
        # print(internal_state)

        # conductance calculation
        for i in range(points):
            conductance_fit[i] = self.G_off * internal_state[i] + self.G_on * (1 - internal_state[i])

        return internal_state, conductance_fit

    @timer
    def fitting(self):
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        alpha_off_num = 10
        alpha_on_num = 10
        alpha_off_list = torch.arange(alpha_off_num) + 1
        alpha_on_list = torch.arange(alpha_on_num) + 1
        alpha_off_list = alpha_off_list.to(device)
        alpha_on_list = alpha_on_list.to(device)

        if None in [self.k_off, self.k_on]:
            k_off_num = 1000
            k_on_num = 1000
            k_off_list = torch.logspace(-4, 9, k_off_num, base=10)
            k_on_list = -torch.logspace(-4, 9, k_on_num, base=10)
        else:
            k_off_num = 1
            k_on_num = 1
            k_off_list = torch.tensor(self.k_off)
            k_on_list = torch.tensor(self.k_on)
        k_off_list = k_off_list.to(device)
        k_on_list = k_on_list.to(device)

        V_write = self.voltage
        # start_point_r = 0
        # points_r = np.sum(V_write > 0)
        # start_point_d = start_point_r + points_r
        # points_d = np.sum(V_write < 0)
        if V_write[0] > 0:
            start_point_r = 0
            points_r = np.sum(V_write > 0)
            start_point_d = start_point_r + points_r
            points_d = np.sum(V_write < 0)
        else:
            start_point_d = 0
            points_d = np.sum(V_write < 0)
            start_point_r = start_point_d + points_d
            points_r = np.sum(V_write > 0)

        V_write_r = torch.tensor(V_write[start_point_r: start_point_r + points_r])
        V_write_d = torch.tensor(V_write[start_point_d: start_point_d + points_d])
        V_write_r = V_write_r.to(device)
        V_write_d = V_write_d.to(device)
        current_r = torch.tensor(self.current[start_point_r: start_point_r + points_r])
        current_d = torch.tensor(self.current[start_point_d: start_point_d + points_d])

        x_init_r = (current_r[0] / self.voltage[0] - self.G_on) / (self.G_off - self.G_on)
        x_init_r = x_init_r if x_init_r > 0 else 0
        x_init_r = x_init_r if x_init_r < 1 else 1
        x_init_d = (current_d[0] / self.voltage[points_r] - self.G_on) / (self.G_off - self.G_on)
        x_init_d = x_init_d if x_init_d > 0 else 0
        x_init_d = x_init_d if x_init_d < 1 else 1

        J1 = 1

        # positive
        mem_x_r = torch.zeros([points_r, alpha_off_num, k_off_num])
        mem_x_r = mem_x_r.to(device)
        mem_x_r[0] = x_init_r * torch.ones([alpha_off_num, k_off_num])
        for j in range(points_r - 1):
            mem_x_r[j + 1] = torch.where(
                V_write_r[j + 1] > self.v_off and V_write_r[j + 1] > 0,
                k_off_list.expand(alpha_off_num, k_off_num)
                * ((V_write_r[j + 1] / self.v_off - 1) ** alpha_off_list.expand(k_off_num, alpha_off_num).T)
                * J1
                * (1 - mem_x_r[j]) ** self.P_off
                * self.delta_t
                * self.duty_ratio
                + mem_x_r[j],
                mem_x_r[j]
            )
            mem_x_r[j + 1] = torch.where(mem_x_r[j + 1] < 0, 0, mem_x_r[j + 1])
            mem_x_r[j + 1] = torch.where(mem_x_r[j + 1] > 1, 1, mem_x_r[j + 1])

        mem_x_r_T = mem_x_r.permute(1, 2, 0)
        mem_c_r = self.G_off * mem_x_r_T + self.G_on * (1 - mem_x_r_T)
        current_fit_r = mem_c_r * V_write_r
        current_r = current_r.to(device)
        # RRMSE calculation
        i_diff_percent = (current_fit_r - current_r) / current_r
        INDICATOR_r = torch.sqrt(
            torch.sum(i_diff_percent * i_diff_percent, dim=2)
            / torch.dot(current_r, current_r)
            / points_r
        )

        # negative
        mem_x_d = torch.zeros([points_d, alpha_on_num, k_on_num])
        mem_x_d = mem_x_d.to(device)
        mem_x_d[0] = x_init_d * torch.ones([alpha_on_num, k_on_num])
        for j in range(points_d - 1):
            mem_x_d[j + 1] = torch.where(
                V_write_d[j + 1] < 0 and V_write_d[j + 1] < self.v_on,
                k_on_list.expand(alpha_on_num, k_on_num)
                * ((V_write_d[j + 1] / self.v_on - 1) ** alpha_on_list.expand(k_on_num, alpha_on_num).T)
                * J1
                * (1 - mem_x_d[j]) ** self.P_on
                * self.delta_t
                * self.duty_ratio
                + mem_x_d[j],
                mem_x_d[j],
            )
            mem_x_d[j + 1] = torch.where(mem_x_d[j + 1] < 0, 0, mem_x_d[j + 1])
            mem_x_d[j + 1] = torch.where(mem_x_d[j + 1] > 1, 1, mem_x_d[j + 1])

        mem_x_d_T = mem_x_d.permute(1, 2, 0)
        mem_c_d = self.G_off * mem_x_d_T + self.G_on * (1 - mem_x_d_T)
        current_fit_d = mem_c_d * V_write_d
        current_d = current_d.to(device)
        # RRMSE calculation
        i_diff_percent = (current_fit_d - current_d) / current_d
        INDICATOR_d = torch.sqrt(
            torch.sum(i_diff_percent * i_diff_percent, dim=2)
            / torch.dot(current_d, current_d)
            / points_r
        )

        min_x = torch.argmin(INDICATOR_r)
        min_y = torch.argmin(INDICATOR_d)
        self.alpha_off = alpha_off_list[min_x // k_off_num].item()
        self.alpha_on = alpha_on_list[min_y // k_on_num].item()
        self.k_off = k_off_list[min_x % k_off_num].item()
        self.k_on = k_on_list[min_y % k_on_num].item()

        torch.cuda.empty_cache()

        return self.alpha_off, self.alpha_on
