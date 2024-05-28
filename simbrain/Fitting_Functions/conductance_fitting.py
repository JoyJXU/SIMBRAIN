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


class Conductance(object):
    def __init__(
            self,
            file,
            dictionary: dict = {},
            **kwargs,
    ):
        # Read excel
        self.data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name=0,
            header=None,
            index_col=None,
        ))
        self.data.columns = ['Pulse Voltage(V)', 'Read Voltage(V)'] + list(self.data.columns[2:] - 2)
        # Read parameters
        self.v_off = dictionary['v_off']
        self.v_on = dictionary['v_on']
        self.G_off = dictionary['G_off']
        self.G_on = dictionary['G_on']
        self.alpha_off = dictionary['alpha_off']
        self.alpha_on = dictionary['alpha_on']

        # Read data
        self.delta_t = dictionary['delta_t']
        self.duty_ratio = dictionary['duty_ratio']
        self.V_write = np.array(self.data['Pulse Voltage(V)'])
        self.read_voltage = np.array(self.data['Read Voltage(V)'][0])
        self.start_point_r = 0
        self.points_r = np.sum(self.V_write > 0)
        self.start_point_d = self.start_point_r + np.sum(self.V_write > 0)
        self.points_d = np.sum(self.V_write < 0)

        # Set batches
        self.device_nums = self.data.shape[1] - 2
        self.batch_size = 2
        self.batch_nums = int(self.device_nums / self.batch_size) + 1

        # Initialize parameters
        # self.P_off = torch.ones(self.device_nums)
        # self.P_on = torch.ones(self.device_nums)
        # self.k_off = torch.ones(self.device_nums)
        # self.k_on = -torch.ones(self.device_nums)
        # self.loss = torch.zeros(self.device_nums)
        # self.loss_index = torch.zeros(self.device_nums)

    @timer
    def fitting(self):
        """
        Calculate parameters k and P for the baseline model.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        J1 = 1
        k_off_num = 500
        k_off_list = torch.logspace(-3, 6, k_off_num, base=10)
        k_off_list = k_off_list.to(device)
        k_on_num = 500
        k_on_list = -torch.logspace(-3, 6, k_on_num, base=10)
        k_on_list = k_on_list.to(device)
        P_off_num = 1000
        P_off_list = torch.logspace(-5, 1, P_off_num, base=10)
        P_off_list = P_off_list.to(device)
        P_on_num = 1000
        P_on_list = torch.logspace(-5, 1, P_on_num, base=10)
        P_on_list = P_on_list.to(device)
        V_write_r = torch.tensor(self.V_write[self.start_point_r: self.start_point_r + self.points_r])
        V_write_r = V_write_r.to(device)
        V_write_d = torch.tensor(self.V_write[self.start_point_d: self.start_point_d + self.points_d])
        V_write_d = V_write_d.to(device)

        INDICATOR_r = torch.zeros([k_off_num, P_off_num])
        INDICATOR_r = INDICATOR_r.to(device)
        INDICATOR_d = torch.zeros([k_on_num, P_on_num])
        INDICATOR_d = INDICATOR_d.to(device)
        self.loss_r = torch.zeros([k_off_num, P_off_num])
        self.loss_r = self.loss_r.to(device)
        self.loss_d = torch.zeros([k_on_num, P_on_num])
        self.loss_d = self.loss_d.to(device)

        for batch_index in range(self.batch_nums):
            start_index = batch_index * self.batch_size
            if batch_index == self.batch_nums - 1:
                self.batch_size = self.device_nums % self.batch_size
            select_columns = np.array([i for i in range(self.batch_size)]) + start_index

            current_r = torch.tensor(
                np.array(self.data[select_columns])[self.start_point_r: self.start_point_r + self.points_r].T)
            current_d = torch.tensor(
                np.array(self.data[select_columns])[self.start_point_d: self.start_point_d + self.points_d].T)
            conductance_r = current_r / self.read_voltage
            conductance_d = current_d / self.read_voltage
            conductance_r = conductance_r.to(device)
            conductance_d = conductance_d.to(device)
            x_r = (conductance_r - self.G_on) / (self.G_off - self.G_on)
            x_d = (conductance_d - self.G_on) / (self.G_off - self.G_on)
            x_init_r = x_r[:, 0]
            x_init_d = x_d[:, 0]

            mem_x_r = torch.zeros([self.points_r, self.batch_size, k_off_num, P_off_num])
            mem_x_r = mem_x_r.to(device)
            mem_x_r[0] = x_init_r.expand(k_off_num, self.batch_size).expand(P_off_num, k_off_num, self.batch_size).permute(2, 1, 0)
            for i in range(self.points_r - 1):
                mem_x_r[i + 1] = torch.where(
                    V_write_r[i + 1] > self.v_off and V_write_r[i + 1] > 0,
                    k_off_list.expand(P_off_num, k_off_num).expand(self.batch_size, P_off_num, k_off_num).permute(0, 2, 1)
                    * ((V_write_r[i + 1] / self.v_off - 1) ** self.alpha_off)
                    * J1
                    * (1 - mem_x_r[i]) ** P_off_list
                    * self.delta_t
                    * self.duty_ratio
                    + mem_x_r[i],
                    mem_x_r[i]
                )
                mem_x_r[i + 1] = torch.where(mem_x_r[i + 1] < 0, 0, mem_x_r[i + 1])
                mem_x_r[i + 1] = torch.where(mem_x_r[i + 1] > 1, 1, mem_x_r[i + 1])
            mem_x_r_T = mem_x_r.permute(2, 3, 1, 0)
            mem_c_r = self.G_off * mem_x_r_T + self.G_on * (1 - mem_x_r_T)
            c_r_diff_percent = (mem_c_r - conductance_r) / conductance_r
            # INDICATOR_r = torch.sqrt(torch.sum(c_r_diff_percent * c_r_diff_percent, dim=3) / self.points_r).permute(2, 0, 1)
            # for i in range(self.batch_size):
            #     min_value = torch.min(INDICATOR_r[i])
            #     min_index = torch.argmin(INDICATOR_r[i])
            #     min_x_r = min_index // P_off_num
            #     min_y_r = min_index % P_off_num
            #     self.k_off[i + start_index] = k_off_list[min_x_r]
            #     self.P_off[i + start_index] = P_off_list[min_y_r]
            #     self.loss[i + start_index] += min_value
            #     self.loss_index[i + start_index] += min_index
            INDICATOR_r += torch.sum(torch.sum(c_r_diff_percent * c_r_diff_percent, dim=3) / self.points_r, dim=2)
            del mem_x_r, mem_x_r_T, mem_c_r, conductance_r, c_r_diff_percent
            torch.cuda.empty_cache()

            mem_x_d = torch.zeros([self.points_d, self.batch_size, k_on_num, P_on_num])
            mem_x_d[0] = x_init_d.expand(k_on_num, self.batch_size).expand(P_on_num, k_on_num, self.batch_size).permute(2, 1, 0)
            mem_x_d = mem_x_d.to(device)
            for i in range(self.points_d - 1):
                mem_x_d[i + 1] = torch.where(
                    V_write_d[i + 1] < 0 and V_write_d[i + 1] < self.v_on,
                    k_on_list.expand(P_on_num, k_on_num).expand(self.batch_size, P_on_num, k_on_num).permute(0, 2, 1)
                    * ((V_write_d[i + 1] / self.v_on - 1) ** self.alpha_on)
                    * J1
                    * mem_x_d[i] ** P_on_list
                    * self.delta_t
                    * self.duty_ratio
                    + mem_x_d[i],
                    mem_x_d[i]
                )
                mem_x_d[i + 1] = torch.where(mem_x_d[i + 1] < 0, 0, mem_x_d[i + 1])
                mem_x_d[i + 1] = torch.where(mem_x_d[i + 1] > 1, 1, mem_x_d[i + 1])
            mem_x_d_T = mem_x_d.permute(2, 3, 1, 0)
            mem_c_d = self.G_off * mem_x_d_T + self.G_on * (1 - mem_x_d_T)
            c_d_diff_percent = (mem_c_d - conductance_d) / conductance_d
            # INDICATOR_d = torch.sqrt(torch.sum(c_d_diff_percent * c_d_diff_percent, dim=3) / self.points_d).permute(2, 0, 1)
            # for i in range(self.batch_size):
            #     min_value = torch.min(INDICATOR_d[i])
            #     min_index = torch.argmin(INDICATOR_d[i])
            #     min_x_d = min_index // P_on_num
            #     min_y_d = min_index % P_on_num
            #     self.k_on[i + start_index] = k_on_list[min_x_d]
            #     self.P_on[i + start_index] = P_on_list[min_y_d]
            #     self.loss[i + start_index] += min_value
            #     self.loss_index[i + start_index] += min_index
            # print(self.loss)
            INDICATOR_d += torch.sum(torch.sum(c_d_diff_percent * c_d_diff_percent, dim=3) / self.points_d, dim=2)
            del mem_x_d, mem_x_d_T, mem_c_d, conductance_d, c_d_diff_percent
            torch.cuda.empty_cache()

        self.loss_r += torch.sqrt(INDICATOR_r / self.device_nums)
        self.loss_d += torch.sqrt(INDICATOR_d / self.device_nums)
        self.loss = torch.min(torch.sqrt((torch.min(INDICATOR_r) + torch.min(INDICATOR_d)) / 2 / self.device_nums))
        min_x_r = (torch.argmin(self.loss_r) // P_off_num).item()
        min_y_r = (torch.argmin(self.loss_r) % P_off_num).item()
        min_x_d = (torch.argmin(self.loss_d) // P_on_num).item()
        min_y_d = (torch.argmin(self.loss_d) % P_on_num).item()
        # print(min_x_r, min_y_r, min_x_d, min_y_d)
        # print(torch.min(self.loss_r), torch.min(self.loss_d), torch.min(self.loss))
        self.k_off = k_off_list[min_x_r].item()
        self.k_on = k_on_list[min_x_d].item()
        self.P_off = P_off_list[min_y_r].item()
        self.P_on = P_on_list[min_y_d].item()

        del k_off_list, k_on_list, P_off_list, P_on_list, V_write_r, V_write_d, INDICATOR_r, INDICATOR_d
        torch.cuda.empty_cache()

        return self.P_off, self.P_on, self.k_off, self.k_on, self.V_write[0]

    @timer
    def mult_P_fitting(self, G_off_variation: np.array, G_on_variation: np.array):
        """
        Calculate the list of P from multiple devices for variation fitting.

        :param G_off_variation: The list of G_off from multiple devices
        :param G_on_variation: The list of G_on from multiple devices
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        G_off_variation = torch.tensor(G_off_variation)
        G_off_variation = G_off_variation.to(device)
        G_on_variation = torch.tensor(G_on_variation)
        G_on_variation = G_on_variation.to(device)
        P_off_variation = np.zeros(self.device_nums)
        P_on_variation = np.zeros(self.device_nums)

        P_off_nums = 1000
        P_on_nums = 1000
        # P_off_unit = 1 / P_off_nums
        # P_on_unit = 1 / P_on_nums
        # P_off_list = torch.linspace(
        #     round(self.P_off, 2) - 49 * P_off_unit,
        #     round(self.P_off, 2) + 50 * P_off_unit,
        #     P_off_nums
        # )
        # P_on_list = torch.linspace(
        #     round(self.P_on, 2) - 49 * P_on_unit,
        #     round(self.P_on, 2) + 50 * P_on_unit,
        #     P_on_nums
        # )
        #
        # for i in range(20):
        #     P_off_list[P_off_list < 0] = 0
        #     if P_off_list[0] <= self.P_off / 2:
        #         P_off_unit = P_off_unit / 2
        #         P_off_list = torch.linspace(round(self.P_off, 2) - 49 * P_off_unit,
        #                                     round(self.P_off, 2) + 50 * P_off_unit, P_off_nums)
        #     P_on_list[P_on_list < 0] = 0
        #     if P_on_list[0] <= self.P_on / 2:
        #         P_on_unit = P_on_unit / 2
        #         P_on_list = torch.linspace(round(self.P_on, 2) - 49 * P_on_unit,
        #                                    round(self.P_on, 2) + 50 * P_on_unit, P_on_nums)
        P_off_list = torch.logspace(-4, 1, P_off_nums, base=10)
        P_on_list = torch.logspace(-4, 1, P_on_nums, base=10)
        P_off_list = P_off_list.to(device)
        P_on_list = P_on_list.to(device)

        J1 = 1
        select_columns = np.array([i for i in range(self.device_nums)])
        current_r = torch.tensor(
            np.array(self.data[select_columns])[self.start_point_r: self.start_point_r + self.points_r].T)
        current_d = torch.tensor(
            np.array(self.data[select_columns])[self.start_point_d: self.start_point_d + self.points_d].T)
        conductance_r = current_r / self.read_voltage
        conductance_d = current_d / self.read_voltage
        conductance_r = conductance_r.to(device)
        conductance_d = conductance_d.to(device)
        # x_r = (conductance_r - self.G_on) / (self.G_off - self.G_on)
        # x_d = (conductance_d - self.G_on) / (self.G_off - self.G_on)
        x_r = (
                (conductance_r - G_on_variation.expand(self.points_r, self.device_nums).T)
                / (G_off_variation.expand(self.points_r, self.device_nums).T - G_on_variation.expand(self.points_r, self.device_nums).T)
        )
        x_d = (
                (conductance_d - G_on_variation.expand(self.points_r, self.device_nums).T)
                / (G_off_variation.expand(self.points_r, self.device_nums).T - G_on_variation.expand(self.points_r, self.device_nums).T)
        )
        V_write_r = torch.tensor(self.V_write[self.start_point_r: self.start_point_r + self.points_r])
        V_write_d = torch.tensor(self.V_write[self.start_point_d: self.start_point_d + self.points_d])
        V_write_r = V_write_r.to(device)
        V_write_d = V_write_d.to(device)
        x_init_r = x_r[:, 0]
        x_init_d = x_d[:, 0]

        mem_x_r = torch.zeros([self.points_r, self.device_nums, P_off_nums])
        mem_x_r[0] = x_init_r.expand(P_off_nums, self.device_nums).T
        mem_x_r = mem_x_r.to(device)
        for j in range(self.points_r - 1):
            mem_x_r[j + 1] = torch.where(
                V_write_r[j + 1] > self.v_off and V_write_r[j + 1] > 0,
                self.k_off
                * ((V_write_r[j + 1] / self.v_off - 1) ** self.alpha_off)
                * J1
                * (1 - mem_x_r[j]) ** P_off_list
                * self.delta_t
                * self.duty_ratio
                + mem_x_r[j],
                mem_x_r[j]
            )
            mem_x_r[j + 1] = torch.where(mem_x_r[j + 1] < 0, 0, mem_x_r[j + 1])
            mem_x_r[j + 1] = torch.where(mem_x_r[j + 1] > 1, 1, mem_x_r[j + 1])
        mem_x_r_T = mem_x_r.permute(2, 1, 0)
        # mem_c_r = self.G_off * mem_x_r_T + self.G_on * (1 - mem_x_r_T)
        mem_c_r = (
                G_off_variation.expand(P_off_nums, self.device_nums).expand(self.points_r, P_off_nums, self.device_nums).permute(1, 2, 0)
                * mem_x_r_T
                + G_on_variation.expand(P_on_nums, self.device_nums).expand(self.points_d, P_on_nums, self.device_nums).permute(1, 2, 0)
                * (1 - mem_x_r_T)
        )
        c_r_diff_percent = (mem_c_r - conductance_r) / conductance_r
        INDICATOR_r = torch.sqrt(torch.sum(c_r_diff_percent * c_r_diff_percent, dim=2) / self.points_r).T
        P_off_variation = P_off_list[torch.argmin(INDICATOR_r, dim=1)]
        P_off_variation = P_off_variation.cpu()

        mem_x_d = torch.zeros([self.points_d, self.device_nums, P_on_nums])
        mem_x_d[0] = x_init_d.expand(P_on_nums, self.device_nums).T
        mem_x_d = mem_x_d.to(device)
        for j in range(self.points_d - 1):
            mem_x_d[j + 1] = torch.where(
                V_write_d[j + 1] < 0 and V_write_d[j + 1] < self.v_on,
                self.k_on
                * ((V_write_d[j + 1] / self.v_on - 1) ** self.alpha_on)
                * J1
                * mem_x_d[j] ** P_on_list
                * self.delta_t
                * self.duty_ratio
                + mem_x_d[j],
                mem_x_d[j]
            )
            mem_x_d[j + 1] = torch.where(mem_x_d[j + 1] < 0, 0, mem_x_d[j + 1])
            mem_x_d[j + 1] = torch.where(mem_x_d[j + 1] > 1, 1, mem_x_d[j + 1])
        mem_x_d_T = mem_x_d.permute(2, 1, 0)
        # mem_c_d = self.G_off * mem_x_d_T + self.G_on * (1 - mem_x_d_T)
        mem_c_d = (
                G_off_variation.expand(P_off_nums, self.device_nums).expand(self.points_r, P_off_nums,
                                                                            self.device_nums).permute(1, 2, 0)
                * mem_x_d_T
                + G_on_variation.expand(P_on_nums, self.device_nums).expand(self.points_d, P_on_nums,
                                                                            self.device_nums).permute(1, 2, 0)
                * (1 - mem_x_d_T)
        )
        c_d_diff_percent = (mem_c_d - conductance_d) / conductance_d
        INDICATOR_d = torch.sqrt(torch.sum(c_d_diff_percent * c_d_diff_percent, dim=2) / self.points_d).T
        P_on_variation = P_on_list[torch.argmin(INDICATOR_d, dim=1)]
        P_on_variation = P_on_variation.cpu()
        # print(P_off_variation)
        # print(P_on_variation)

        torch.cuda.empty_cache()

        return P_off_variation.numpy(), P_on_variation.numpy()
