import os
import gc
import psutil
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

        # Set default batches
        self.device_nums = self.data.shape[1] - 2
        self.batch_size = np.int64(32)
        self.batch_nums = int(self.device_nums / self.batch_size) + 1

        # Initialize parameters
        self.k_off = 1
        self.k_on = -1
        self.P_off = 1
        self.P_on = 1
        self.k_off_devices = torch.ones(self.device_nums)
        self.k_on_devices = torch.ones(self.device_nums)
        self.P_off_devices = torch.ones(self.device_nums)
        self.P_on_devices = torch.ones(self.device_nums)
        self.loss = 1

    def set_batch_size(self, k_off_nums, k_on_nums, P_off_nums, P_on_nums):
        # Modify the batch size according to your system memory size
        available_mem = np.array(psutil.virtual_memory().available, dtype=np.int64)
        mem_cost = self.batch_size * (self.points_r * k_off_nums * P_off_nums + self.points_d * k_on_nums * P_on_nums) * 2
        while mem_cost * 2 >= available_mem:
            self.batch_size = np.int64(self.batch_size / 2)
            self.batch_nums = int(self.device_nums / self.batch_size) + 1
            mem_cost = self.batch_size * (self.points_r * k_off_nums * P_off_nums + self.points_d * k_on_nums * P_on_nums) * 2
            if self.batch_size == 1 and mem_cost * 2 >= available_mem:
                raise Exception("Error! Out of memory!")

        # Modify the batch size according to your VRAM size
        if torch.cuda.is_available():
            total_mem = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
            while mem_cost * 2 >= total_mem:
                self.batch_size = np.int64(self.batch_size / 2)
                self.batch_nums = int(self.device_nums / self.batch_size) + 1
                mem_cost = self.batch_size * (self.points_r * k_off_nums * P_off_nums + self.points_d * k_on_nums * P_on_nums) * 2
                if self.batch_size == 1 and mem_cost * 2 >= available_mem:
                    raise Exception("Error! Out of memory!")

    @timer
    def fitting(self):
        """
        Calculate parameters k and P for the baseline model.
        """

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        # Set parameters' range
        J1 = 1
        k_off_nums = 500
        k_on_nums = 500
        k_off_list = torch.logspace(-3, 6, k_off_nums, base=10, dtype=torch.float64)
        k_on_list = -torch.logspace(-3, 6, k_on_nums, base=10, dtype=torch.float64)
        k_off_list = k_off_list.to(device)
        k_on_list = k_on_list.to(device)
        P_off_nums = 1000
        P_on_nums = 1000
        # P_off_list = torch.logspace(-5, 1, P_off_nums, base=10, dtype=torch.float64)
        # P_on_list = torch.logspace(-5, 1, P_on_nums, base=10, dtype=torch.float64)
        P_off_list = torch.linspace(0, 10, P_off_nums, dtype=torch.float64)
        P_on_list = torch.linspace(0, 10, P_on_nums, dtype=torch.float64)
        P_off_list = P_off_list.to(device)
        P_on_list = P_on_list.to(device)
        V_write_r = torch.from_numpy(self.V_write[self.start_point_r: self.start_point_r + self.points_r])
        V_write_d = torch.from_numpy(self.V_write[self.start_point_d: self.start_point_d + self.points_d])
        V_write_r = V_write_r.to(device)
        V_write_d = V_write_d.to(device)

        # Initialize loss
        INDICATOR_r = torch.zeros([k_off_nums, P_off_nums])
        INDICATOR_d = torch.zeros([k_on_nums, P_on_nums])
        self.loss_r = torch.zeros([k_off_nums, P_off_nums])
        self.loss_d = torch.zeros([k_on_nums, P_on_nums])

        # Set batch size
        print('target batch size: {}'.format(self.batch_size))
        self.set_batch_size(k_off_nums, k_on_nums, P_off_nums, P_on_nums)
        print('actual batch size: {}'.format(self.batch_size))

        for batch_index in range(self.batch_nums):
            start_index = batch_index * self.batch_size
            if batch_index == self.batch_nums - 1:
                self.batch_size = self.device_nums % self.batch_size
            select_columns = np.array([i for i in range(self.batch_size)]) + start_index

            current_r = torch.tensor(
                np.array(self.data[select_columns])[self.start_point_r: self.start_point_r + self.points_r].T,
                dtype=torch.float32
            )
            current_d = torch.tensor(
                np.array(self.data[select_columns])[self.start_point_d: self.start_point_d + self.points_d].T,
                dtype=torch.float32
            )
            conductance_r = current_r / self.read_voltage
            conductance_d = current_d / self.read_voltage
            x_r = (conductance_r - self.G_on) / (self.G_off - self.G_on)
            x_d = (conductance_d - self.G_on) / (self.G_off - self.G_on)
            x_init_r = x_r[:, 0]
            x_init_d = x_d[:, 0]

            mem_x_r = torch.zeros([self.points_r, self.batch_size, k_off_nums, P_off_nums], device=device)
            mem_x_r[0] = x_init_r.expand(k_off_nums, self.batch_size).expand(P_off_nums, k_off_nums,
                                                                             self.batch_size).permute(2, 1, 0)
            for i in range(self.points_r - 1):
                mem_x_r[i + 1] = torch.where(
                    V_write_r[i + 1] > self.v_off and V_write_r[i + 1] > 0,
                    k_off_list.expand(P_off_nums, k_off_nums).expand(self.batch_size, P_off_nums, k_off_nums).permute(0, 2, 1)
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
            mem_x_r_T = mem_x_r.permute(2, 3, 1, 0).cpu()
            # mem_c_r = self.G_off * mem_x_r_T + self.G_on * (1 - mem_x_r_T)
            del mem_x_r
            # TODO: Add a fitting indicator: diff / diff_percent
            # c_r_diff_percent = (mem_c_r - conductance_r) / conductance_r
            # c_r_diff = (mem_c_r - conductance_r)
            x_r_diff = mem_x_r_T - x_r

            INDICATOR_r_i = torch.sqrt(torch.sum(x_r_diff * x_r_diff, dim=3) / self.points_r).permute(2, 0, 1)
            for i in range(self.batch_size):
                min_index = torch.argmin(INDICATOR_r_i[i])
                min_x_r = min_index // P_off_nums
                min_y_r = min_index % P_off_nums
                self.k_off_devices[start_index + i] = k_off_list[min_x_r]
                self.P_off_devices[start_index + i] = P_off_list[min_y_r]
            INDICATOR_r += torch.sum(torch.sum(x_r_diff * x_r_diff, dim=3) / (self.points_r + self.points_d), dim=2)
            del mem_x_r_T, x_r_diff, INDICATOR_r_i
            torch.cuda.empty_cache()
            gc.collect()

            mem_x_d = torch.zeros([self.points_d, self.batch_size, k_on_nums, P_on_nums], device=device)
            mem_x_d[0] = x_init_d.expand(k_on_nums, self.batch_size).expand(P_on_nums, k_on_nums,
                                                                            self.batch_size).permute(2, 1, 0)
            mem_x_d = mem_x_d.to(device)
            for i in range(self.points_d - 1):
                mem_x_d[i + 1] = torch.where(
                    V_write_d[i + 1] < 0 and V_write_d[i + 1] < self.v_on,
                    k_on_list.expand(P_on_nums, k_on_nums).expand(self.batch_size, P_on_nums, k_on_nums).permute(0, 2, 1)
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
            mem_x_d_T = mem_x_d.permute(2, 3, 1, 0).cpu()
            # mem_c_d = self.G_off * mem_x_d_T + self.G_on * (1 - mem_x_d_T)
            del mem_x_d
            # RRMSE
            # c_d_diff_percent = (mem_c_d - conductance_d) / conductance_d
            # RMSE
            # c_d_diff = (mem_c_d - conductance_d)
            x_d_diff = mem_x_d_T - x_d

            INDICATOR_d_i = torch.sqrt(torch.sum(x_d_diff * x_d_diff, dim=3) / self.points_d).permute(2, 0, 1)
            for i in range(self.batch_size):
                min_index = torch.argmin(INDICATOR_d_i[i])
                min_x_r = min_index // P_on_nums
                min_y_r = min_index % P_on_nums
                self.k_on_devices[start_index + i] = k_on_list[min_x_r]
                self.P_on_devices[start_index + i] = P_on_list[min_y_r]
            INDICATOR_d += torch.sum(torch.sum(x_d_diff * x_d_diff, dim=3) / (self.points_r + self.points_d), dim=2)
            del mem_x_d_T, x_d_diff, INDICATOR_d_i
            torch.cuda.empty_cache()
            gc.collect()

        INDICATOR_r = INDICATOR_r.cpu()
        INDICATOR_d = INDICATOR_d.cpu()
        self.loss_r += torch.sqrt(INDICATOR_r * (self.points_r + self.points_d) / self.points_r / self.device_nums)
        self.loss_d += torch.sqrt(INDICATOR_d * (self.points_r + self.points_d) / self.points_d / self.device_nums)
        self.loss = torch.min(torch.sqrt(
            (torch.min(INDICATOR_r) + torch.min(INDICATOR_d)) / self.device_nums
        ))
        min_x_r = (torch.argmin(self.loss_r) // P_off_nums).item()
        min_y_r = (torch.argmin(self.loss_r) % P_off_nums).item()
        min_x_d = (torch.argmin(self.loss_d) // P_on_nums).item()
        min_y_d = (torch.argmin(self.loss_d) % P_on_nums).item()
        self.k_off = k_off_list[min_x_r].item()
        self.k_on = k_on_list[min_x_d].item()
        self.P_off = P_off_list[min_y_r].item()
        self.P_on = P_on_list[min_y_d].item()
        self.k_off_devices = self.k_off_devices.cpu()
        self.P_off_devices = self.P_off_devices.cpu()

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

        G_off_variation = torch.from_numpy(G_off_variation)
        G_on_variation = torch.from_numpy(G_on_variation)
        G_off_variation = G_off_variation.to(device)
        G_on_variation = G_on_variation.to(device)

        P_off_nums = 1000
        P_on_nums = 1000
        # P_off_list = torch.logspace(-5, 1, P_off_nums, base=10, dtype=torch.float64)
        # P_on_list = torch.logspace(-5, 1, P_on_nums, base=10, dtype=torch.float64)
        P_off_list = torch.linspace(0, 10, P_off_nums, dtype=torch.float64)
        P_on_list = torch.linspace(0, 10, P_on_nums, dtype=torch.float64)
        P_off_list = P_off_list.to(device)
        P_on_list = P_on_list.to(device)

        J1 = 1
        select_columns = np.array([i for i in range(self.device_nums)])
        current_r = torch.from_numpy(
            np.array(self.data[select_columns])[self.start_point_r: self.start_point_r + self.points_r].T)
        current_d = torch.from_numpy(
            np.array(self.data[select_columns])[self.start_point_d: self.start_point_d + self.points_d].T)
        conductance_r = current_r / self.read_voltage
        conductance_d = current_d / self.read_voltage
        conductance_r = conductance_r.to(device)
        conductance_d = conductance_d.to(device)
        x_r = (
                (conductance_r - G_on_variation.expand(self.points_r, self.device_nums).T)
                / (G_off_variation.expand(self.points_r, self.device_nums).T
                   - G_on_variation.expand(self.points_r, self.device_nums).T)
        )
        x_d = (
                (conductance_d - G_on_variation.expand(self.points_d, self.device_nums).T)
                / (G_off_variation.expand(self.points_d, self.device_nums).T
                   - G_on_variation.expand(self.points_d, self.device_nums).T)
        )
        V_write_r = torch.from_numpy(self.V_write[self.start_point_r: self.start_point_r + self.points_r])
        V_write_d = torch.from_numpy(self.V_write[self.start_point_d: self.start_point_d + self.points_d])
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
        mem_c_r = (
                G_off_variation.expand(P_off_nums, self.device_nums).expand(self.points_r, P_off_nums,
                                                                            self.device_nums).permute(1, 2, 0)
                * mem_x_r_T
                + G_on_variation.expand(P_on_nums, self.device_nums).expand(self.points_r, P_on_nums,
                                                                            self.device_nums).permute(1, 2, 0)
                * (1 - mem_x_r_T)
        )
        # c_r_diff_percent = (mem_c_r - conductance_r) / conductance_r
        c_r_diff = (mem_c_r - conductance_r)
        INDICATOR_r = torch.sqrt(torch.sum(c_r_diff * c_r_diff, dim=2) / self.points_r).T
        P_off_variation = P_off_list[torch.argmin(INDICATOR_r, dim=1)].cpu().numpy()

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
        mem_c_d = (
                G_off_variation.expand(P_off_nums, self.device_nums).expand(self.points_d, P_off_nums,
                                                                            self.device_nums).permute(1, 2, 0)
                * mem_x_d_T
                + G_on_variation.expand(P_on_nums, self.device_nums).expand(self.points_d, P_on_nums,
                                                                            self.device_nums).permute(1, 2, 0)
                * (1 - mem_x_d_T)
        )
        # RRMSE
        # c_d_diff_percent = (mem_c_d - conductance_d) / conductance_d
        # RMSE
        c_d_diff = (mem_c_d - conductance_d)
        INDICATOR_d = torch.sqrt(torch.sum(c_d_diff * c_d_diff, dim=2) / self.points_d).T
        P_on_variation = P_on_list[torch.argmin(INDICATOR_d, dim=1)].cpu().numpy()

        torch.cuda.empty_cache()

        return P_off_variation, P_on_variation