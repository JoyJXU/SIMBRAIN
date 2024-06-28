import numpy as np
import pandas as pd
import torch
import torch.nn as nn


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


# data_tensor = 2 * torch.arange(10)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# data_tensor = data_tensor.to(device)
# print(data_tensor ** 2)


@timer
def fitting():
    # file = "../../memristor_data/conductance_.xlsx"
    # data = pd.DataFrame(pd.read_excel(
    #     file,
    #     sheet_name=0,
    #     header=None,
    #     index_col=None,
    # ))
    # data.columns = ['Pulse Voltage(V)', 'Read Voltage(V)', 'Current(A)'] + list(data.columns[3:] - 2)
    #
    # V_write = np.array(data['Pulse Voltage(V)'])
    # read_voltage = np.array(data['Read Voltage(V)'][0])
    #
    # start_point_r = 0
    # points_r = np.sum(V_write > 0)
    # start_point_d = start_point_r + np.sum(V_write > 0)
    # points_d = np.sum(V_write < 0)
    #
    # current_r = torch.tensor(data['Current(A)'])[start_point_r: start_point_r + points_r]
    # current_d = torch.tensor(data['Current(A)'])[start_point_d: start_point_d + points_d]
    # conductance_r = current_r / read_voltage
    # conductance_d = current_d / read_voltage
    # G_off = torch.mean(conductance_r[conductance_r.shape[0] - 10:])
    # G_on = torch.mean(conductance_d[conductance_d.shape[0] - 10:])
    # x_r = (conductance_r - G_on) / (G_off - G_on)
    # x_d = (conductance_d - G_on) / (G_off - G_on)
    #
    # J1 = 1
    # delta_t = 0.02
    # v_off = 2
    # v_on = -1.5
    # alpha_off = 5
    # alpha_on = 5
    #
    # k_off_num = 500
    # k_off_list = torch.logspace(-4, 9, k_off_num, base=10)
    # k_off_list = k_off_list.to(device)
    # k_on_num = 500
    # k_on_list = -torch.logspace(-4, 9, k_on_num, base=10)
    # k_on_list = k_on_list.to(device)
    # P_off_num = 100
    # P_off_list = torch.logspace(-1, 1, P_off_num, base=10)
    # P_off_list = P_off_list.to(device)
    # P_on_num = 100
    # P_on_list = torch.logspace(-1, 1, P_on_num, base=10)
    # P_on_list = P_on_list.to(device)
    #
    # V_write_r = torch.tensor(V_write[start_point_r: start_point_r + points_r])
    # x_init_r = x_r[0]
    # V_write_d = torch.tensor(V_write[start_point_d: start_point_d + points_d])
    # x_init_d = x_d[0]
    #
    # mem_x_r = torch.zeros([points_r, k_off_num, P_off_num])
    # mem_x_r = mem_x_r.to(device)
    # mem_x_r[0] = x_init_r * torch.ones([k_off_num, P_off_num])
    # for i in range(points_r - 1):
    #     mem_x_r[i + 1] = (
    #             k_off_list.expand(P_off_num, k_off_num).T
    #             * ((V_write_r[i + 1] / v_off - 1) ** alpha_off)
    #             * J1
    #             * (1 - mem_x_r[i]) ** P_off_list
    #             * delta_t
    #             + mem_x_r[i]
    #     )
    #     mem_x_r[i + 1] = torch.where(mem_x_r[i + 1] < 0, 0, mem_x_r[i + 1])
    #     mem_x_r[i + 1] = torch.where(mem_x_r[i + 1] > 1, 1, mem_x_r[i + 1])
    #
    # mem_x_r_T = mem_x_r.permute(1, 2, 0)
    # mem_c_r = G_off * mem_x_r_T + G_on * (1 - mem_x_r_T)
    # conductance_r = conductance_r.to(device)
    # c_r_diff_percent = (mem_c_r - conductance_r) / conductance_r
    # INDICATOR_r = torch.sqrt(torch.sum(c_r_diff_percent * c_r_diff_percent, dim=2) / points_r)
    # print(INDICATOR_r)
    # min_value = torch.min(INDICATOR_r)
    # min_index = torch.argmin(INDICATOR_r)
    # min_x_r = min_index // P_off_num
    # min_y_r = min_index % P_off_num
    # k_off = k_off_list[min_x_r]
    # P_off = P_off_list[min_y_r]
    # print(k_off, P_off)
    #
    # mem_x_d = torch.zeros([points_d, k_on_num, P_on_num])
    # mem_x_d = mem_x_d.to(device)
    # mem_x_d[0] = x_init_d * torch.ones([k_on_num, P_on_num])
    # for i in range(points_d - 1):
    #     mem_x_d[i + 1] = (
    #             k_on_list.expand(P_on_num, k_on_num).T
    #             * ((V_write_d[i + 1] / v_on - 1) ** alpha_on)
    #             * J1
    #             * mem_x_d[i] ** P_on_list
    #             * delta_t
    #             + mem_x_d[i]
    #     )
    #     mem_x_d[i + 1] = torch.where(mem_x_d[i + 1] < 0, 0, mem_x_d[i + 1])
    #     mem_x_d[i + 1] = torch.where(mem_x_d[i + 1] > 1, 1, mem_x_d[i + 1])
    #
    # mem_x_d_T = mem_x_d.permute(1, 2, 0)
    # mem_c_d = G_off * mem_x_d_T + G_on * (1 - mem_x_d_T)
    # conductance_d = conductance_d.to(device)
    # c_d_diff_percent = (mem_c_d - conductance_d) / conductance_d
    # INDICATOR_d = torch.sqrt(torch.sum(c_d_diff_percent * c_d_diff_percent, dim=2) / points_d)
    # print(INDICATOR_d)
    # min_value = torch.min(INDICATOR_d)
    # min_index = torch.argmin(INDICATOR_d)
    # min_x_d = min_index // P_on_num
    # min_y_d = min_index % P_on_num
    # k_on = k_on_list[min_x_d]
    # P_on = P_on_list[min_y_d]
    # print(k_on, P_on)

    # x = torch.tensor([1, 2, 3])
    # y = torch.tensor([4, 5])
    # z = x.expand(2, y.shape[0], x.shape[0])
    # print(z)
    # print(z.permute(2, 0, 1))
    # m = torch.sum(z * z, dim=2)
    # print(m)
    # n = torch.stack([m, m], dim=1)
    # print(n)

    # device_num = 10
    # points_r = 3
    # J1 = 1
    # v_off = 1
    # P_off = 1
    # alpha_off_num = 10
    # alpha_off_list = torch.linspace(1, alpha_off_num, alpha_off_num)
    # k_off_num = 5
    # k_off_list = torch.linspace(1, k_off_num, k_off_num)
    # x_init_r = 0.1 * torch.ones(device_num)
    # V_write_r = 3 * torch.ones(3)
    # delta_t = 1
    # mem_x_r = torch.zeros([points_r, device_num, alpha_off_num, k_off_num])
    # mem_x_r[0] = x_init_r.expand(alpha_off_num, k_off_num, device_num).permute(2, 1, 0)
    # for j in range(points_r - 1):
    #     mem_x_r[j + 1] = (
    #             k_off_list.expand(alpha_off_num, k_off_num)
    #             * ((V_write_r[j + 1] / v_off - 1) ** alpha_off_list.expand(k_off_num, alpha_off_num).T)
    #             * J1
    #             * (1 - mem_x_r[j]) ** P_off
    #             * delta_t
    #             + mem_x_r[j]
    #     )
    # print(mem_x_r)

    # tensor = np.array([[[19, 16, 8, 11],
    #                     [14, 6, 2, 22],
    #                     [4, 9, 17, 13]],
    #                    [[7, 12, 0, 5],
    #                     [21, 15, 20, 10],
    #                     [18, 3, 23, 1]]])
    #
    # # 按第一个维度切片，得到两个子矩阵
    # sub_tensor1 = tensor[0]
    # sub_tensor2 = tensor[1]
    #
    # # 计算第一个子矩阵的最小值及索引
    # min_value_sub1 = np.min(sub_tensor1)
    # min_index_sub1 = np.argmin(sub_tensor1)
    # min_index_sub1 = np.unravel_index(min_index_sub1, sub_tensor1.shape)
    # print(min_index_sub1[0], min_index_sub1[1])
    #
    # # 计算第二个子矩阵的最小值及索引
    # min_value_sub2 = np.min(sub_tensor2)
    # min_index_sub2 = np.argmin(sub_tensor2)
    # min_index_sub2 = np.unravel_index(min_index_sub2, sub_tensor2.shape)
    #
    # # 打印结果
    # print("第一个子矩阵的最小值:", min_value_sub1)
    # print("第一个子矩阵的最小值的索引:", min_index_sub1)
    # print("第二个子矩阵的最小值:", min_value_sub2)
    # print("第二个子矩阵的最小值的索引:", min_index_sub2)

    # a = torch.tensor(
    #     [
    #         [
    #             [
    #                 [1, 2, 3, 4], [4, 5, 6, 4], [7, 8, 9, 4]
    #             ],
    #             [
    #                 [1, 2, 3, 4], [4, 5, 6, 4], [7, 8, 9, 4]
    #             ]
    #         ],
    #         [
    #             [
    #                 [1, 2, 3, 4], [4, 5, 6, 4], [7, 8, 9, 4]
    #             ],
    #             [
    #                 [1, 2, 3, 4], [4, 5, 6, 4], [7, 8, 9, 4]
    #             ]
    #         ]
    #     ]
    # )
    # # print(a, a.shape)
    # a[0] = torch.tensor([[[1, 2, 2, 4], [5, 5, 2, 4], [3, 3, 7, 4]]])
    # print(a)
    # b = torch.randn(2).expand(3, 2).expand(4, 3, 2).permute(2, 1, 0)
    # print(b)
    # a[1] = b
    # print(a)

    # k = torch.linspace(1, 3, 3).expand(5, 3)
    # print(k)
    # a = 2 ** torch.linspace(1, 5, 5).expand(3, 5).T
    # print(a)
    # m = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
    # print(m)
    # z =k * a * m
    # print(z)

    import torch

    def estimate_memory(tensor_size, dtype=torch.float32):
        element_size = dtype().itemsize
        tensor_size_bytes = torch.prod(torch.tensor(tensor_size)) * element_size
        return tensor_size_bytes / (1024 ** 3)  # 转换为GB单位

    def auto_iteration_process(tensor_size, device_memory_limit_gb, dtype=torch.float32):
        tensor_memory_gb = estimate_memory(tensor_size, dtype)
        max_tensor_memory_gb = device_memory_limit_gb / 2  # 为了避免超出设备内存限制，将最大张量内存设置为设备内存的一半
        max_tensor_size_bytes = max_tensor_memory_gb * (1024 ** 3)  # 将GB转换为字节单位

        num_iterations = (tensor_size[0] + 1) // 2  # 初始迭代次数为张量数量的一半
        while True:
            iteration_tensor_size = (num_iterations,) + tensor_size[1:]  # 计算每个子张量的大小
            iteration_tensor_memory_gb = estimate_memory(iteration_tensor_size, dtype)
            if iteration_tensor_memory_gb <= max_tensor_memory_gb:  # 如果子张量的内存占用小于等于设备内存的一半
                break  # 找到合适的迭代次数，退出循环
            num_iterations -= 1  # 否则，减小迭代次数

        return num_iterations

    # 示例用法
    tensor_size = (20, 300, 500, 100)  # 每个子张量的大小
    device_memory_limit_gb = 16  # 设备内存限制，单位为GB
    dtype = torch.float32  # 张量数据类型

    num_iterations = auto_iteration_process(tensor_size, device_memory_limit_gb, dtype)
    print("Estimated iterations:", num_iterations)


if __name__ == '__main__':
    fitting()
