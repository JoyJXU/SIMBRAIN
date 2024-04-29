import numpy as np
import pandas as pd
from scipy.optimize import curve_fit


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


class RetentionLoss(object):
    def __init__(
            self,
            file,
            **kwargs,
    ) -> None:
        data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name=0,
            header=None,
            names=[
                'Time(s)',
                'Conductance(S)'
            ]
        ))

        self.time = np.array(data['Time(s)'])
        self.conductance = np.array(data['Conductance(S)'])
        self.delta_t = data['Time(s)'][1] - data['Time(s)'][0]
        # TODO: Use mean value instead?
        self.w_init = self.conductance[0]
        self.points = len(self.time)

    def retention_loss(self, time, k, beta):
        internal_state = np.zeros(400)
        internal_state[0] = self.w_init

        for i in range(self.points - 1):
            internal_state[i + 1] = internal_state[i] - self.delta_t * k * internal_state[i] * (
                    (time[i]) ** (beta - 1))

        return internal_state

    # def RRMSE_MEAN(self, conductance_100):
    #     R_diff = list(map(lambda x: x[0] - x[1], zip(conductance_100, self.conductance)))
    #     square_sum = np.dot(R_diff, R_diff)
    #     mean_square_sum = square_sum / self.points
    #     RMSE = np.sqrt(mean_square_sum)
    #     RRMSE_mean = RMSE / np.mean(self.conductance)
    #     return RRMSE_mean

    @timer
    def fitting(self):
        params, pconv = curve_fit(
            self.retention_loss,
            self.time,
            self.conductance
        )
        # print(params, pconv)
        tau, beta = params[0], params[1]
        # tau = 0.012478, beta = 1.066
        return tau, beta

    # @timer
    # def fitting_RRMSE(self):
    #     min = 1000
    #     mink = 1
    #     minb = 1
    #     k_num = 100
    #     b_num = 100
    #     k = np.logspace(-10, 0, k_num, base=2)
    #     b = np.logspace(0, 1, b_num, base=2)
    #
    #     for i in range(k_num):
    #         for j in range(b_num):
    #             m = self.RRMSE_MEAN(self.retention_loss(self.time, k[i], b[j]))
    #             if m < min:
    #                 min = m
    #                 mink = i
    #                 minb = j
    #
    #     return k[mink], b[minb]
