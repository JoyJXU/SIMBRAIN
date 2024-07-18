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
        self.w_init = self.conductance[0]
        self.points = len(self.time)

    def retention_loss(self, time, tau_reciprocal, beta):
        internal_state = np.zeros(400)
        internal_state[0] = self.w_init

        for i in range(self.points - 1):
            internal_state[i + 1] = internal_state[i] - self.delta_t * beta * tau_reciprocal ** beta * internal_state[i] * (
                    (time[i]) ** (beta - 1))

        return internal_state

    @timer
    def fitting(self):
        w_init_list = self.conductance[0] * (1 + np.random.uniform(-0.1, 0.1, 100))
        tau_reciprocal_list = np.zeros(100)
        beta_list = np.zeros(100)
        for i in range(100):
            self.w_init = w_init_list[i]
            params, pconv = curve_fit(
                self.retention_loss,
                self.time,
                self.conductance
            )
            tau_reciprocal_list[i], beta_list[i] = params[0], params[1]
        tau_reciprocal = np.mean(tau_reciprocal_list)
        beta = np.mean(beta_list)
        # tau = 0.012478, beta = 1.066
        self.w_init = self.conductance[0]
        return tau_reciprocal, beta
