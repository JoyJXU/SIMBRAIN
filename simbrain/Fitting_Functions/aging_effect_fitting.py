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


class AgingEffect(object):
    def __init__(
            self,
            file,
            dictionary,
            **kwargs,
    ) -> None:
        data = pd.DataFrame(pd.read_excel(
            file,
            sheet_name=0,
            header=None,
            names=[
                'X',
                'Y',
                'Cycle',
                'HRS',
                'LRS'
            ]
        ))

        self.X = data['X']
        self.Y = data['Y']
        self.Cycle = data['Cycle']
        self.HRS = data['HRS']
        self.LRS = data['LRS']
        self.points = len(self.Cycle)

        self.CYCLE = self.Cycle
        self.LCS = 1 / self.HRS
        self.HCS = 1 / self.LRS

        self.sampling_rate = dictionary['sampling_rate']
        if self.sampling_rate is None:
            self.sampling_rate = 1
        self.G_0 = 1
        G_0_i = 0
        self.G_0_H = 1 / np.median(self.HRS[G_0_i:G_0_i + self.sampling_rate])
        self.G_0_L = 1 / np.median(self.LRS[G_0_i:G_0_i + self.sampling_rate])

    # For calculating
    def equation_1(self, CYCLE, r):
        return self.G_0 * np.power(1 - r, CYCLE)

    def equation_2(self, CYCLE, k, b):
        return k * CYCLE + b

    # For plotting
    def equation_1_log(self, CYCLE, r):
        return np.log(self.G_0) + CYCLE * np.log(1 - r)

    @timer
    def fitting_equation1(self):
        self.G_0 = self.G_0_L
        params_off, pconv_off = curve_fit(
            self.equation_1_log,
            self.CYCLE,
            np.log(self.HCS),
            p0=1e-4
        )

        self.G_0 = self.G_0_H
        params_on, pconv_on = curve_fit(
            self.equation_1_log,
            self.CYCLE,
            np.log(self.LCS),
            p0=1e-4
        )

        Aging_off = params_off[0]
        Aging_on = params_on[0]

        return Aging_off, Aging_on

    @timer
    def fitting_equation2(self):
        params_off, pconv_off = curve_fit(
            self.equation_2,
            self.CYCLE,
            self.HCS,
        )

        params_on, pconv_on = curve_fit(
            self.equation_2,
            self.CYCLE,
            self.LCS,
        )

        Aging_off = params_off[0]
        b_off = params_off[1]
        Aging_on = params_on[0]
        b_on = params_on[1]

        return Aging_off, b_off, Aging_on, b_on
        