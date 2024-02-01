import sys
import importlib
# importlib.reload(sys)
# sys.setdefaultencoding('gbk')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import xlsxwriter
import xlrd2
import time
import math
import random
import json


# %% RRMSE Function for I-V curve
def RRMSE_I_V(i_vteam, i_ref):
    points = len(i_ref)
    i_diff = list(map(lambda x: x[0] - x[1], zip(i_vteam, i_ref)))
    RRMSE_i_v = np.sqrt(np.dot(i_diff, i_diff) / np.dot(i_ref, i_ref) / points)
    return RRMSE_i_v


def IV_curve_fitting(file, dictionary):
    # Read excel data
    data = pd.read_excel(file)

    # Read parameters
    v_off = dictionary['v_off']
    v_on = dictionary['v_on']
    G_off = dictionary['G_off']
    G_on = dictionary['G_on']
    k_off = dictionary['k_off']
    k_on = dictionary['k_on']
    P_off = dictionary['P_off']
    P_on = dictionary['P_on']
    alpha_off = 1
    alpha_on = 1

    if None in [k_off, k_on]:
        k_off = 100
        k_on = 100
    if None in [P_off, P_on]:
        P_off = 1
        P_on = 1

    return alpha_off, alpha_on

def main():
    alpha_off, alpha_on = IV_curve_fitting("../../memristordata/IV_curve.xlsx",
                                           {"G_on": 100, "G_off": 1000, "v_on": -1,
                                            "v_off": 1, "k_off": None, "k_on": None,
                                            "P_off": None, "P_on": None})
    return 0


if __name__ == '__main__':
    main()
