#!python3

"""
MIT License

Copyright (c) 2023 Dimitrios Stathis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# Crossbar dimensions:
# Rows --> Input sources
# Columns --> Outputs during computation, ground during programming

# Calculation of Voltages depending on the state of the devices (R) and the Voltage sources

from utility import utility
import time
import os
import torch


#############################################################
# Run a simulation of the crossbar based on the configuration
def run_sim(_crossbar, _rep, _rows, _cols, _logs=[None, None, False, False, None]):
    print("<========================================>")
    print("Test case: ", _rep)
    file_name = "test_case_r"+str(_rows)+"_c_" + \
        str(_cols)+"_rep_"+str(_rep)+".csv"
    file_path = _logs[0] #main file path
    header = ['var_abs', 'var_rel']
    for x in range(_cols):
        header.append(str(x))
    file = file_path+"/"+file_name # Location to the file for the main results
    # Only write header once
    if not (os.path.isfile(file)):
        utility.write_to_csv(file_path, file_name, header)

    print("<==============>")
    start_time = time.time()
    print("Row No. ", _rows, " Column No. ", _cols)

    # matrix and vector random generation
    matrix = torch.rand(_rows, _cols)
    vector = -1 + 2 * torch.rand(_rows)
    print("Randomized input")

    # Golden results calculation
    golden_model = torch.matmul(vector, matrix)

    # Memristor-based results simulation
    # Memristor crossbar program
    _crossbar.mapping_write_mimo(target_x=matrix)
    # Memristor crossbar perform matrix vector multiplication
    cross = _crossbar.mapping_read_mimo(target_v=vector)

    # Error calculation
    error = utility.cal_error(golden_model, cross)
    error = error.flatten(0, 2)

    _var_abs = 0
    _var_rel = 0
    data = [str(_var_abs), str(_var_rel)]
    [data.append(str(e.item())) for e in error]
    utility.write_to_csv(file_path, file_name, data)

    end_time = time.time()
    exe_time = end_time - start_time
    print("Execution time: ", exe_time)