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

import argparse
import sys
sys.path.append('../../')
from testbenches import *
from simbrain.mapping import MimoMapping

#############################################################
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--memristor_structure", type=str, default='mimo') # trace, mimo or crossbar 
parser.add_argument("--memristor_device", type=str, default='ferro') # ideal, ferro, or hu
parser.add_argument("--c2c_variation", type=bool, default=True)
parser.add_argument("--d2d_variation", type=int, default=0) # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
parser.add_argument("--stuck_at_fault", type=bool, default=False)
parser.add_argument("--retention_loss", type=int, default=0) # retention loss, 0: without it, 1: during pulse, 2: no pluse for a long time
parser.add_argument("--aging_effect", type=int, default=0) # 0: No aging effect, 1: equation 1, 2: equation 2
parser.add_argument("--processNode", type=int, default=32)
args = parser.parse_args()

def main():
    seed = args.seed
    gpu = args.gpu
    # Sets up Gpu use
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1]))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # torch.manual_seed(seed)
    if gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    else:
        torch.manual_seed(seed)
        device = "cpu"
        if gpu:
            gpu = False

    torch.set_num_threads(os.cpu_count() - 1)
    print("Running on Device = ", device)

    _rows = 8
    _cols = 1
    _rep = 100000
    _logs = ['test_data', None, False, False, None]

    mem_device = {'device_structure': args.memristor_structure, 'device_name': args.memristor_device,
                 'c2c_variation': args.c2c_variation, 'd2d_variation': args.d2d_variation,
                 'stuck_at_fault': args.stuck_at_fault, 'retention_loss': args.retention_loss,
                 'aging_effect': args.aging_effect, 'processNode': args.processNode, 'batch_interval': 402}
    
    _crossbar = MimoMapping(sim_params=mem_device, shape=(_rows, _cols))
    _crossbar.to(device)
    _crossbar.set_batch_size_mimo(_rep)
    run_c2c_sim(_crossbar, _rep, _rows, _cols, mem_device, device, _logs)

if __name__ == "__main__":
    main()
