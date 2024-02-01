import argparse
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataset
import mlp
from module import *

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--rep", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument('--data_root', default='data/', help='folder to save the model')
parser.add_argument("--memristor_structure", type=str, default='crossbar') # trace, mimo or crossbar
parser.add_argument("--memristor_device", type=str, default='ferro') # ideal, ferro, or hu
parser.add_argument("--c2c_variation", type=bool, default=False)
parser.add_argument("--d2d_variation", type=int, default=0) # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
parser.add_argument("--stuck_at_fault", type=bool, default=False)
parser.add_argument("--retention_loss", type=int, default=0) # retention loss, 0: without it, 1: during pulse, 2: no pluse for a long time
parser.add_argument("--aging_effect", type=int, default=0) # 0: No aging effect, 1: equation 1, 2: equation 2
parser.add_argument("--process_node", type=int, default=10000)
args = parser.parse_args()

# Sets up Gpu use
os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [1]))
seed = args.seed
gpu = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
print("Running on Device = ", device)

# Mem device setup
mem_device = {'device_structure': args.memristor_structure, 'device_name': args.memristor_device,
              'c2c_variation': args.c2c_variation, 'd2d_variation': args.d2d_variation,
              'stuck_at_fault': args.stuck_at_fault, 'retention_loss': args.retention_loss,
              'aging_effect': args.aging_effect, 'process_node': args.process_node, 'batch_interval': 100}

# Dataset prepare
print('==> Preparing data..')
test_loader = dataset.get(batch_size=args.batch_size, data_root=args.data_root, num_workers=1, train=False, val=True)

# Network Model
# model = mlp.mlp_mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=True,)
model = mlp.mem_mnist(input_dims=784, n_hiddens=[256, 256], n_class=10, pretrained=True, mem_device=mem_device)

# Area print
total_area = 0
for layer_name, layer in model.layers.items():
    if isinstance(layer, Mem_Linear):
        total_area += layer.crossbar_pos.mem_array.area.array_area
        total_area += layer.crossbar_neg.mem_array.area.array_area
print("total crossbar area=", total_area, " m2")

# Memristor write
print('==> Write Memristor..')
start_time = time.time()
for layer_name, layer in model.layers.items():
    if isinstance(layer, Mem_Linear):
        layer.mem_update()
end_time = time.time()
exe_time = end_time - start_time
print("Execution time: ", exe_time)

# print write power results
total_energy = 0
average_power = 0
for layer_name, layer in model.layers.items():
    if isinstance(layer, Mem_Linear):
        layer.crossbar_pos.mem_array.total_energy_calculation()
        layer.crossbar_neg.mem_array.total_energy_calculation()
        sim_power_pos = layer.crossbar_pos.mem_array.power.sim_power
        sim_power_neg = layer.crossbar_neg.mem_array.power.sim_power
        total_energy += sim_power_pos['total_energy'] + sim_power_neg['total_energy']
        average_power += sim_power_pos['average_power'] + sim_power_neg['average_power']
print("\ttotal_write_energy=", total_energy)
print("\taverage_write_power=", average_power)

model.to(device)

# Repeated Experiment
print('==> Read Memristor..')
out_root = 'MLP_inference_results.txt'
for test_cnt in range(args.rep):
    # Reset Dataset
    test_loader.idx = 0

    # Record
    out = open(out_root, 'a')

    # Evaluate
    print('==> Evaluate..')
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            indx_target = target.clone()
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            correct += pred.cpu().eq(indx_target).sum()

    acc = 100. * correct / len(test_loader.dataset)
    print('\tTest Accuracy: {}/{} ({:.0f}%)'.format(correct, len(test_loader.dataset), acc))

    # print power results
    total_energy = 0
    average_power = 0
    for layer_name, layer in model.layers.items():
        if isinstance(layer, Mem_Linear):
            layer.crossbar_pos.mem_array.total_energy_calculation()
            layer.crossbar_neg.mem_array.total_energy_calculation()
            sim_power_pos = layer.crossbar_pos.mem_array.power.sim_power
            sim_power_neg = layer.crossbar_neg.mem_array.power.sim_power
            total_energy += sim_power_pos['total_energy'] + sim_power_neg['total_energy']
            average_power += sim_power_pos['average_power'] + sim_power_neg['average_power']
    print("\ttotal_energy=", total_energy)
    print("\taverage_power=", average_power)

    out_txt = 'Accuracy:' + str(acc) + '\n'
    out.write(out_txt)
    out.close()