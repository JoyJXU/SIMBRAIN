#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 18:46:30 2023

@author: jwxu
"""

import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict

import os
import sys
sys.path.append('../../')

from models import *
from simbrain.mapping import ANNMapping

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--gpu", dest="gpu", action="store_true", default='gpu')
parser.add_argument("--rep", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--memristor_structure", type=str, default='crossbar') # trace, mimo or crossbar
parser.add_argument("--memristor_device", type=str, default='ferro') # ideal, ferro, or hu
parser.add_argument("--c2c_variation", type=bool, default=False)
parser.add_argument("--d2d_variation", type=int, default=0) # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
parser.add_argument("--stuck_at_fault", type=bool, default=False)
parser.add_argument("--retention_loss", type=int, default=0) # retention loss, 0: without it, 1: during pulse, 2: no pluse for a long time
parser.add_argument("--aging_effect", type=int, default=0) # 0: No aging effect, 1: equation 1, 2: equation 2
parser.add_argument("--processNode", type=int, default=32)
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
              'aging_effect': args.aging_effect, 'processNode': args.processNode, 'batch_interval': None}

# Dataset prepare
print('==> Preparing data..')
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Network Model
print('==> Building golden model..')
net = VGG('VGG16')
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load Pre-trained Model
checkpoint = torch.load('./checkpoint/ckpt.pth')
recorded_best_acc = checkpoint['acc']

# Repeated Experiment
out_root = 'Inference_results.txt'
for test_cnt in range(args.rep):
    # Reset Dataset
    testloader.idx=0
    
    # Record
    out = open(out_root, 'a')

    # Reload Weight
    state_dict = checkpoint['net']
    mem_state_dict = OrderedDict()

    print('==> Memristor Write..')
    for k, v in state_dict.items():
        if 'weight' in k and len(v.size())>1:
            scale_factor = 1 #(v.abs()).max()
            
            v_norm = v.clone()
            v_norm = torch.div(v_norm, scale_factor)

            crossbar_pos = CNNMapping(sim_params=mem_device, shape=v.shape)
            crossbar_neg = CNNMapping(sim_params=mem_device, shape=v.shape)
            
            # normal_relative = torch.normal(0., sigma_relative, size = v_norm.size()).to(v_norm.device)
            # normal_absolute = torch.normal(0., sigma_absolute, size = v_norm.size()).to(v_norm.device)
            # device_v = torch.mul(v_norm, normal_relative) + normal_absolute
            #
            # v_norm = v_norm + device_v
            # v_norm = torch.mul(v_norm, scale_factor)
            
        else:
            v_norm = v
        
        variation_state_dict[k] = v_norm
    
    net.load_state_dict(variation_state_dict)
    
    # Evaluate
    print('==> Evaluate..')
    criterion = nn.CrossEntropyLoss()
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
    
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    acc = 100.*correct/total
    
    print('Accuracy Results:' + str(acc))
    
    out_txt = 'Accuracy:' + str(acc) + '\n'
    out.write(out_txt)
    out.close()