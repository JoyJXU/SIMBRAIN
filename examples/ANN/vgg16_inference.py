#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 18:46:30 2023

@author: jwxu
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from collections import OrderedDict

from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
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

# Model
print('==> Building model..')
net = VGG('VGG16')

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

# Load Pre-trained Model
checkpoint = torch.load('./checkpoint/ckpt.pth')
# net.load_state_dict(checkpoint['net'])
recorded_best_acc = checkpoint['acc']

# Add variation
print('==> Add variation..')
out_root = 'Fe_inference_variation_results.txt'
for test_cnt in range(100):
    # Reset Dataset
    testloader.idx=0
    
    # Record
    out = open(out_root, 'a')

    # Reload Weight
    state_dict = checkpoint['net']
    variation_state_dict = OrderedDict()
    
    # # Hu
    # sigma_relative = 0.024386543
    # sigma_absolute = 0.005490317
    
    # Ferro
    sigma_relative = 0.103221662
    sigma_absolute = 0.005784286
    
    # # Ideal
    # sigma_relative = 0.0
    # sigma_absolute = 0.0
    
    for k, v in state_dict.items():
        if 'weight' in k and len(v.size())>1:
            # print((v.abs()).max())
            scale_factor = 1 #(v.abs()).max()
            
            v_norm = v.clone()
            v_norm = torch.div(v_norm, scale_factor)
            
            normal_relative = torch.normal(0., sigma_relative, size = v_norm.size()).to(v_norm.device)
            normal_absolute = torch.normal(0., sigma_absolute, size = v_norm.size()).to(v_norm.device)
            device_v = torch.mul(v_norm, normal_relative) + normal_absolute
            
            v_norm = v_norm + device_v
            v_norm = torch.mul(v_norm, scale_factor)
            
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