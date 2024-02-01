import os
import sys
import torch
from torch import nn
from spikingjelly.clock_driven.layer import SeqToANNContainer

root_dir = ""
while not os.path.isdir(os.path.join(root_dir, 'bpsr')):
    root_dir = os.path.join(root_dir, '..')
sys.path.append(root_dir)
from bpsr import neuron, surrogate
from bpsr.layer import RecurrentSpikingLayer, MultiStepSpikingLayer


class MITBIH_MLP(nn.Module):
    def __init__(self, size):
        super(MITBIH_MLP, self).__init__()

        tau = 0.25
        param_tau = True
        self.fc1 = RecurrentSpikingLayer(
            synapse=nn.Linear(4 + size[0], size[0]),
            neuron=neuron.LIFNode(
                tau=torch.ones(size[0]) * tau,
                param_tau=param_tau),
            outputshape=(size[0],)
        )
        self.fc2 = MultiStepSpikingLayer(
            synapse=nn.Linear(size[0], size[1]),
            neuron=neuron.MultiStepLIFNode(tau=tau, param_tau=param_tau)
        )
        self.fc3 = MultiStepSpikingLayer(
            synapse=nn.Linear(size[1], size[2]),
            neuron=neuron.MultiStepLIFNode(tau=tau, param_tau=param_tau)
        )

    def forward(self, x):
        for model_layer in self.children():
            x = model_layer(x)
        return x.mean(axis=0)
