import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('../../')
from simbrain.mapping import MLPMapping


class Mem_Linear(nn.Linear):
    # language=rst
    """
    Abstract base class for custom linear layer that support memristor-based implementation.
    """
    def __init__(self, input_dims, output_dims, mem_device):
        # language=rst
        """
        Abstract base class constructor.
        :param input_dims: Input size of the linear layer.
        :param output_dims: Output size of the linear layer.
        :param mem_device: Memristor device to be used in learning.
        """
        super(Mem_Linear, self).__init__(input_dims, output_dims)

        self.crossbar_pos = MLPMapping(sim_params=mem_device, shape=(input_dims, output_dims))
        self.crossbar_neg = MLPMapping(sim_params=mem_device, shape=(input_dims, output_dims))

        self.crossbar_pos.set_batch_size_mlp(1)
        self.crossbar_neg.set_batch_size_mlp(1)

    def mem_update(self):
        # Enable signed matrix
        matrix_pos = torch.relu(self.weight.T)
        matrix_neg = torch.relu(self.weight.T * -1)

        # Memristor-based results simulation
        # Memristor crossbar program
        self.crossbar_pos.mapping_write_mlp(target_x=matrix_pos.unsqueeze(0))
        self.crossbar_neg.mapping_write_mlp(target_x=matrix_neg.unsqueeze(0))


    def forward(self, x):
        cross_pos = self.crossbar_pos.mapping_read_mlp(target_v=x)
        cross_neg = self.crossbar_neg.mapping_read_mlp(target_v=(x * -1))
        output = cross_pos + cross_neg

        if self.bias is not None:
            output += self.bias

        return output