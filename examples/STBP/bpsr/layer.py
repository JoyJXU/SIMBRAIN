import torch
from torch import nn
from collections import OrderedDict
from typing import Callable, Sequence, Union
from .neuron import LIFNode, MultiStepLIFNode, MultiStepRateLIFNode


class MultiStepSpikingLayer(torch.nn.Sequential):
    def __init__(self, synapse: Callable[..., nn.Module],
                 neuron: Union[MultiStepLIFNode, MultiStepRateLIFNode]):
        super().__init__(OrderedDict([
            ('synapse', synapse), ('neuron', neuron),
        ]))


class RecurrentSpikingLayer(torch.nn.Sequential):
    """
    目前仅验证了Linear层的Recurrent
    """
    def __init__(self, synapse: Callable[..., nn.Module], neuron: LIFNode, outputshape: Sequence[int]):
        assert isinstance(neuron, LIFNode)
        super().__init__(OrderedDict([
            ('synapse', synapse), ('neuron', neuron),
        ]))
        self.output_shape = outputshape

    def forward(self, x: torch.Tensor):
        out = torch.zeros((x.shape[0], x.shape[1]) + tuple(self.output_shape), device=x.device, dtype=x.dtype)
        out[0] = super().forward(
            torch.cat((x[0, ...],
                       torch.zeros((x.shape[1], ) + tuple(self.output_shape), device=x.device, dtype=x.dtype)
                       ), dim=1))
        for t in range(1, x.shape[0]):
            out[t] = super().forward(torch.cat((x[t, ...], out[t - 1, ...]), dim=1))
        return out
