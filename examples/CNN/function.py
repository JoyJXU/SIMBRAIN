import torch
from torch.autograd import Function
from torch import Tensor
from typing import Tuple, Optional

import sys
sys.path.append('../../')
from simbrain.mapping import MLPMapping

class MemLinearFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor, crossbar_pos: MLPMapping, crossbar_neg: MLPMapping) -> Tensor:
        output_ref = input @ weight.T + bias[None, ...]

        cross_pos = crossbar_pos.mapping_read_mlp(target_v=input)
        cross_neg = crossbar_neg.mapping_read_mlp(target_v=(input * -1))
        output = cross_pos + cross_neg

        if bias is not None:
            output += bias

        ctx.save_for_backward(input, weight)

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        input, weight = ctx.saved_tensors
        grad_input = grad_output @ weight
        grad_weight = grad_output.T @ input
        grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None, None


class MemConv2dFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Tensor, stride: int, padding: int,
                crossbar_pos: MLPMapping, crossbar_neg: MLPMapping) -> Tensor:
        batch_size, channels, height, width = input.size()
        kernel_size = weight.size(2)
        out_channels = weight.size(0)

        # Compute output dimensions
        out_height = (height - kernel_size + 2 * padding) // stride + 1
        out_width = (width - kernel_size + 2 * padding) // stride + 1

        # Add padding to the input
        input_padded = torch.nn.functional.pad(input, (padding, padding, padding, padding))

        # Unfold the input tensor to extract patches
        unfolded_input = torch.nn.functional.unfold(input_padded, (kernel_size, kernel_size), stride=stride)

        # Reshape input to the memristor array input
        input_reshape = unfolded_input.transpose(1, 2)
        s0 = input_reshape.size(0)
        s1 = input_reshape.size(1)
        s2 = input_reshape.size(2)
        input_reshape = input_reshape.reshape(-1, s2)

        # Matrix-Multiplication
        # weight_reshape = weight.reshape(out_channels, -1).t()
        # out_unfolded = input_reshape.matmul(weight_reshape)
        cross_pos = crossbar_pos.mapping_read_cnn(target_v=input_reshape)
        cross_neg = crossbar_neg.mapping_read_cnn(target_v=(input_reshape * -1))
        out_unfolded = cross_pos + cross_neg

        # Reshape the output
        out_unfolded = out_unfolded.reshape(s0, s1, -1)
        out_unfolded = out_unfolded.transpose(1, 2)

        # Fold the output
        output = torch.nn.functional.fold(out_unfolded, (out_height, out_width), (1, 1))

        if bias is not None:
            output += bias.unsqueeze(-1).unsqueeze(-1)

        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        input, weight = ctx.saved_tensors
        grad_input = grad_output @ weight
        grad_weight = grad_output.T @ input
        grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias, None, None