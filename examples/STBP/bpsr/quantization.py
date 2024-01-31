from abc import ABC
from typing import Literal, Union, Optional, Sequence
import torch
from torch import nn
from torch.nn.parameter import Parameter
import warnings
from bpsr import neuron


class Quantizer:
    def __init__(self, sign: Union[int, bool], word: int, frac: int,
                 round_method: Literal['floor', 'round', 'ceil'] = 'round'):
        assert sign == 1 or sign == 0
        assert word >= frac + sign
        self.sign = (sign == 1)
        self.word = word
        self.frac = frac
        self.precision = pow(2, -frac)
        self.saturation = 2 ** (word - frac - sign) - self.precision
        self.round_method = round_method

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return FunctionalQuantize.apply(tensor, self.precision, self.saturation, self.sign,
                                        self.round_method)

    def __repr__(self):
        return f'Q{self.word - self.sign - self.frac}.{self.frac}'


class FunctionalQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        tensor, precision, saturation, sign, round_method = args
        # print('quant')
        # if round_method == 'floor':
        #     tensor = torch.floor(tensor / precision) * precision
        # elif round_method == 'round':
        #     tensor = torch.round(tensor / precision) * precision
        # elif round_method == 'ceil':
        #     tensor = torch.ceil(tensor / precision) * precision
        # else:
        #     raise NotImplementedError
        # tensor = torch.clamp(tensor, max=saturation,
        #                       min=-saturation if sign else 0.)
        
        # Hu
        sigma_relative = 0.024386543
        sigma_absolute = 0.005490317
        
        # # Ferro
        # sigma_relative = 0.103221662
        # sigma_absolute = 0.005784286
        
        # # Ideal
        # sigma_relative = 10.0
        # sigma_absolute = 10.0
        
        
        scale_factor = 1 #(x.abs()).max()
                
        v_norm = tensor.clone()
        v_norm = torch.div(v_norm, scale_factor)
        
        normal_relative = torch.normal(0., sigma_relative, size = v_norm.size()).to(v_norm.device)
        normal_absolute = torch.normal(0., sigma_absolute, size = v_norm.size()).to(v_norm.device)
        device_v = torch.mul(v_norm, normal_relative) + normal_absolute
        
        v_norm = v_norm + device_v
        v_norm = torch.mul(v_norm, scale_factor)
        return v_norm

    @staticmethod
    def backward(ctx, *grad_outputs):
        return grad_outputs[0], None, None, None, None


class BPSR_Quantizer(ABC):
    def __init__(self, name: Union[str, Sequence[str]],
                 quant: Quantizer, estimator: Literal['ste', 'ab'] = 'ste'):
        self._tensor_name = [name] if isinstance(name, str) else name
        self.quant = quant
        self.estimator = estimator
        if estimator == 'ab':
            self.alpha = 0.

    def __call__(self, module, inputs):
        for name_ in self._tensor_name:
            if self.estimator == 'ste':
                quant_tensor = self.quant(getattr(module, name_ + '_full'))
            else:
                quant_tensor = (1 - self.alpha) * getattr(module, name_ + '_full') +\
                               self.alpha * self.quant(getattr(module, name_ + '_full').detach())
            setattr(module, name_, quant_tensor)

    def set_alpha(self, alpha):
        assert hasattr(self, 'alpha')
        self.alpha = alpha

    def remove(self, module):
        for name_ in self._tensor_name:
            assert name_ + '_full' in module._parameters, \
                f'Module {module} has to be quantified before pruning can be removed'
            quant = self.quant(getattr(module, name_ + '_full')).clone().detach()
            if hasattr(module, name_):
                delattr(module, name_)
            delattr(module, name_ + '_full')
            module.register_parameter(name_, Parameter(quant))


def quantizate(net: nn.Module, param_quant: Quantizer,
               v_quant: Optional[Quantizer] = None, tau_quant: Optional[Quantizer] = None,
               estimator: Literal['ste', 'ab'] = 'ste'):
    if v_quant is None:
        v_quant = param_quant
    if tau_quant is None:
        tau_quant = Quantizer(sign=False, word=param_quant.frac + 1, frac=v_quant.frac,
                              round_method=param_quant.round_method)
    assert not tau_quant.sign

    for name, module in net.named_modules():
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            assert not any('_full' in n for n, v in module.named_parameters()), 'Network has been quantified'
            name = [n for n, v in module.named_parameters() if not '_mask' in n]
            for name_ in name:
                full = getattr(module, name_).clone().detach()
                module.register_parameter(name_ + '_full', Parameter(full))
                delattr(module, name_)
                setattr(module, name_, param_quant(full))
            method = BPSR_Quantizer(name, param_quant, estimator)
            module.register_forward_pre_hook(method)
            if len(module._forward_pre_hooks) > 1:
                key_ = [k for k, v in module._forward_pre_hooks.items() if v == method][0]
                module._forward_pre_hooks.move_to_end(key_, last=False)
        # elif isinstance(module, (neuron.LIFNode, neuron.MultiStepLIFNode, neuron.MultiStepRateLIFNode)):
        #     assert not hasattr(module, 'v_quant'), 'Network has been quantified'
        #     assert v_quant.saturation >= module.v_threshold, 'Inappropriate potential quantization'
        #     setattr(module, 'v_quant', v_quant)
        #     if 'tau' in module._parameters:
        #         full = getattr(module, 'tau').clone().detach()
        #         module.register_parameter('tau_full', Parameter(full))
        #         delattr(module, 'tau')
        #         setattr(module, 'tau', tau_quant(full))
        #         method = BPSR_Quantizer('tau', tau_quant, estimator)
        #         module.register_forward_pre_hook(method)
        #     else:
        #         module.tau = tau_quant(module.tau)
        # elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        #     warnings.warn("There is BatchNorm layer in network")


def remove(net: nn.Module):
    for name, module in net.named_modules():
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, BPSR_Quantizer):
                hook.remove(module)
                del module._forward_pre_hooks[k]
        if isinstance(module, (neuron.LIFNode, neuron.MultiStepLIFNode, neuron.MultiStepRateLIFNode)):
            delattr(module, 'v_quant')


def set_alpha(net: nn.Module, alpha: float):
    assert 0. <= alpha <= 1.
    for name, module in net.named_modules():
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, BPSR_Quantizer):
                hook.set_alpha(alpha)
