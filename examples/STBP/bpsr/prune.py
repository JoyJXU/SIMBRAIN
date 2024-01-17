from typing import Union
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torch.nn.utils import prune
from bpsr import surrogate as surrogate

# import pydevd
# pydevd.settrace(suspend=False, trace_only_current_thread=True)


class BPSR_Prune(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, surrogate_function):
        super().__init__()
        self.surrogate_function = surrogate_function

    def compute_mask(self, t, default_mask):
        return default_mask

    def apply_mask(self, module):
        assert self._tensor_name is not None, f'Module {module} has to be pruned'
        mask = getattr(module, self._tensor_name + '_mask')
        orig = getattr(module, self._tensor_name + '_orig')
        pruned_tensor = orig * self.surrogate_function(mask)
        return pruned_tensor

    @classmethod
    def apply(cls, module, name, *args, importance_scores=None, **kwargs):
        surrogate_function = kwargs['surrogate_function']
        method = super(BPSR_Prune, cls).apply(module, name,
                                              surrogate_function=surrogate_function)
        mask = getattr(module, name + '_mask').clone().detach()
        del module._buffers[name + "_mask"]
        module.register_parameter(name + '_mask', Parameter(mask))
        return method

    def remove(self, module):
        assert hasattr(module, self._tensor_name + '_mask'), \
            f'Module {module} has to be pruned before pruning can be removed'
        weight = self.apply_mask(module).clone().detach()
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        delattr(module, self._tensor_name + '_orig')
        delattr(module, self._tensor_name + '_mask')
        module.register_parameter(self._tensor_name, Parameter(weight))

    def fix(self, module):
        assert self._tensor_name + '_mask' in module._parameters, \
            f'Module {module} has to be pruned before pruning can be fixed'
        mask = getattr(module, self._tensor_name + '_mask').clone().detach()
        mask = (mask >= 0).to(mask) * 2. - 1.
        del module._parameters[self._tensor_name + '_mask']
        module.register_buffer(self._tensor_name + '_mask', mask)


def bpsr_prune(module: nn.Module, name: str,
               surrogate_function: Union[surrogate.SurrogateFunctionBase,
                                         surrogate.MultiArgsSurrogateFunctionBase] = surrogate.Sigmoid()
               ) -> prune.BasePruningMethod:
    """
    非结构化剪枝，通过可学习的mask实现网络层修剪

    :param module: 待剪枝的网络层，注意该层必须直接包含待剪枝参数
    :type module: nn.Module

    :param name: 待剪枝参数名称
    :type name: str

    :param surrogate_function: mask所使用的替代梯度近似
    :type surrogate_function: bpsr.surrogate类

    :return: prune.BasePruningMethod
    """
    return BPSR_Prune.apply(module, name, surrogate_function=surrogate_function)


def remove(module, name):
    """
    固化剪枝结果并移除mask，但将其设为不可学习的buffer
    :param module: 剪枝的网络层，注意该层必须直接包含待剪枝参数
    :param name: 剪枝参数名称
    """
    prune.remove(module, name)


def fix(module, name):
    """
    固定剪枝结果，不同于remove，fix函数仍保留mask，但将其设为不可学习的buffer
    :param module: 剪枝的网络层，注意该层必须直接包含待剪枝参数
    :param name: 剪枝参数名称
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BPSR_Prune) and hook._tensor_name == name:
            hook.fix(module)
            return module
    raise ValueError(f'Parameter "{name}" of module {module} '
                     f'has to be pruned before pruning can be removed')


class Tw_Prune(prune.BasePruningMethod):
    PRUNING_TYPE = 'unstructured'

    def __init__(self, Tw):
        super().__init__()
        self.Tw = Tw

    def compute_mask(self, t, default_mask):
        return default_mask * (t.abs() >= self.Tw)

    @classmethod
    def apply(cls, module, name, Tw):
        return super(Tw_Prune, cls).apply(module, name, Tw=Tw)
