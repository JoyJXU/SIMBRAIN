import torch
from torch import nn
from bpsr import neuron
from bpsr.prune import BPSR_Prune
from typing import Optional, Union, Iterable, Literal, Sequence
from math import prod
from dataclasses import dataclass
import warnings


@dataclass
class SOP:
    in_spike: Union[torch.Tensor, float] = 0.
    fan_out: Optional[torch.Tensor] = None
    synapse: Optional[torch.Tensor] = None


class Monitor:
    def __init__(self, net: nn.Module,
                 encoding: Optional[Union[nn.Module, Sequence[nn.Module]]] = None,
                 exclude: Optional[Union[nn.Module, Sequence[nn.Module]]] = None):
        self.spiking_module = []
        self.synaptic_module = []
        self.encoding_module = []
        if exclude is None or isinstance(exclude, nn.Module):
            exclude = [exclude]
        if encoding is None or isinstance(encoding, nn.Module):
            encoding = [encoding]
        for name, module in net.named_modules():
            if module in exclude:
                continue
            elif module in encoding and isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                self.encoding_module.append(module)
            elif isinstance(module, (neuron.LIFNode, neuron.MultiStepLIFNode, neuron.MultiStepRateLIFNode)):
                self.spiking_module.append(module)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                self.synaptic_module.append(module)

    def enable(self, record_v: bool = False, record_s: bool = False):
        self.spiking_handle = dict.fromkeys(self.spiking_module, None)
        self.spike = dict.fromkeys(self.spiking_module, 0.)
        if record_v:
            self.v_record = dict.fromkeys(self.spiking_module)
        if record_s:
            self.s_record = dict.fromkeys(self.spiking_module)
        for module in self.spiking_module:
            self.spiking_handle[module] = module.register_forward_hook(self.spiking_hook)
        self.synaptic_handle = dict.fromkeys(self.synaptic_module, None)
        self.sop = dict.fromkeys(self.synaptic_module, SOP())
        for module in self.synaptic_module:
            self.synaptic_handle[module] = module.register_forward_hook(self.synaptic_hook)
        self.encoding_handle = dict.fromkeys(self.encoding_module, None)
        self.mac = dict.fromkeys(self.encoding_module, 0.)
        for module in self.encoding_module:
            self.encoding_handle[module] = module.register_forward_hook(self.encoding_hook)
        self.reset()

    def disable(self):
        for key_, val_ in self.spiking_handle:
            val_.remove()
        for key_, val_ in self.synaptic_handle:
            val_.remove()
        for key_, val_ in self.encoding_handle:
            val_.remove()
        del self.spiking_handle, self.spike, self.synaptic_handle, self.sop, self.encoding_handle, self.mac
        if hasattr(self, 'v_record'):
            del self.v_record
        if hasattr(self, 's_record'):
            del self.s_record

    def reset(self):
        for key_ in self.spike.keys():
            self.spike[key_] = 0.
        for key_ in self.sop.keys():
            self.sop[key_] = SOP(0., None, synapse=self.sop[key_].synapse)
        for key_ in self.mac.keys():
            self.mac[key_] = 0.
        if hasattr(self, 'v_record'):
            for key_ in self.v_record.keys():
                if self.v_record[key_] is None:
                    self.v_record[key_] = []
                else:
                    self.v_record[key_].clear()
        if hasattr(self, 's_record'):
            for key_ in self.s_record.keys():
                if self.s_record[key_] is None:
                    self.s_record[key_] = []
                else:
                    self.s_record[key_].clear()

    def spiking_hook(self, module, input, output):
        self.spike[module] += output.abs().sum()
        if hasattr(self, 'v_record'):
            self.v_record[module].append(module.v_seq if hasattr(module, 'v_seq') else module.v)
        if hasattr(self, 's_record'):
            self.s_record[module].append(output)

    def synaptic_hook(self, module, input, output):
        if self.sop[module].fan_out is None:
            if hasattr(module, 'weight_mask'):
                BPSR_Prune_Obj = None
                for key_, hook_ in module._forward_pre_hooks.items():
                    if isinstance(hook_, BPSR_Prune):
                        BPSR_Prune_Obj = hook_
                        break
                synapses = module.weight_mask \
                    if BPSR_Prune_Obj is None else \
                    BPSR_Prune_Obj.surrogate_function(module.weight_mask).abs()
                synapses = synapses * (module.weight != 0).detach()
            else:
                synapses = (module.weight != 0)
            if isinstance(module, nn.Linear):
                self.sop[module].synapse = synapses.sum()
                self.sop[module].fan_out = synapses.sum() / synapses.shape[1]
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                n_dim = int(module.__class__.__name__[4])
                self.sop[module].synapse = synapses.sum() * prod(output.shape[-n_dim:]) / module.groups
                self.sop[module].fan_out = synapses.sum() * prod(output.shape[-n_dim:]) \
                         / prod(input[0].shape[-n_dim-1:]) / module.groups
        self.sop[module].in_spike += input[0].abs().sum()

    def encoding_hook(self, module, input, output):
        if hasattr(module, 'weight_mask'):
            BPSR_Prune_Obj = None
            for key_, hook_ in module._forward_pre_hooks.items():
                if isinstance(hook_, BPSR_Prune):
                    BPSR_Prune_Obj = hook_
                    break
            synapses = module.weight_mask \
                if BPSR_Prune_Obj is None else \
                BPSR_Prune_Obj.surrogate_function(module.weight_mask).abs()
        else:
            synapses = (module.weight != 0)
        if isinstance(module, nn.Linear):
            fan_out = synapses.sum() / synapses.shape[1]
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            n_dim = int(module.__class__.__name__[4])
            fan_out = synapses.sum() * prod(output.shape[-n_dim:]) \
                     / prod(input[0].shape[-n_dim-1:]) / module.groups
        self.mac[module] += input[0].numel() * fan_out

    def get_synapses(self, detach_module: Optional[Union[Literal['all'], nn.Module, Iterable[nn.Module]]] = None):
        if detach_module is None or isinstance(detach_module, nn.Module):
            detach_module = [detach_module]
        count = 0
        for key_, val_ in self.sop.items():
            try:
                count += val_.synapse.detach() if detach_module == 'all' or key_ in detach_module else val_.synapse
            except TypeError:
                warnings.warn('Run network before get_synapses')
                return torch.nan
        return count

    def get_spikes(self, input: Optional[torch.Tensor] = None,
                   detach_module: Optional[Union[Literal['all'], nn.Module, Iterable[nn.Module]]] = None):
        if detach_module is None or isinstance(detach_module, nn.Module):
            detach_module = [detach_module]
        count = 0 if input is None else input.sum()
        for key_, val_ in self.spike.items():
            count += val_.detach() if detach_module == 'all' or key_ in detach_module else val_
        return count

    def get_synaptic_ops(self,
                         detach_module: Optional[Union[Literal['all'], nn.Module, Iterable[nn.Module]]] = None,
                         detach_property: Literal['all', 'spike', 'synapse'] = 'all'):
        if detach_module is None or isinstance(detach_module, nn.Module):
            detach_module = [detach_module]
        count = 0
        for key_, val_ in self.sop.items():
            spike, fan_out = val_.in_spike, val_.fan_out
            if detach_module == 'all' or key_ in detach_module:
                if detach_property == 'all' or detach_property == 'synapse':
                    fan_out = fan_out.detach()
                if detach_property == 'all' or detach_property == 'spike':
                    spike = spike.detach()
            count += spike * fan_out
        return count

    def get_macs(self, input: Optional[torch.Tensor] = None,
                   detach_module: Optional[Union[Literal['all'], nn.Module, Iterable[nn.Module]]] = None):
        if detach_module is None or isinstance(detach_module, nn.Module):
            detach_module = [detach_module]
        count = 0 if input is None else input.sum()
        for key_, val_ in self.mac.items():
            count += val_.detach() if detach_module == 'all' or key_ in detach_module else val_
        return count

    def get_neuron_num(self, input_neu: int = 0):
        num = input_neu
        for module in self.spiking_module:
            if 'multistep' in module.__class__.__name__.lower():
                num += prod(module.v_seq.shape[2:])
            else:
                num += prod(module.v.shape[1:])
        return num
