from typing import Iterable, Optional, Union
from bindsnet.network.array import MemristorArray
import json
import torch

class Mapping(torch.nn.Module):
    # language=rst
    """
    Abstract base class for mapping neural networks to memristor arrays.
    """

    def __init__(
        self,
        mem_device: dict = {},
        shape: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param mem_device: Memristor device to be used in learning.
        :param shape: The dimensionality of the layer.
        """
        super().__init__()
    
        self.shape = shape  
        self.device_name = mem_device['device_name']        
        
        self.register_buffer("mem_v", torch.Tensor())
        self.register_buffer("mem_t", torch.Tensor())
        self.register_buffer("x", torch.Tensor())  
        
        with open('../memristor_device_info.json', 'r') as f:
            self.memristor_info_dict = json.load(f)
        assert self.device_name in self.memristor_info_dict.keys(), "Invalid Memristor Device!"  
        self.vneg = self.memristor_info_dict[self.device_name]['vinput_neg']
        self.vpos = self.memristor_info_dict[self.device_name]['vinput_pos']
        self.Gon = self.memristor_info_dict[self.device_name]['G_on']
        self.Goff = self.memristor_info_dict[self.device_name]['G_off']
        self.dt = self.memristor_info_dict[self.device_name]['delta_t']
        
        self.trans_ratio = 1 / (self.Goff - self.Gon)
        
        self.mem_array = MemristorArray(mem_device=mem_device, shape=self.shape, memristor_info_dict=self.memristor_info_dict)

        self.batch_size = None


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when memristor is used to mapping traces.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
    
        self.mem_v = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)
        self.mem_t = torch.zeros(batch_size, *self.shape, device=self.mem_t.device)
        self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)
        
        self.mem_array.set_batch_size(batch_size=self.batch_size)


    def mem_t_calculate(self,mem_step):
        # Calculate the mem_t
        if self.mem_t.dim() == 4:
            self.mem_t[:, 0, :, :] = mem_step.view(-1, 1, 1)
        elif self.mem_t.dim() == 2:
            self.mem_t[:, :] = mem_step.view(-1, 1)
        else:
            print("Wrong mem_t shape!!!!!!!!")
        self.mem_t *= self.dt 
        
        
    def reset_memristor_variables(self,mem_step) -> None:
        # language=rst
        """
        Abstract base class method for resetting state variables.
        """
        self.mem_v.fill_(-self.vpos)
        self.mem_t_calculate(mem_step=mem_step)        
        # Adopt large negative pulses to reset the memristor array
        self.mem_array.memristor_compute(mem_v=self.mem_v, mem_t=self.mem_t)

        
    def update_SAF_mask(self) -> None:
        self.mem_array.update_SAF_mask()


    def mapping(self,s,mem_step):
        # nn to mem
        self.mem_v = s.float()
        self.mem_v[self.mem_v == 0] = self.vneg
        self.mem_v[self.mem_v == 1] = self.vpos   
        self.mem_t_calculate(mem_step=mem_step)      

        mem_c = self.mem_array.memristor_compute(mem_v=self.mem_v, mem_t=self.mem_t)
        
        # mem to nn
        self.x = (mem_c - self.Gon) * self.trans_ratio

        return self.x  
        