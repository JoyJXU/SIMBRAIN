import torch
from typing import Iterable, Optional, Union
import json

class Power(torch.nn.Module):

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        memristor_info_dict: dict = {},
        length_row: float = 0,
        length_col: float = 0,
        **kwargs,
    ) -> None:

        super().__init__()
        
        with open('../../technology_info.json', 'r') as file:
            self.tech_info_dict = json.load(file)  
    
        self.shape = shape    
        self.device_name = sim_params['device_name']
        self.device_structure = sim_params['device_structure'] 
        self.process_node = sim_params['process_node'] 
        self.average_power = 0
        self.total_energy = 0
        self.read_energy = 0
        self.write_energy = 0
        self.dynamic_read_energy = 0
        self.dynamic_write_energy = 0
        self.static_read_energy = 0
        self.static_write_energy = 0   
        self.register_buffer("read_Isum", torch.Tensor()) 
        self.register_buffer("selected_write_energy", torch.Tensor())
        self.register_buffer("half_selected_write_energy", torch.Tensor()) 
        self.register_buffer("mem_v", torch.Tensor()) 
        self.register_buffer("mem_v_read", torch.Tensor()) 
        self.register_buffer("mem_c", torch.Tensor()) 
        self.register_buffer("mem_c_pre", torch.Tensor()) 
        
        self.wire_cap_row = length_row * 0.2e-15/1e-6
        self.wire_cap_col = length_col * 0.2e-15/1e-6
        self.dt = memristor_info_dict[self.device_name]['delta_t']
        
        relax_ratio = 1.25
        mem_size = memristor_info_dict[self.device_name]['mem_size']
        AR = self.tech_info_dict[str(self.process_node)]['AR']
        Rho = self.tech_info_dict[str(self.process_node)]['Rho']
        wire_resistance_unit = relax_ratio * mem_size * Rho / (AR * self.process_node * self.process_node * 1e-18)
        self.total_wire_resistance = wire_resistance_unit * (torch.arange(1, self.shape[1] + 1, device=self.mem_c.device) + 
                                                              torch.arange(self.shape[0], 0, -1, device=self.mem_c.device)[:, None])
    
        self.sim_power = {}
        


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size

        self.mem_c = torch.zeros(batch_size, *self.shape, device=self.mem_c.device)
        self.mem_c_pre = torch.zeros(batch_size, *self.shape, device=self.mem_c_pre.device)
        self.mem_v = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)
        self.mem_v_read = torch.zeros(batch_size, *(self.shape[0],), device=self.mem_v_read.device)
        self.read_Isum = torch.zeros(batch_size, *self.shape, device=self.read_Isum.device)
        self.selected_write_energy = torch.zeros(batch_size, *(1,self.shape[1]), device=self.selected_write_energy.device)
        self.half_selected_write_energy = torch.zeros(batch_size, *self.shape, device=self.selected_write_energy.device)
        self.total_wire_resistance = torch.stack([self.total_wire_resistance] * self.batch_size).cuda()
        
    
    def read_energy_calculation(self, mem_v_read, mem_c) -> None:
        
        self.mem_v_read = mem_v_read
        self.mem_c = mem_c
        self.dynamic_read_energy += torch.sum(self.mem_v_read * self.mem_v_read * self.wire_cap_row)
        self.read_Isum = self.mem_v_read[:, :, :, None]/(1/self.mem_c + self.total_wire_resistance)[:, None, :, :]
        self.static_read_energy += torch.sum(self.read_Isum[:, :, :, :] * self.mem_v_read[:, :, :, None] * self.dt * 1/2)
        self.read_energy = self.dynamic_read_energy + self.static_read_energy


    def write_energy_calculation(self, mem_v, mem_c, mem_c_pre) -> None:
        
        self.mem_v = mem_v
        self.mem_c = mem_c
        self.mem_c_pre = mem_c_pre
        
        if self.device_structure == 'trace':
            self.dynamic_write_energy += torch.sum(self.mem_v.squeeze() * self.mem_v.squeeze() * self.wire_cap_col)
            self.selected_write_energy = self.mem_v * self.mem_v * 1/2 * self.dt /(1/2 * (self.mem_c + self.mem_c_pre) + self.total_wire_resistance)
            self.static_write_energy += torch.sum(self.selected_write_energy)
        elif self.device_structure in {'crossbar', 'mimo'}:
            for write_row in range(self.shape[0]):
                self.dynamic_write_energy += torch.sum(self.mem_v[:,write_row,:] * self.mem_v[:,write_row,:] * self.wire_cap_col)
                self.dynamic_write_energy += torch.sum(torch.max(self.mem_v[:,write_row,:], dim=1)[0] * torch.max(self.mem_v[:,write_row,:], dim=1)[0] * self.wire_cap_row * (1 + 1/4 * (self.shape[0] - 1)))
                self.selected_write_energy = (self.mem_v[:,write_row,:] * self.mem_v[:,write_row,:] * 1/2 * self.dt /(1/2 * (self.mem_c[:,write_row,:] + self.mem_c_pre[:,write_row,:]) + self.total_wire_resistance[:,write_row,:])).unsqueeze(1)
                half_selected_write_energy_pre = self.mem_v[:,0:write_row,:] * self.mem_v[:,0:write_row,:] * 1/2 * 1/2 * 1/2 * self.dt /(self.mem_c[:,0:write_row,:]+ self.total_wire_resistance[:,0:write_row,:])
                half_selected_write_energy_post = self.mem_v[:,write_row+1:,:] * self.mem_v[:,write_row+1:,:] * 1/2 * 1/2 * 1/2 * self.dt /(self.mem_c_pre[:,write_row+1:,:] + self.total_wire_resistance[:,write_row+1:,:])
                self.half_selected_write_energy = torch.cat([half_selected_write_energy_pre,torch.zeros_like(self.selected_write_energy), half_selected_write_energy_post], dim=1)
                self.static_write_energy += (torch.sum(self.selected_write_energy)+torch.sum(self.half_selected_write_energy))
        else:
            raise Exception("Only trace, mimo and crossbar architecture are supported!") 
        self.write_energy = self.dynamic_write_energy + self.static_write_energy
            
    def total_energy_calculation(self, mem_t) -> None:
        
        self.total_energy = self.read_energy + self.write_energy
        self.average_power = self.total_energy / (torch.max(mem_t) * self.dt)
        self.sim_power = {'dynamic_read_energy': self.dynamic_read_energy, 'dynamic_write_energy': self.dynamic_write_energy,
                 'static_read_energy': self.static_read_energy, 'static_write_energy': self.static_write_energy,
                 'read_energy':self.read_energy, 'write_energy': self.write_energy,
                 'total_energy':self.total_energy, 'average_power':self.average_power}        
