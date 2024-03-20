from typing import Iterable, Optional, Union
from simbrain.periphpower import PeriphPower
#from simbrain.peripharea import PeriphArea
import torch
import json

class PeriphCircuit(torch.nn.Module):
    # language=rst
    """
    Abstract base class for peripheral circuit.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        CMOS_tech_info_dict: dict = {},
        memristor_info_dict: dict = {},
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the crossbar.
        :param memristor_info_dict: The parameters of the memristor device.
        """
        super().__init__()
    
        self.shape = shape
        self.sim_params = sim_params
        self.CMOS_tech_info_dict = CMOS_tech_info_dict
        self.memristor_info_dict = memristor_info_dict
        self.device_structure = sim_params['device_structure']
        self.input_bit = sim_params['input_bit']
        self.ADC_precision = sim_params['ADC_precision']    
        self.device_name = sim_params['device_name']
        self.Goff = self.memristor_info_dict[self.device_name]['G_off']
        self.read_v_amp = self.memristor_info_dict[self.device_name]['v_read']
        self.periph_power = PeriphPower(sim_params=self.sim_params, shape=self.shape, CMOS_tech_info_dict=self.CMOS_tech_info_dict, memristor_info_dict=self.memristor_info_dict)


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.periph_power.set_batch_size(batch_size=self.batch_size)
        
        
    def DAC_read(self, mem_v, sgn) -> None:
        
        if self.device_structure == 'trace':
            activity_read = torch.nonzero(mem_v).size(0) / mem_v.numel()            
            self.periph_power.switch_matrix_read_energy_calculation(activity_read=activity_read, mem_v=mem_v)
            return mem_v
        else:
            activity_read = 1/2
            
            # Get normalization ratio
            # TODO check the shape
            read_norm = torch.max(torch.abs(mem_v), dim=2)[0]
    
            # mem_v shape [write_batch_size, read_batch_size, row_no]
            # increase one dimension of the input by input_bit
            read_sequence = torch.zeros(self.input_bit, *(mem_v.shape), device=mem_v.device, dtype=bool)
    
            # positive read sequence generation
            if sgn == 'pos':
                v_read = torch.relu(mem_v)
            elif sgn == 'neg':
                v_read = torch.relu(mem_v * -1)            
    
            v_read = v_read / read_norm.unsqueeze(2)
            v_read = torch.round(v_read * (2 ** self.input_bit - 1))
            v_read = torch.clamp(v_read, 0, 2 ** self.input_bit - 1)
            
            if self.input_bit <= 8:
                v_read = v_read.to(torch.uint8)
            else:
                v_read = v_read.to(torch.int64)     
            # TODO 16 32 64   
            for i in range(self.input_bit):
                bit = torch.bitwise_and(v_read, 2 ** i).bool()
                read_sequence[i] = bit
            v_read = read_sequence.clone()
            read_sequence.zero_()
            bit = None
            
            self.periph_power.switch_matrix_read_energy_calculation(activity_read=activity_read, mem_v=mem_v)            
            return v_read
        

    def ADC_read(self, mem_i_sequence, total_wire_resistance) -> None:

      # Shift add to get the output current
        mem_i = torch.zeros(self.batch_size, mem_i_sequence.shape[2], self.shape[1], device=mem_i_sequence.device)
        
        # Plan A: get max and min from mem_i_sequence
        # mem_i_sequence_max, _ = mem_i_sequence.max(dim=-1)
        #mem_i_sequence_max = mem_i_sequence_max.unsqueeze(-1).expand_as(mem_i_sequence)
        # mem_i_sequence_min, _ = mem_i_sequence.min(dim=-1)   
        #mem_i_sequence_min = mem_i_sequence_min.unsqueeze(-1).expand_as(mem_i_sequence)    
        
        # Plan B: calculate the real max and min
        mem_i_sequence_min = torch.zeros_like(mem_i_sequence)
        mem_i_max = torch.sum(self.read_v_amp/(1/self.Goff + total_wire_resistance), dim=0)
        mem_i_sequence_max = mem_i_max.unsqueeze(0).unsqueeze(1).unsqueeze(2).expand_as(mem_i_sequence)
        mem_i_step = (mem_i_sequence_max - mem_i_sequence_min) / (2**self.ADC_precision)
        mem_i_index = torch.where(mem_i_step!=0, (mem_i_sequence - mem_i_sequence_min) / mem_i_step, 0)                           
        mem_i_index = torch.clamp(torch.floor(mem_i_index), 0, 2**self.ADC_precision-1)  
        mem_i_sequence_quantized = mem_i_index * mem_i_step + mem_i_sequence_min    
       
        for i in range(self.input_bit):
            mem_i += mem_i_sequence_quantized[i, :, :, :] * 2 ** i
     
        self.periph_power.sarADC_energy_calculation(mem_i_sequence=mem_i_sequence)
        self.periph_power.shift_add_energy_calculation(mem_i_sequence=mem_i_sequence)
        
        return mem_i
    
    def DAC_col_write(self, mem_v) -> None:
        self.periph_power.switch_matrix_col_write_energy_calculation(mem_v=mem_v)
        return mem_v
        
    
    def DAC_row_write(self, mem_v_amp) -> None:    
        self.periph_power.switch_matrix_row_write_energy_calculation(mem_v_amp=mem_v_amp)

    
    def total_energy_calculation(self, mem_t) -> None:
        self.periph_power.total_energy_calculation(mem_t=mem_t)
    