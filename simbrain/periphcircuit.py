from typing import Iterable, Optional, Union
from simbrain.periphpower import PeriphPower
from simbrain.peripharea import PeriphArea
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
        length_row: float = 0,
        length_col: float = 0,
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
        self.device_structure = sim_params['device_structure']
        self.input_bit = sim_params['input_bit']
        self.ADC_accuracy = sim_params['ADC_accuracy']
        
        self.periph_power = PeriphPower(sim_params=sim_params, shape=self.shape, CMOS_tech_info_dict=self.CMOS_tech_info_dict)
        self.periph_area = PeriphArea(sim_params=sim_params, shape=self.shape)

        ShiftAdder_Area = self.periph_area.ShiftAdder_area_calculation(0, length_row)
        SarADC_Area = self.periph_area.SarADC_area_calculation(0, length_row)
        Switchmatrix_Area_Row = self.periph_area.Switchmatrix_area_calculation(length_col, 0, "ROW_MODE")
        Switchmatrix_Area_Col = self.periph_area.Switchmatrix_area_calculation(0, length_row, "COL_MODE")
        Periph_Area = ShiftAdder_Area + SarADC_Area + Switchmatrix_Area_Row + Switchmatrix_Area_Col

        print("ShiftAdder_Area = ", ShiftAdder_Area)
        print("SarADC_Area = ", SarADC_Area)
        print("Switchmatrix_Area_Row = ", Switchmatrix_Area_Row)
        print("Switchmatrix_Area_Col = ", Switchmatrix_Area_Col)
        print("Periph_Area = ", Periph_Area)

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        
        
    def DAC_read(self, mem_v, sgn, mem_v_amp) -> None:
        
        if self.device_structure == 'trace':
            activity_read = torch.nonzero(mem_v).size(0) / mem_v.numel()
            return mem_v
        else:
            activity_read = 1/2
            
            # Get normalization ratio
            read_norm = torch.max(torch.abs(mem_v), dim=1)[0]
    
            # mem_v shape [write_batch_size, read_batch_size, row_no]
            # increase one dimension of the input by input_bit
            read_sequence = torch.zeros(self.input_bit, *(mem_v.shape), device=mem_v.device, dtype=bool)
    
            # positive read sequence generation
            if sgn == 'pos':
                v_read = torch.relu(mem_v)
            elif sgn == 'neg':
                v_read = torch.relu(mem_v * -1)            
    
            v_read = v_read / read_norm.unsqueeze(1)
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

        self.periph_power.switch_matrix_read_energy_calculation(activity_read=activity_read, mem_v=mem_v, mem_v_amp=mem_v_amp)
        
        return v_read
        

    def ADC_read(self, mem_i_sequence) -> None:

      # Shift add to get the output current
        mem_i = torch.zeros(self.batch_size, mem_i_sequence.shape[2], self.shape[1], device=mem_i_sequence.device)

        # TODO SarADC
        for i in range(self.input_bit):
            mem_i += mem_i_sequence[i, :, :, :] * 2 ** i       
     
        self.periph_power.sarADC_energy_calculation(mem_i=mem_i)
        self.periph_power.shift_add_energy_calculation(mem_i=mem_i)
        
        return mem_i

    
        
    