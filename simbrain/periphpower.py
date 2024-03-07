import torch
from typing import Iterable, Optional, Union

class PeriphPower(torch.nn.Module):
    # language=rst
    """
    Abstract base class for power estimation of memristor crossbar.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        CMOS_tech_info_dict: dict = {},
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the crossbar.
        :param memristor_info_dict: The parameters of the memristor device.
        :param length_row: The physical length of the horizontal wire in the crossbar.
        :param length_col: The physical length of the vertical wire in the crossbar.
        """
        super().__init__()
        self.shape = shape
        self.sim_params = sim_params
        self.CMOS_technode = sim_params['CMOS_technode']
        self.device_roadmap = sim_params['device_roadmap']
        self.input_bit = sim_params['input_bit']
        self.CMOS_tech_info_dict = CMOS_tech_info_dict
        self.vdd = self.CMOS_tech_info_dict[self.device_roadmap][self.CMOS_technode]['vdd']
        self.switch_matrix_cap_tg_drain = 1 # TODO
        self.switch_matrix_cap_gateN = 1 # TODO
        self.switch_matrix_cap_gateP = 1 # TODO
        self.DFF_cap_inv_input = 1 #TODO
        self.DFF_cap_inv_output = 1 #TODO
        self.DFF_cap_tg_gateN = 1 #TODO
        self.DFF_cap_tg_gateP = 1 #TODO
        self.DFF_cap_tg_drain = 1 #TODO
        self.switch_matrix_energy = 0
        self.DFF_energy = 0 

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        
        
    def switch_matrix_read_energy_calculation(self, activity_read, mem_v, mem_v_amp) -> None:
        read_times = mem_v.shape[0] * mem_v.shape[1]
        self.switch_matrix_energy += (self.switch_matrix_cap_tg_drain * 3) * mem_v_amp * mem_v_amp * self.input_bit * activity_read * read_times
        self.switch_matrix_energy += (self.switch_matrix_cap_gateN + self.switch_matrix_cap_gateN) * self.vdd * self.vdd * self.input_bit * activity_read * read_times
        self.switch_matrix_energy += self.DFF_energy_calculation(DFF_num=self.shape[0], DFF_bit=self.input_bit) * read_times
    
    
    def DFF_energy_calculation(self, DFF_num, DFF_bit) -> None:
        self.DFF_energy = 0
        # Assume input D=1 and the energy of CLK INV and CLK TG are for 1 clock cycles
        # CLK INV (all DFFs have energy consumption)
        self.DFF_energy += (self.DFF_cap_inv_input + self.DFF_cap_inv_output) * self.vdd * self.vdd * 4 * DFF_num
		# CLK TG (all DFFs have energy consumption)
        self.DFF_energy += self.DFF_cap_tg_gateN * self.vdd * self.vdd * 2 * DFF_num
        self.DFF_energy += self.DFF_cap_tg_gateP * self.vdd * self.vdd * 2 * DFF_num
		# D to Q path (only selected DFFs have energy consumption)
        self.DFF_energy += (self.DFF_cap_tg_drain * 3 + self.DFF_cap_inv_input) * self.vdd * self.vdd * DFF_num
        self.DFF_energy += (self.DFF_cap_tg_drain  + self.DFF_cap_inv_output) * self.vdd * self.vdd * DFF_num
        self.DFF_energy += (self.DFF_cap_inv_input + self.DFF_cap_inv_output) * self.vdd * self.vdd * DFF_num

        self.DFF_energy *= DFF_bit
        return self.DFF_energy
    
    def sarADC_energy_calculation(self, mem_i) -> None:
        x = 1
        
    def shift_add_energy_calculation(self, mem_i) -> None:
        x = 1