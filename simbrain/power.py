import torch
from typing import Iterable, Optional, Union
import pickle

class Power(torch.nn.Module):
    # language=rst
    """
    Abstract base class for power estimation of memristor crossbar.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        memristor_info_dict: dict = {},
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
        :param length_row: The physical length of the horizontal wire in the crossbar.
        :param length_col: The physical length of the vertical wire in the crossbar.
        """
        super().__init__()
    
        self.shape = shape    
        self.device_name = sim_params['device_name']
        self.device_structure = sim_params['device_structure']
        self.average_power = 0
        self.total_energy = 0
        self.read_energy = 0
        self.write_energy = 0
        self.dynamic_read_energy = 0
        self.dynamic_write_energy = 0
        self.static_read_energy = 0
        self.static_write_energy = 0
        self.register_buffer("selected_write_energy", torch.Tensor())
        self.register_buffer("half_selected_write_energy", torch.Tensor())
        
        self.wire_cap_row = length_row * 0.2e-15/1e-6
        self.wire_cap_col = length_col * 0.2e-15/1e-6
        self.dt = memristor_info_dict[self.device_name]['delta_t']
    
        self.sim_power = {}

        with open('../../memristor_lut.pkl', 'rb') as f:
            self.memristor_luts = pickle.load(f)
        assert self.device_name in self.memristor_luts.keys(), "No Look-Up-Table Data Available for the Target Memristor Type!"
        

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.selected_write_energy = torch.zeros(batch_size, 1, self.shape[1], device=self.selected_write_energy.device)
        self.half_selected_write_energy = torch.zeros(batch_size, *self.shape, device=self.half_selected_write_energy.device)
        
    
    def read_energy_calculation(self, mem_v_read, mem_i) -> None:
        # language=rst
        """
        Calculate read energy for memrisotr crossbar. Called when the crossbar is read.

        :param mem_v_read: Read voltage, shape [batchsize, read_no=1, crossbar_row].
        :param mem_i: Read current of every memristor, shape [batchsize, read_no=1, crossbar_row, crossbar_col].
        """
        self.dynamic_read_energy += torch.sum(mem_v_read * mem_v_read * self.wire_cap_row)
        self.static_read_energy += torch.sum(mem_i * mem_v_read[:, :, :, None] * self.dt * 1/2)
        self.read_energy = self.dynamic_read_energy + self.static_read_energy


    def write_energy_calculation(self, mem_v, mem_c, mem_c_pre, total_wire_resistance) -> None:
        # language=rst
        """
        Calculate write energy for memrisotr crossbar. Called when the crossbar is wrote.

        :param mem_v: Write voltage, shape [batchsize, crossbar_row, crossbar_col].
        :param mem_c: Memristor crossbar conductance after write, shape [batchsize, crossbar_row, crossbar_col].
        :param mem_c_pre: Memristor crossbar conductance before write, shape [batchsize, crossbar_row, crossbar_col].
        :param total_wire_resistance: Wire resistance for every memristor in the crossbar, shape [batchsize, crossbar_row, crossbar_col].
        """
        if self.device_structure == 'trace':
            mem_r = 1.0 / (1 / 2 * (mem_c + mem_c_pre))
            mem_r = mem_r + total_wire_resistance
            mem_c = 1.0 / mem_r
            self.selected_write_energy = mem_v * mem_v * 1 / 2 * self.dt * mem_c
            self.static_write_energy += torch.sum(self.selected_write_energy)

        elif self.device_structure in {'crossbar', 'mimo'}:
            V_write = self.memristor_luts[self.device_name]['voltage']

            # Col cap dynamic write energy
            self.dynamic_write_energy += torch.sum((V_write - mem_v) * (V_write - mem_v) * self.wire_cap_col)

            # Row cap dynamic write energy
            # 1 selected using V_write; (self.shape[0] - 1) half selected using 1/2 V_write
            self.dynamic_write_energy += self.shape[0] * V_write * V_write * self.wire_cap_row * (1 + 1 / 4 * (
                        self.shape[0] - 1))

            # static write energy
            for write_row in range(self.shape[0]):
                # Selected write energy
                selected_mem_r = 1.0 / (1/2 * (mem_c[:, write_row, :] + mem_c_pre[:, write_row, :]))
                selected_mem_r = selected_mem_r + total_wire_resistance[write_row, :].unsqueeze(0)
                selected_mem_c = 1.0 / selected_mem_r
                self.selected_write_energy = (mem_v[:, write_row, :] * mem_v[:, write_row, :] * 1/2 * self.dt * selected_mem_c).unsqueeze(1)

                # half selected write energy
                half_selected_mem_r = 1.0 / mem_c[:, 0:write_row, :]
                half_selected_mem_r = half_selected_mem_r + total_wire_resistance[0:write_row, :].unsqueeze(0)
                half_selected_mem_c = 1.0 / half_selected_mem_r
                self.half_selected_write_energy[:, 0:write_row, :] = (1/2 * V_write) * (1/2 * V_write) * 1/2 * self.dt * half_selected_mem_c

                self.half_selected_write_energy[:, write_row, :] = torch.zeros_like(self.selected_write_energy).squeeze(1)

                half_selected_mem_r = 1.0 / mem_c_pre[:, write_row+1:, :]
                half_selected_mem_r = half_selected_mem_r + total_wire_resistance[write_row+1:, :].unsqueeze(0)
                half_selected_mem_c = 1.0 / half_selected_mem_r
                self.half_selected_write_energy[:, write_row+1:, :] = (1 / 2 * V_write) * (1 / 2 * V_write) * 1/2 * self.dt * half_selected_mem_c

                self.static_write_energy += torch.sum(self.selected_write_energy) + torch.sum(self.half_selected_write_energy)

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
