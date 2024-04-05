from typing import Iterable, Optional, Union
from simbrain.memarray import MemristorArray
import json
import pickle
import torch
import math
from typing import Tuple
import torch.nn.functional as F
import matplotlib.pyplot as plt

class Mapping(torch.nn.Module):
    # language=rst
    """
    Abstract base class for mapping neural networks to memristor arrays.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the layer.
        """
        super().__init__()

        self.sim_params = sim_params
        self.device_name = sim_params['device_name']
        self.device_structure = sim_params['device_structure']
        self.input_bit = sim_params['input_bit']

        if self.device_structure == 'trace':
            self.shape = [1, 1]  # Shape of the memristor crossbar
            for element in shape:
                self.shape[1] *= element
            self.shape = tuple(self.shape)
        elif self.device_structure in {'crossbar', 'mimo'}:
            self.shape = shape
        else:
            raise Exception("Only trace, mimo and crossbar architecture are supported!")
        
        self.register_buffer("mem_v", torch.Tensor())
        self.register_buffer("mem_x_read", torch.Tensor())
        self.register_buffer("mem_t", torch.Tensor())

        with open('../../memristor_device_info.json', 'r') as f:
            self.memristor_info_dict = json.load(f)
        assert self.device_name in self.memristor_info_dict.keys(), "Invalid Memristor Device!"
        self.Gon = self.memristor_info_dict[self.device_name]['G_on']
        self.Goff = self.memristor_info_dict[self.device_name]['G_off']
        self.v_read = self.memristor_info_dict[self.device_name]['v_read']

        self.trans_ratio = 1 / (self.Goff - self.Gon)

        self.batch_size = None
        self.learning = None

        self.sim_power = {}
        self.sim_area = {}


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when memristor is used to mapping traces.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.mem_v = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)
        self.mem_x_read = torch.zeros(batch_size, 1, self.shape[1], device=self.mem_x_read.device)
        self.mem_t = torch.zeros(batch_size, *self.shape, device=self.mem_t.device)
       

    def update_SAF_mask(self) -> None:
        self.mem_array.update_SAF_mask()


class STDPMapping(Mapping):
    # language=rst
    """
    Mapping STDP (Bindsnet) to memristor arrays.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the memristor array.
        """
        super().__init__(
            sim_params=sim_params,
            shape=shape
        )

        self.mem_array = MemristorArray(sim_params=sim_params, shape=self.shape,
                                        memristor_info_dict=self.memristor_info_dict)
        self.batch_interval = sim_params['batch_interval']

        self.register_buffer("mem_v_read", torch.Tensor())
        self.register_buffer("x", torch.Tensor())
        self.register_buffer("s", torch.Tensor())
        self.vneg = 0
        self.vpos = 0
        self.trace_decay = 0


    def set_batch_size_stdp(self, batch_size, learning) -> None:
        self.learning = learning
        self.set_batch_size(batch_size)
        self.mem_array.set_batch_size(batch_size=self.batch_size)

        self.mem_v_read = torch.zeros(1, batch_size, 1, self.shape[0], device=self.mem_v_read.device)
        self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)
        self.s = torch.zeros(batch_size, *self.shape, device=self.s.device)

        if self.learning:
            mem_t_matrix = (self.batch_interval * torch.arange(self.batch_size, device=self.mem_t.device))
            self.mem_t[:, :, :] = mem_t_matrix.view(-1, 1, 1)
        else:
            self.mem_t.fill_(torch.min(self.mem_t_batch_update[:]))

        self.mem_array.mem_t = self.mem_t


    def voltage_generation(self, trace_decay, plot) -> None:
        # Simulation Setup
        points = 150
        spike = torch.zeros(points)
        ori_trace = torch.zeros(points)
        mem_x = torch.zeros(points)

        # STDP Setup
        spike[10] = 1

        # Original Trace
        for i in range(len(spike) - 1):
            ori_trace[i + 1] = ori_trace[i] * trace_decay + spike[i]

            if ori_trace[i + 1] > 1:
                ori_trace[i + 1] = 1

        # Memristor-based Trace
        test_array = MemristorArray(sim_params=self.sim_params, shape=(1, 1), memristor_info_dict=self.memristor_info_dict)
        test_array.set_batch_size(batch_size=1)
        mem_info = self.memristor_info_dict[self.device_name]

        dt = mem_info['delta_t'] * mem_info['duty_ratio']
        k_off = mem_info['k_off']
        v_off = mem_info['v_off']
        alpha_off = mem_info['alpha_off']
        v_pos = v_off * (math.pow(1 / (dt * k_off), 1.0 / alpha_off) + 1)

        if mem_info['P_on'] == 1:
            k_on = mem_info['k_on']
            v_on = mem_info['v_on']
            alpha_on = mem_info['alpha_on']
            v_neg = v_on * (math.pow((trace_decay - 1) / (dt * k_on), 1.0 / alpha_on) + 1)
        else:
            # Enable batch processing for searching the best v_neg
            v_on = mem_info['v_on']
            n_test = 500
            v_tensor = torch.arange(v_on, v_on - 0.01 * n_test, -0.01)

            test_array.set_batch_size(batch_size=n_test)
            test_x = torch.zeros(points, n_test, 1, 1)

            for t in range(points - 1):
                mem_s = torch.tensor(spike[t], dtype=torch.float64)
                mem_v = v_tensor if mem_s == 0 else torch.tensor(v_pos).expand(n_test)

                mem_c = test_array.memristor_write(mem_v=mem_v.unsqueeze(1).unsqueeze(2))
                test_x[t + 1] = (mem_c - self.Gon) * self.trans_ratio

            # Compare results
            golden_x = ori_trace.reshape(points, 1, 1, 1).expand(-1, n_test, -1, -1)
            mse = torch.zeros(n_test)
            for i in range(n_test):
                mse[i] = F.mse_loss(test_x[:, i, :, :], golden_x[:, i, :, :])
            min_mse, min_index = torch.min(mse, 0)
            v_neg = float(v_tensor[min_index])

        if plot:
            blue = (47 / 255, 130 / 255, 189 / 255)
            green = (98 / 255, 149 / 255, 61 / 255)

            plt.figure(figsize=(13, 4.5))
            grid = plt.GridSpec(14, 17, wspace=0.5, hspace=0.5)
            ax = plt.subplot(grid[0:14, 0:17])

            test_array.set_batch_size(batch_size=1)
            for t in range(points - 1):
                mem_s = torch.tensor(spike[t], dtype=torch.float64)
                mem_v = mem_s.unsqueeze(0).unsqueeze(0).unsqueeze(1)
                mem_v[mem_v == 0] = v_neg
                mem_v[mem_v == 1] = v_pos

                mem_c = test_array.memristor_write(mem_v=mem_v)

                # mem to nn
                temp_x = (mem_c - self.Gon) * self.trans_ratio
                mem_x[t+1] = temp_x.squeeze()

            # Plot the original trace and memristor trace
            plot_x = range(points)
            # Original
            ax.plot(ori_trace, color=blue, label='Original Trace')
            ax.plot(mem_x, color=green, label='Memristor Trace')
            ax.legend(frameon=False)

            plt.tight_layout()
            plt.savefig('voltage_generation.png', dpi=300, bbox_inches='tight')
            plt.show()

        self.vneg = v_neg
        self.vpos = v_pos


    def mapping_write_stdp(self, s):
        if self.device_structure == 'trace':
            if s.dim() == 4:
                self.s = s.flatten(2, 3)
            elif s.dim() == 2:
                self.s = torch.unsqueeze(s, 1)
        
        # nn to mem
        self.mem_v = self.s.float()
        self.mem_v[self.mem_v == 0] = self.vneg
        self.mem_v[self.mem_v == 1] = self.vpos      

        mem_c = self.mem_array.memristor_write(mem_v=self.mem_v)
        
        # mem to nn
        self.x = (mem_c - self.Gon) * self.trans_ratio

        if self.device_structure == 'trace':
            if s.dim() == 4:
                self.x = self.x.reshape(s.size(0), s.size(1), s.size(2), s.size(3))
            elif s.dim() == 2:
                self.x = self.x.squeeze()

        return self.x


    def mapping_read_stdp(self, s):
        if self.device_structure == 'trace':
            if s.dim() == 4:
                s = s.flatten(2, 3)
            elif s.dim() == 2:
                s = torch.unsqueeze(s, 1)

        # Read Voltage generation
        # For every batch, read is not necessary when there is no spike s
        s_sum = torch.sum(s, dim=2).squeeze()
        s_sum = torch.unsqueeze(s_sum, 1)

        self.mem_v_read.zero_()
        self.mem_v_read[0, s_sum.bool()] = 1

        mem_i = self.mem_array.memristor_read(mem_v=self.mem_v_read)

        # current to trace
        self.mem_x_read = (mem_i/self.v_read - self.Gon) * self.trans_ratio

        self.mem_x_read[0, ~s_sum.bool()] = 0

        return self.mem_x_read


    def reset_memristor_variables(self) -> None:
        # language=rst
        """
        Abstract base class method for resetting state variables.
        """
        self.mem_v.fill_(-self.vpos)
        
        # Adopt large negative pulses to reset the memristor array
        self.mem_array.memristor_write(mem_v=self.mem_v)


    def mem_t_update(self) -> None:
        self.mem_array.mem_t += self.batch_interval * (self.batch_size - 1)


class MimoMapping(Mapping):
    # language=rst
    """
    Mapping MIMO to memristor arrays.
    """

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the memristor array.
        """
        super().__init__(
            sim_params=sim_params,
            shape=shape
        )

        self.register_buffer("write_pulse_no", torch.Tensor())

        with open('../../memristor_lut.pkl', 'rb') as f:
            self.memristor_luts = pickle.load(f)
        assert self.device_name in self.memristor_luts.keys(), "No Look-Up-Table Data Available for the Target Memristor Type!"

        # Corssbar for positive input and positive weight
        self.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and positive weight
        self.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for positive input and negative weight
        self.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and negative weight
        self.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)

        self.batch_interval = sim_params['batch_interval']


    def set_batch_size_mimo(self, batch_size) -> None:
        self.set_batch_size(batch_size)
        self.mem_pos_pos.set_batch_size(batch_size=batch_size)
        self.mem_neg_pos.set_batch_size(batch_size=batch_size)
        self.mem_pos_neg.set_batch_size(batch_size=batch_size)
        self.mem_neg_neg.set_batch_size(batch_size=batch_size)

        self.write_pulse_no = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)

        mem_t_matrix = (self.batch_interval * torch.arange(self.batch_size, device=self.mem_t.device))
        self.mem_t[:, :, :] = mem_t_matrix.view(-1, 1, 1)

        self.mem_pos_pos.mem_t = self.mem_t.clone()
        self.mem_neg_pos.mem_t = self.mem_t.clone()
        self.mem_pos_neg.mem_t = self.mem_t.clone()
        self.mem_neg_neg.mem_t = self.mem_t.clone()


    def mapping_write_mimo(self, target_x):
        # Memristor reset first
        self.mem_v.fill_(-100)  # TODO: check the reset voltage
        # Adopt large negative pulses to reset the memristor array
        # self.mem_array.memristor_reset(mem_v=self.mem_v)
        self.mem_pos_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_pos_neg.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_neg.memristor_reset(mem_v=self.mem_v)

        total_wr_cycle = self.memristor_luts[self.device_name]['total_no']
        write_voltage = self.memristor_luts[self.device_name]['voltage']
        counter = torch.ones_like(self.mem_v)

        # Positive weight write
        matrix_pos = torch.relu(target_x)
        # Vector to Pulse Serial
        self.write_pulse_no = self.m2v(matrix_pos)
        # Matrix to memristor
        # Memristor programming using multiple identical pulses (up to 400)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < self.write_pulse_no) * write_voltage
            self.mem_pos_pos.memristor_write(mem_v=self.mem_v)
            self.mem_neg_pos.memristor_write(mem_v=self.mem_v)

        # Negative weight write
        matrix_neg = torch.relu(target_x * -1)
        # Vector to Pulse Serial
        self.write_pulse_no = self.m2v(matrix_neg)
        # Matrix to memristor
        # Memristor programming using multiple identical pulses (up to 400)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < self.write_pulse_no) * write_voltage
            self.mem_pos_neg.memristor_write(mem_v=self.mem_v)
            self.mem_neg_neg.memristor_write(mem_v=self.mem_v)


    def mapping_read_mimo(self, target_v):
        # target_v shape [write_batch_size, read_batch_size, row_no]
        # increase one dimension of the input by input_bit
        read_sequence = torch.zeros(self.input_bit, *(target_v.shape), device=target_v.device, dtype=bool)

        # positive read sequence generation
        v_read_pos = torch.relu(target_v)
        v_read_pos = torch.round(v_read_pos * (2 ** self.input_bit - 1))
        v_read_pos = torch.clamp(v_read_pos, 0, 2 ** self.input_bit - 1)
        v_read_pos = v_read_pos.to(torch.int64)
        for i in range(self.input_bit):
            bit = torch.bitwise_and(v_read_pos, 2 ** i).bool()
            read_sequence[i] = bit
        v_read_pos = read_sequence.clone()
        read_sequence.zero_()

        # negative read sequence generation
        v_read_neg = torch.relu(target_v * -1)
        v_read_neg = torch.round(v_read_neg * (2 ** self.input_bit - 1))
        v_read_neg = torch.clamp(v_read_neg, 0, 2 ** self.input_bit - 1)
        v_read_neg = v_read_neg.to(torch.int64)
        for i in range(self.input_bit):
            bit = torch.bitwise_and(v_read_neg, 2 ** i).bool()
            read_sequence[i] = bit
        v_read_neg = read_sequence.clone()

        # memrstor sequential read
        mem_i_sequence = self.mem_pos_pos.memristor_read(mem_v=v_read_pos)
        mem_i_sequence -= self.mem_neg_pos.memristor_read(mem_v=v_read_neg)
        mem_i_sequence -= self.mem_pos_neg.memristor_read(mem_v=v_read_pos)
        mem_i_sequence += self.mem_neg_neg.memristor_read(mem_v=v_read_neg)

        # Shift add to get the output current
        mem_i = torch.zeros(self.batch_size, target_v.shape[1], self.shape[1], device=target_v.device)
        for i in range(self.input_bit):
            mem_i += mem_i_sequence[i, :, :, :] * 2 ** i

        # Current to results
        self.mem_x_read = self.trans_ratio * mem_i / (2 ** self.input_bit - 1) / self.v_read

        return self.mem_x_read


    def m2v(self, target_matrix):
        # Target_matrix ranging [0, 1]
        within_range = (target_matrix >= 0) & (target_matrix <= 1)
        assert torch.all(within_range), "The target Matrix Must be in the Range [0, 1]!"

        # Target x to target conductance
        target_c = target_matrix / self.trans_ratio + self.Gon

        # Get access to the look-up-table of the target memristor
        luts = self.memristor_luts[self.device_name]['conductance']

        # Find the nearest conductance value
        c_diff = torch.abs(torch.tensor(luts, device=target_c.device) - target_c.unsqueeze(3))
        nearest_pulse_no = torch.argmin(c_diff, dim=3)

        return nearest_pulse_no


    def mem_t_update(self) -> None:
        self.mem_pos_pos.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_neg_pos.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_pos_neg.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_neg_neg.mem_t += self.batch_interval * (self.batch_size - 1)


    def total_energy_calculation(self) -> None:
        # language=rst
        """
        Calculate total energy for memristor-based architecture. Called when power is reported.
        """
        self.mem_pos_pos.total_energy_calculation()
        self.mem_neg_pos.total_energy_calculation()
        self.mem_pos_neg.total_energy_calculation()
        self.mem_neg_neg.total_energy_calculation()

        self.sim_power = {key: self.mem_pos_pos.power.sim_power[key] + self.mem_neg_pos.power.sim_power[key] +
                               self.mem_pos_neg.power.sim_power[key] + self.mem_neg_neg.power.sim_power[key] if key != 'time'
                          else self.mem_pos_pos.power.sim_power[key]
                          for key in self.mem_pos_pos.power.sim_power.keys()}


    def total_area_calculation(self) -> None:
        # language=rst
        """
        Calculate total area for memristor-based architecture. Called when power is reported.
        """
        self.sim_area = {'mem_area': self.mem_pos_pos.area.array_area + self.mem_neg_pos.area.array_area +
                                     self.mem_pos_neg.area.array_area + self.mem_neg_neg.area.array_area}


class MLPMapping(Mapping):
    # language=rst
    """
    Mapping Multi-Layer Perceptron (MLP) to memristor arrays.
    """

    def __init__(
            self,
            sim_params: dict = {},
            shape: Optional[Iterable[int]] = None,
            **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the memristor array.
        """
        super().__init__(
            sim_params=sim_params,
            shape=shape
        )

        self.register_buffer("norm_ratio", torch.Tensor())
        self.register_buffer("write_pulse_no", torch.Tensor())

        with open('../../memristor_lut.pkl', 'rb') as f:
            self.memristor_luts = pickle.load(f)
        assert self.device_name in self.memristor_luts.keys(), "No Look-Up-Table Data Available for the Target Memristor Type!"

        # Corssbar for positive input and positive weight
        self.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                        memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and positive weight
        self.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                        memristor_info_dict=self.memristor_info_dict)
        # Corssbar for positive input and negative weight
        self.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and negative weight
        self.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)

        self.batch_interval = sim_params['batch_interval']


    def set_batch_size_mlp(self, batch_size) -> None:
        self.set_batch_size(batch_size)
        self.mem_pos_pos.set_batch_size(batch_size=batch_size)
        self.mem_neg_pos.set_batch_size(batch_size=batch_size)
        self.mem_pos_neg.set_batch_size(batch_size=batch_size)
        self.mem_neg_neg.set_batch_size(batch_size=batch_size)

        self.write_pulse_no = torch.zeros(batch_size, *self.shape, device=self.write_pulse_no.device)

        self.norm_ratio = torch.zeros(batch_size, device=self.norm_ratio.device)
        # TODO: For MLP, batch_interval consist of reset + write + read?
        # self.batch_interval = 1 + self.memristor_luts[self.device_name]['total_no'] * self.shape[0] + self.shape[1]
        mem_t_matrix = (self.batch_interval * torch.arange(self.batch_size, device=self.mem_t.device))
        self.mem_t[:, :, :] = mem_t_matrix.view(-1, 1, 1)

        self.mem_pos_pos.mem_t = self.mem_t.clone()
        self.mem_neg_pos.mem_t = self.mem_t.clone()
        self.mem_pos_neg.mem_t = self.mem_t.clone()
        self.mem_neg_neg.mem_t = self.mem_t.clone()


    def mapping_write_mlp(self, target_x):
        # Memristor reset first
        self.mem_v.fill_(-100)  # TODO: check the reset voltage
        # Adopt large negative pulses to reset the memristor array
        self.mem_pos_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_pos_neg.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_neg.memristor_reset(mem_v=self.mem_v)

        # Transform target_x to [0, 1]
        self.norm_ratio = torch.max(torch.abs(target_x.reshape(target_x.shape[0], -1)), dim=1)[0]
        total_wr_cycle = self.memristor_luts[self.device_name]['total_no']
        write_voltage = self.memristor_luts[self.device_name]['voltage']
        counter = torch.ones_like(self.mem_v)

        # Positive weight write
        matrix_pos = torch.relu(target_x)
        # Vector to Pulse Serial
        self.write_pulse_no = self.m2v(matrix_pos / self.norm_ratio)
        # Matrix to memristor
        # Memristor programming using multiple identical pulses (up to 400)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < self.write_pulse_no) * write_voltage
            self.mem_pos_pos.memristor_write(mem_v=self.mem_v)
            self.mem_neg_pos.memristor_write(mem_v=self.mem_v)

        # Negative weight write
        matrix_neg = torch.relu(target_x * -1)
        # Vector to Pulse Serial
        self.write_pulse_no = self.m2v(matrix_neg / self.norm_ratio)
        # Matrix to memristor
        # Memristor programming using multiple identical pulses (up to 400)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < self.write_pulse_no) * write_voltage
            self.mem_pos_neg.memristor_write(mem_v=self.mem_v)
            self.mem_neg_neg.memristor_write(mem_v=self.mem_v)


    def mapping_read_mlp(self, target_v):
        # Get normalization ratio
        read_norm = torch.max(torch.abs(target_v), dim=1)[0]

        # increase one dimension of the input by input_bit
        read_sequence = torch.zeros(self.input_bit, *(target_v.shape), device=target_v.device)

        # positive read sequence generation
        v_read_pos = torch.relu(target_v)
        v_read_pos = v_read_pos / read_norm.unsqueeze(1)
        v_read_pos = torch.round(v_read_pos * (2 ** self.input_bit - 1))
        v_read_pos = torch.clamp(v_read_pos, 0, 2 ** self.input_bit - 1)
        v_read_pos = v_read_pos.to(torch.uint8)
        for i in range(self.input_bit):
            bit = torch.bitwise_and(v_read_pos, 2 ** i).bool()
            read_sequence[i] = bit
        v_read_pos = read_sequence.clone()
        read_sequence.zero_()

        # negative read sequence generation
        v_read_neg = torch.relu(target_v * -1)
        v_read_neg = v_read_neg / read_norm.unsqueeze(1)
        v_read_neg = torch.round(v_read_neg * (2 ** self.input_bit - 1))
        v_read_neg = torch.clamp(v_read_neg, 0, 2 ** self.input_bit - 1)
        v_read_neg = v_read_neg.to(torch.uint8)
        for i in range(self.input_bit):
            bit = torch.bitwise_and(v_read_neg, 2 ** i).bool()
            read_sequence[i] = bit
        v_read_neg = read_sequence.clone()

        # memrstor sequential read
        mem_i_sequence = self.mem_pos_pos.memristor_read(mem_v=v_read_pos.unsqueeze(1))
        mem_i_sequence -= self.mem_neg_pos.memristor_read(mem_v=v_read_neg.unsqueeze(1))
        mem_i_sequence -= self.mem_pos_neg.memristor_read(mem_v=v_read_pos.unsqueeze(1))
        mem_i_sequence += self.mem_neg_neg.memristor_read(mem_v=v_read_neg.unsqueeze(1))

        # Shift add to get the output current
        mem_i = torch.zeros(self.batch_size, target_v.shape[0], self.shape[1], device=target_v.device)
        for i in range(self.input_bit):
            mem_i += mem_i_sequence[i, :, :, :] * 2**i

        # Current to results
        self.mem_x_read = read_norm.unsqueeze(1) / (
                    2 ** self.input_bit - 1) * self.norm_ratio * self.trans_ratio * mem_i / self.v_read

        return self.mem_x_read.squeeze(0)


    def m2v(self, target_matrix):
        # Target_matrix ranging [0, 1]
        within_range = (target_matrix >= 0) & (target_matrix <= 1)
        assert torch.all(within_range), "The target Matrix Must be in the Range [0, 1]!"

        # Target x to target conductance
        target_c = target_matrix / self.trans_ratio + self.Gon

        # Get access to the look-up-table of the target memristor
        luts = self.memristor_luts[self.device_name]['conductance']

        # Find the nearest conductance value
        c_diff = torch.abs(torch.tensor(luts, device=target_c.device) - target_c.unsqueeze(3))
        nearest_pulse_no = torch.argmin(c_diff, dim=3)

        return nearest_pulse_no


    def mem_t_update(self) -> None:
        self.mem_pos_pos.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_neg_pos.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_pos_neg.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_neg_neg.mem_t += self.batch_interval * (self.batch_size - 1)


    def total_energy_calculation(self) -> None:
        # language=rst
        """
        Calculate total energy for memristor-based architecture. Called when power is reported.
        """
        self.mem_pos_pos.total_energy_calculation()
        self.mem_neg_pos.total_energy_calculation()
        self.mem_pos_neg.total_energy_calculation()
        self.mem_neg_neg.total_energy_calculation()

        self.sim_power = {key: self.mem_pos_pos.power.sim_power[key] + self.mem_neg_pos.power.sim_power[key] +
                               self.mem_pos_neg.power.sim_power[key] + self.mem_neg_neg.power.sim_power[key]
                          for key in self.mem_pos_pos.power.sim_power.keys()}


    def total_area_calculation(self) -> None:
        # language=rst
        """
        Calculate total area for memristor-based architecture. Called when power is reported.
        """
        self.sim_area = {'mem_area': self.mem_pos_pos.area.array_area + self.mem_neg_pos.area.array_area +
                               self.mem_pos_neg.area.array_area + self.mem_neg_neg.area.array_area}


class CNNMapping(Mapping):
    # language=rst
    """
    Mapping convolutional layers (Conv2D) to memristor arrays.
    """

    def __init__(
            self,
            sim_params: dict = {},
            shape: Optional[Iterable[int]] = None,
            **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.
        :param sim_params: Memristor device to be used in learning.
        :param shape: The dimensionality of the memristor array.
        """
        super().__init__(
            sim_params=sim_params,
            shape=shape
        )

        self.register_buffer("norm_ratio", torch.Tensor())
        # self.register_buffer("write_pulse_no", torch.Tensor())

        with open('../../memristor_lut.pkl', 'rb') as f:
            self.memristor_luts = pickle.load(f)
        assert self.device_name in self.memristor_luts.keys(), "No Look-Up-Table Data Available for the Target Memristor Type!"

        # Corssbar for positive input and positive weight
        self.mem_pos_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and positive weight
        self.mem_neg_pos = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for positive input and negative weight
        self.mem_pos_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)
        # Corssbar for negative input and negative weight
        self.mem_neg_neg = MemristorArray(sim_params=sim_params, shape=self.shape,
                                          memristor_info_dict=self.memristor_info_dict)

        self.batch_interval = sim_params['batch_interval']

    def set_batch_size_cnn(self, batch_size) -> None:
        self.set_batch_size(batch_size)
        self.mem_pos_pos.set_batch_size(batch_size=batch_size)
        self.mem_neg_pos.set_batch_size(batch_size=batch_size)
        self.mem_pos_neg.set_batch_size(batch_size=batch_size)
        self.mem_neg_neg.set_batch_size(batch_size=batch_size)

        # self.write_pulse_no = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)
        self.norm_ratio_pos = torch.zeros(batch_size, device=self.norm_ratio.device)
        self.norm_ratio_neg = torch.zeros(batch_size, device=self.norm_ratio.device)
        # self.batch_interval = 1 + self.memristor_luts[self.device_name]['total_no'] * self.shape[0] + self.shape[1]
        mem_t_matrix = (self.batch_interval * torch.arange(self.batch_size, device=self.mem_t.device))
        self.mem_t[:, :, :] = mem_t_matrix.view(-1, 1, 1)

        self.mem_pos_pos.mem_t = self.mem_t.clone()
        self.mem_neg_pos.mem_t = self.mem_t.clone()
        self.mem_pos_neg.mem_t = self.mem_t.clone()
        self.mem_neg_neg.mem_t = self.mem_t.clone()

    def mapping_write_cnn(self, target_x):
        # Memristor reset first
        self.mem_v.fill_(-100)  # TODO: check the reset voltage
        # Adopt large negative pulses to reset the memristor array
        self.mem_pos_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_pos.memristor_reset(mem_v=self.mem_v)
        self.mem_pos_neg.memristor_reset(mem_v=self.mem_v)
        self.mem_neg_neg.memristor_reset(mem_v=self.mem_v)

        # Transform target_x to [0, 1]
        self.norm_ratio_pos = torch.max(torch.relu(target_x).reshape(target_x.shape[0], -1), dim=1)[0]
        self.norm_ratio_neg = torch.max(torch.relu(target_x * -1).reshape(target_x.shape[0], -1), dim=1)[0]
        total_wr_cycle = self.memristor_luts[self.device_name]['total_no']
        write_voltage = self.memristor_luts[self.device_name]['voltage']
        counter = torch.ones_like(self.mem_v)

        # Positive weight write
        matrix_pos = torch.relu(target_x)
        # Vector to Pulse Serial
        write_pulse_no = self.m2v(matrix_pos / self.norm_ratio_pos)
        # Matrix to memristor
        # Memristor programming using multiple identical pulses (up to 400)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < write_pulse_no) * write_voltage
            self.mem_pos_pos.memristor_write(mem_v=self.mem_v)
            self.mem_neg_pos.memristor_write(mem_v=self.mem_v)

        # Negative weight write
        matrix_neg = torch.relu(target_x * -1)
        # Vector to Pulse Serial
        write_pulse_no = self.m2v(matrix_neg / self.norm_ratio_neg)
        # Matrix to memristor
        # Memristor programming using multiple identical pulses (up to 400)
        for t in range(total_wr_cycle):
            self.mem_v = ((counter * t) < write_pulse_no) * write_voltage
            self.mem_pos_neg.memristor_write(mem_v=self.mem_v)
            self.mem_neg_neg.memristor_write(mem_v=self.mem_v)


    def mapping_read_cnn(self, target_v):
        # Get normalization ratio
        read_norm = torch.max(torch.abs(target_v), dim=1)[0]

        # positive read sequence generation
        v_read_pos = torch.relu(target_v)
        v_read_pos = v_read_pos / read_norm.unsqueeze(1)
        v_read_pos = torch.round(v_read_pos * (2 ** self.input_bit - 1))
        v_read_pos = torch.clamp(v_read_pos, 0, 2 ** self.input_bit - 1)
        v_read_pos = v_read_pos.to(torch.uint8)
        # increase one dimension of the input by input_bit
        read_sequence = torch.zeros(self.input_bit, *(target_v.shape), device=target_v.device, dtype=bool)
        for i in range(self.input_bit):
            bit = torch.bitwise_and(v_read_pos, 2 ** i).bool()
            read_sequence[i] = bit
        v_read_pos = read_sequence.clone()
        read_sequence.zero_()
        bit = None

        # negative read sequence generation
        v_read_neg = torch.relu(target_v * -1)
        v_read_neg = v_read_neg / read_norm.unsqueeze(1)
        v_read_neg = torch.round(v_read_neg * (2 ** self.input_bit - 1))
        v_read_neg = torch.clamp(v_read_neg, 0, 2 ** self.input_bit - 1)
        v_read_neg = v_read_neg.to(torch.uint8)
        # increase one dimension of the input by input_bit
        # read_sequence = torch.zeros(self.input_bit, *(target_v.shape), device=target_v.device, dtype=bool)
        for i in range(self.input_bit):
            bit = torch.bitwise_and(v_read_neg, 2 ** i).bool()
            read_sequence[i] = bit
        v_read_neg = read_sequence.clone()
        read_sequence = None
        bit = None

        mem_i_sequence = self.mem_pos_pos.memristor_read(mem_v=v_read_pos.unsqueeze(1))
        mem_i_sequence -= self.mem_neg_pos.memristor_read(mem_v=v_read_neg.unsqueeze(1))

        # Shift add to get the output current
        mem_i = torch.zeros(self.batch_size, target_v.shape[0], self.shape[1], device=target_v.device)
        for i in range(self.input_bit):
            mem_i += mem_i_sequence[i, :, :, :] * 2 ** i

        # Current to results
        self.mem_x_read = self.norm_ratio_pos * self.trans_ratio * (
                    read_norm.unsqueeze(1) / (2 ** self.input_bit - 1) / self.v_read * mem_i - (
                        target_v.sum(dim=1) * self.Gon).unsqueeze(0).unsqueeze(2))

        mem_i_sequence = self.mem_pos_neg.memristor_read(mem_v=v_read_pos.unsqueeze(1))
        mem_i_sequence -= self.mem_neg_neg.memristor_read(mem_v=v_read_neg.unsqueeze(1))

        # Shift add to get the output current
        mem_i.zero_()
        for i in range(self.input_bit):
            mem_i += mem_i_sequence[i, :, :, :] * 2 ** i

        # Current to results
        self.mem_x_read -= self.norm_ratio_neg * self.trans_ratio * (
                    read_norm.unsqueeze(1) / (2 ** self.input_bit - 1) / self.v_read * mem_i - (
                        target_v.sum(dim=1) * self.Gon).unsqueeze(0).unsqueeze(2))

        return self.mem_x_read.squeeze(0)


    def m2v(self, target_matrix):
        # Target_matrix ranging [0, 1]
        within_range = (target_matrix >= 0) & (target_matrix <= 1)
        assert torch.all(within_range), "The target Matrix Must be in the Range [0, 1]!"

        # Target x to target conductance
        target_c = target_matrix / self.trans_ratio + self.Gon

        # Get access to the look-up-table of the target memristor
        luts = self.memristor_luts[self.device_name]['conductance']

        # Find the nearest conductance value
        len_luts = len(luts)
        section = 1 + (len_luts - 1) // 100
        seg_len = len_luts // section
        nearest_pulse_no = torch.zeros_like(target_c)
        for i in range(section):
            c_diff = torch.abs(torch.tensor(luts[(i * seg_len):(i * seg_len + seg_len + 1)], device=target_c.device) - target_c.unsqueeze(3))
            nearest_pulse_no += torch.argmin(c_diff, dim=3)
            c_diff = None

        return nearest_pulse_no


    def mem_t_update(self) -> None:
        self.mem_pos_pos.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_neg_pos.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_pos_neg.mem_t += self.batch_interval * (self.batch_size - 1)
        self.mem_neg_neg.mem_t += self.batch_interval * (self.batch_size - 1)


    def total_energy_calculation(self) -> None:
        # language=rst
        """
        Calculate total energy for memristor-based architecture. Called when power is reported.
        """
        self.mem_pos_pos.total_energy_calculation()
        self.mem_neg_pos.total_energy_calculation()
        self.mem_pos_neg.total_energy_calculation()
        self.mem_neg_neg.total_energy_calculation()

        self.sim_power = {key: self.mem_pos_pos.power.sim_power[key] + self.mem_neg_pos.power.sim_power[key] +
                               self.mem_pos_neg.power.sim_power[key] + self.mem_neg_neg.power.sim_power[key]
                          for key in self.mem_pos_pos.power.sim_power.keys()}


    def total_area_calculation(self) -> None:
        # language=rst
        """
        Calculate total area for memristor-based architecture. Called when power is reported.
        """
        self.sim_area = {'mem_area': self.mem_pos_pos.area.array_area + self.mem_neg_pos.area.array_area +
                               self.mem_pos_neg.area.array_area + self.mem_neg_neg.area.array_area}

