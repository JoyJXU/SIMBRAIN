from simbrain.mapping import MimoMapping
from typing import Iterable, Optional
import torch

class Interface:
    # language=rst
    """
    Interface class for connecting RTL to pytorch-based simulation framework for memristor-based MIMO.
    """
    def __init__(self, shape: Optional[Iterable[int]] = None):
        self.shape = shape # (no_rows, no_cols)

        # parameter setting for memristor crossbar simulation - constants
        # Memristor Array
        memristor_structure = 'mimo'
        memristor_device = 'new_ferro' # ideal, ferro, new_ferro, or hu
        c2c_variation = False
        d2d_variation = 0 # 0: No d2d variation, 1: both, 2: Gon/Goff only, 3: nonlinearity only
        stuck_at_fault = False
        retention_loss = 0 # retention loss, 0: without it, 1: during pulse, 2: no pluse for a long time
        aging_effect = 0 # 0: No aging effect, 1: equation 1, 2: equation 2
        # Peripheral Circuit
        wire_width = 200 # In practice, wire_width shall be set around 1/2 of the memristor size; Hu: 10um; Ferro:200nm;
        input_bit = 8
        CMOS_technode = 14
        ADC_precision = 32
        ADC_setting = 4 # 2:two memristor crossbars use one ADC; 4:one memristor crossbar use one ADC
        ADC_rounding_function = 'floor'  # floor or round
        device_roadmap = 'HP' # HP: High Performance or LP: Low Power
        temperature = 300
        hardware_estimation = True # area and power estimation

        self.sim_params = {'device_structure': memristor_structure, 'device_name': memristor_device,
                      'c2c_variation': c2c_variation, 'd2d_variation': d2d_variation,
                      'stuck_at_fault': stuck_at_fault, 'retention_loss': retention_loss,
                      'aging_effect': aging_effect, 'wire_width': wire_width, 'input_bit': input_bit,
                      'batch_interval': 1, 'CMOS_technode': CMOS_technode, 'ADC_precision': ADC_precision,
                      'ADC_setting': ADC_setting, 'ADC_rounding_function': ADC_rounding_function,
                      'device_roadmap': device_roadmap, 'temperature': temperature,
                      'hardware_estimation': hardware_estimation}

        # MimoMaping Initialization
        device = 'cpu'
        batch_size = 1
        self.interface = MimoMapping(sim_params=self.sim_params, shape=self.shape)
        self.interface.to(device)
        self.interface.set_batch_size_mimo(batch_size)

    def program(self, matrix):
        # Shape for matrix [write_batch_size, no_row, no_cols]
        self.interface.mapping_write_mimo(target_x=matrix)
        return 1

    def compute(self, vector):
        # Shape for vector [write_batch_size, read_batch_size, no_row]
        output = self.interface.mapping_read_mimo(target_v=vector)
        # Shape for vector [write_batch_size, read_batch_size, no_cols]
        return output


# Instance of Interface
interface_instance = Interface()


def mem_program(value):
    # array to tensor
    matrix_tensor = torch.tensor(value).unsqueeze(0)

    return interface_instance.program(matrix_tensor)


def mem_compute(value):
    # array to tensor
    vector_tensor = torch.tensor(value).unsqueeze(0).unsqueeze(0)

    output_tensor = interface_instance.compute(vector_tensor)
    output_tensor = output_tensor.squeezze(0).squeeze(0)

    # tensor to array
    output = output_tensor.numpy()
    return output

