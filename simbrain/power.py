import torch
from typing import Iterable, Optional, Union
import json

class Power(torch.nn.Module):

    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        memristor_info_dict: dict = {},
        **kwargs,
    ) -> None:

        super().__init__()
    
        self.shape = shape    

        self.device_name = sim_params['device_name']
        self.device_structure = sim_params['device_structure'] 
        #self.processNode = sim_params['processNode'] 
        self.register_buffer("mem_c", torch.Tensor())
        self.register_buffer("mem_cpre", torch.Tensor())
        self.register_buffer("mem_v", torch.Tensor())
        self.register_buffer("mem_t", torch.Tensor())
        self.register_buffer("readEnergy_dynamic", torch.Tensor())
        self.register_buffer("readEnergy_static", torch.Tensor())
        self.register_buffer("readEnergy", torch.Tensor())
        self.register_buffer("writeEnergy_dynamic", torch.Tensor())
        self.register_buffer("writeEnergy_static", torch.Tensor())
        self.register_buffer("writeEnergy", torch.Tensor())
        self.register_buffer("total_resistance", torch.Tensor())        
        self.register_buffer("total_wire_resistance", torch.Tensor())
        self.memristor_info_dict = memristor_info_dict
        self.count_n = 0
        self.total_readEnergy = 0
        self.total_writeEnergy = 0
        self.total_Energy = 0
        self.average_Power = 0

        with open('../../power_estimation_info.json', 'r') as file:
            self.power_info_dict = json.load(file)  
        self.widthInFeatureSize = self.power_info_dict['2F']['widthInFeatureSize']
        self.wireResistanceUnit = self.power_info_dict['2F']['wireResistanceUnit']
        self.wireWidth = self.power_info_dict['2F']['wireWidth']

        self.sim_power = {'readEnergy_static': self.readEnergy_static, 'writeEnergy_static': self.writeEnergy_static,
                 'readEnergy_dynamic': self.readEnergy_dynamic, 'writeEnergy_dynamic': self.writeEnergy_dynamic,
                 'readEnergy':self.readEnergy, 'writeEnergy': self.writeEnergy,
                 'total_readEnergy':self.total_readEnergy, 'total_writeEnergy':self.total_writeEnergy,
                 'total_energy':self.total_Energy, 'average_power':self.average_Power}
        
        self.arrayColSize = self.shape[1]
        self.arrayRowSize = self.shape[0]

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size

        self.mem_c = torch.zeros(batch_size, *self.shape, device=self.mem_c.device)
        self.mem_cpre = torch.zeros(batch_size, *self.shape, device=self.mem_cpre.device)
        self.mem_v = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)
        self.mem_t = torch.zeros(batch_size, *self.shape, device=self.mem_t.device)
        self.total_resistance = torch.zeros(batch_size, *self.shape, device=self.total_resistance.device)
        self.total_wire_resistance = torch.zeros(self.shape, device=self.total_wire_resistance.device)
        self.readEnergy_dynamic = torch.zeros(batch_size, *self.shape, device=self.readEnergy_dynamic.device)
        self.readEnergy_static = torch.zeros(batch_size, *self.shape, device=self.readEnergy_static.device)
        self.readEnergy = torch.zeros(batch_size, *self.shape, device=self.readEnergy.device)
        self.writeEnergy_dynamic = torch.zeros(batch_size, *self.shape, device=self.writeEnergy_dynamic.device)
        self.writeEnergy_static = torch.zeros(batch_size, *self.shape, device=self.writeEnergy_static.device)
        self.writeEnergy = torch.zeros(batch_size, *self.shape, device=self.writeEnergy.device)


    def read_energy_dynamic_trace(self):
        # mem_info = self.memristor_info_dict[self.device_name]
        # if layer == 'X':
        #     arrayColSize = mem_info['arrayColSize_X']
        # elif layer == 'Y':
        #     arrayColSize = mem_info['arrayColSize_Y']    
        # widthInFeatureSize = mem_info['widthInFeatureSize']
        # processNode = mem_info['processNode']

		#1.1calculate numCol
        numCellPerSynapse = 1
        numCol = self.arrayColSize * numCellPerSynapse
		
		#1.2calculate cellWidth
        cellWidth = self.widthInFeatureSize
		
		#1.3calculate featureSize
        featureSize = self.wireWidth * 1e-9
		
		#1.4calculate lengthRow
        lengthRow = numCol * cellWidth * featureSize

		#1.5calculate wireCapRow
        wireCapRow = lengthRow *0.2e-15/1e-6

		#2calculate readEnergy_dynamic
        self.readEnergy_dynamic = (wireCapRow  * (self.mem_v * self.mem_v)).squeeze()
        self.sim_power['readEnergy_dynamic'] = self.readEnergy_dynamic
        return self.readEnergy_dynamic
    
    def read_energy_static_trace(self):
        mem_info = self.memristor_info_dict[self.device_name]
        delta_t = mem_info['delta_t']
        # wireResistanceUnit = mem_info['wireResistanceUnit']
        readPulseWidth = (1/2) * delta_t
        self.readEnergy_static = (self.mem_v * self.mem_c * self.mem_v * readPulseWidth + self.mem_v / (
                    2 * self.wireResistanceUnit) * self.mem_v * readPulseWidth).squeeze()
        self.sim_power['readEnergy_static'] = self.readEnergy_static
        return self.readEnergy_static
    
    def read_energy_static_crossbar(self):
        mem_info = self.memristor_info_dict[self.device_name]
        delta_t = mem_info['delta_t']
        # wireResistanceUnit = mem_info['wireResistanceUnit']
        readPulseWidth = (1/2) * delta_t
        self.total_wire_resistance = (self.wireResistanceUnit * (
                    torch.arange(1, self.shape[1] + 1) + torch.arange(self.shape[0], 0, -1)[:, None])).cuda()
        self.total_wire_resistance = torch.stack([self.total_wire_resistance] * self.batch_size)
        self.total_resistance = self.total_wire_resistance + 1/self.mem_c
        # TODO: check the calculation of total_current
        total_current = torch.sum(self.mem_v/self.total_resistance, dim=1)
        total_current = torch.unsqueeze(total_current, 1)
        self.readEnergy_static = (total_current * self.mem_v * readPulseWidth).squeeze()
        self.sim_power['readEnergy_static'] = self.readEnergy_static
        return self.readEnergy_static
    
    def read_energy(self):
        if self.device_structure == 'trace':
            self.readEnergy = self.read_energy_dynamic_trace() + self.read_energy_static_trace()
            self.sim_power['readEnergy'] = self.readEnergy
        elif self.device_structure == 'crossbar':
            self.readEnergy = self.read_energy_dynamic_crossbar() + self.read_energy_static_crossbar()
            self.sim_power['readEnergy'] = self.readEnergy
        else:
            print("Unsupported Architecture for Read Energy Calculation!")

        self.total_readEnergy += torch.sum(self.sim_power['readEnergy'])
        self.sim_power['total_readEnergy'] = self.total_readEnergy

    def write_energy_dynamic_trace(self):
        # mem_info = self.memristor_info_dict[self.device_name]
        # if layer == 'X':
        #     arrayColSize = mem_info['arrayColSize_X']
        # elif layer == 'Y':
        #     arrayColSize = mem_info['arrayColSize_Y']
        # widthInFeatureSize = mem_info['widthInFeatureSize']
        # processNode = mem_info['processNode']
        
		#1.1calculate numCol, numRow
        numRow = self.arrayRowSize
		#1.2calculate cellWidth
        cellWidth = self.widthInFeatureSize
		
		#1.3calculate featureSize
        featureSize = self.wireWidth * 1e-9
		
		#1.4calculate lengthRow, lengthCol
        lengthCol = numRow * cellWidth * featureSize

		#1.5calculate wireCapRow, wireCapCol
        wireCapCol = lengthCol * 0.2e-15/1e-6

		#2calculate readEnergy_dynamic
        self.writeEnergy_dynamic = ( wireCapCol * (self.mem_v * self.mem_v)).squeeze()
        self.sim_power['writeEnergy_dynamic'] = self.writeEnergy_dynamic
        return self.writeEnergy_dynamic
    
    def write_energy_static_trace(self):
        mem_info = self.memristor_info_dict[self.device_name]
        delta_t = mem_info['delta_t']
        # wireResistanceUnit = mem_info['wireResistanceUnit']
        writePulseWidth = (1/2) * delta_t
        if self.count_n == 0:
            self.mem_cpre = self.mem_c
            self.writeEnergy_static = (self.mem_v * self.mem_c * self.mem_v * writePulseWidth + self.mem_v/(2 * self.wireResistanceUnit) * self.mem_v * writePulseWidth).squeeze()
            self.count_n = 1
        else:
            self.writeEnergy_static = (self.mem_v * (self.mem_c + self.mem_cpre)/2 * self.mem_v * writePulseWidth + self.mem_v/(2 * self.wireResistanceUnit) * self.mem_v * writePulseWidth).squeeze()
            self.mem_cpre = self.mem_c
        self.sim_power['writeEnergy_static'] = self.writeEnergy_static
        return self.writeEnergy_static
    
    def write_energy(self):
        if self.device_structure == 'trace':
            self.writeEnergy = self.write_energy_dynamic_trace() + self.write_energy_static_trace()
            self.sim_power['writeEnergy'] = self.writeEnergy
        elif self.device_structure == 'crossbar':
            self.writeEnergy = self.write_energy_dynamic_crossbar() + self.write_energy_static_crossbar()
            self.sim_power['writeEnergy'] = self.writeEnergy
        else:
            print("Unsupported Architecture for Write Energy Calculation!")

        self.total_writeEnergy += torch.sum(self.sim_power['writeEnergy'])
        self.sim_power['total_writeEnergy'] += self.total_writeEnergy

    def totalEnergy(self):
        self.read_energy()
        self.write_energy()
        self.total_Energy = self.total_readEnergy + self.total_writeEnergy
        self.sim_power['total_Energy'] = self.total_Energy
        self.average_Power = self.total_Energy / (self.mem_t.max())
        self.sim_power['average_Power'] = self.average_Power