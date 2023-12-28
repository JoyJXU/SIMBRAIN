import torch
from typing import Iterable, Optional, Union

class Power(torch.nn.Module):

    def __init__(
        self,
        mem_device: dict = {},
        shape: Optional[Iterable[int]] = None,
        memristor_info_dict: dict = {},
        **kwargs,
    ) -> None:

        super().__init__()
    
        self.shape = shape    

        self.device_name = mem_device['device_name']
        self.device_structure = mem_device['device_structure'] 
        self.register_buffer("mem_c", torch.Tensor())
        self.register_buffer("mem_v", torch.Tensor())
        self.register_buffer("readEnergy_dynamic", torch.Tensor())
        self.register_buffer("readEnergy_static", torch.Tensor())
        self.register_buffer("readEnergy", torch.Tensor())
        self.register_buffer("total_resistance", torch.Tensor())        
        self.register_buffer("total_wire_resistance", torch.Tensor())
        self.memristor_info_dict = memristor_info_dict

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.
    
        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size

        self.mem_c = torch.zeros(batch_size, *self.shape, device=self.mem_c.device)
        self.mem_v = torch.zeros(batch_size, *self.shape, device=self.mem_v.device)
        self.total_resistance = torch.zeros(batch_size, *self.shape, device=self.total_resistance.device)
        self.total_wire_resistance = torch.zeros(self.shape, device=self.total_wire_resistance.device)
        self.readEnergy_dynamic = torch.zeros(batch_size, *(self.shape[0],), device=self.readEnergy_dynamic.device)
        self.readEnergy_static = torch.zeros(batch_size, *(self.shape[0],), device=self.readEnergy_static.device)
        self.readEnergy = torch.zeros(batch_size, *self.shape, device=self.readEnergy.device)

    def read_energy_dynamic_trace(self):
        mem_info = self.memristor_info_dict[self.device_name]
        arrayColSize = mem_info['arrayColSize']
        widthInFeatureSize = mem_info['widthInFeatureSize']
        processNode = mem_info['processNode']

		#1.1calculate numCol
        numCellPerSynapse = 1
        numCol = arrayColSize * numCellPerSynapse
		
		#1.2calculate cellWidth
        cellWidth = widthInFeatureSize
		
		#1.3calculate featureSize
        _featureSizeInNano = processNode #technology node in nm
        featureSize = _featureSizeInNano * 1e-9
		
		#1.4calculate lengthRow
        lengthRow = numCol * cellWidth * featureSize

		#1.5calculate wireCapRow
        wireCapRow = lengthRow *0.2e-15/1e-6

		#2calculate readEnergy_dynamic
        self.readEnergy_dynamic = (wireCapRow  * (self.mem_v * self.mem_v)).squeeze()

        return self.readEnergy_dynamic
    
    def read_energy_static_trace(self):
        mem_info = self.memristor_info_dict[self.device_name]
        delta_t = mem_info['delta_t']
        readPulseWidth = (1/4) * delta_t
        self.readEnergy_static = (self.mem_v * self.mem_c * self.mem_v * readPulseWidth).squeeze()
        return self.readEnergy_static
    
    def read_energy_static_crossbar(self):
        mem_info = self.memristor_info_dict[self.device_name]
        delta_t = mem_info['delta_t']
        wireResistanceUnit = mem_info['wireResistanceUnit']
        readPulseWidth = (1/4) * delta_t
        self.total_wire_resistance =  (wireResistanceUnit * (torch.arange(1, self.shape[1] + 1) + torch.arange(self.shape[0],0,-1)[:,None])).cuda()
        self.total_wire_resistance = torch.stack([self.total_wire_resistance] * self.batch_size)
        self.total_resistance = self.total_wire_resistance + 1/self.mem_c
        # TODO: check the calculation of total_current
        total_current = torch.sum(self.mem_v/self.total_resistance, dim=1)
        total_current = torch.unsqueeze(total_current, 1)
        self.readEnergy_static = (total_current * self.mem_v * readPulseWidth).squeeze()
        return self.readEnergy_static
    
    def read_energy(self):
        if self.device_structure == 'trace':
            return self.read_energy_dynamic_trace() + self.read_energy_static_trace()
        elif self.device_structure == 'crossbar':
            return self.read_energy_dynamic_crossbar() + self.read_energy_static_crossbar()
        else:
            print("read_energy_calcaulate_wrong!")