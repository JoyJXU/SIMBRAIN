import torch
from typing import Iterable, Optional, Union
import json

class Area(torch.nn.Module):
    def __init__(
        self,
        sim_params: dict = {},
        shape: Optional[Iterable[int]] = None,
        memristor_info_dict: dict = {},
        **kwargs,
    ) -> None:
        super().__init__()
    
        #self.shape = shape
        with open('../../power_estimation_info.json', 'r') as file:
            self.power_info_dict = json.load(file)  
        self.widthInFeatureSize = self.power_info_dict['2F']['widthInFeatureSize']
        self.wireWidth = self.power_info_dict['2F']['wireWidth']

        self.arrayColSize = self.shape[1]
        self.arrayRowSize = self.shape[0]

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.
    
        :param batch_size: Mini-batch size.
        """
        #self.batch_size = batch_size

    def cal_area(self):
        numCellPerSynapse = 1
        numCol = self.arrayColSize * numCellPerSynapse
        numRow = self.arrayRowSize

        cellWidth = self.widthInFeatureSize
        featureSize = self.wireWidth * 1e-9

        lengthRow = numCol * cellWidth * featureSize
        lengthCol = numRow * cellWidth * featureSize

        area_value = lengthRow * lengthCol

        return area_value