from typing import Iterable, Optional, Union
from simbrain.mempower import Power
from simbrain.memarea import Area
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