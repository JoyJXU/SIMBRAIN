from abc import ABC, abstractmethod
from functools import reduce
from operator import mul
from typing import Iterable, Optional, Union

import torch
import json


class Nodes(torch.nn.Module):
    # language=rst
    """
    Abstract base class for groups of neurons.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        learning: bool = True,
        mem_device: dict = {},
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract base class constructor.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param learning: Whether to be in learning or testing.
        :param mem_device: Memristor device to be used in learning.
        """
        super().__init__()

        assert (
            n is not None or shape is not None
        ), "Must provide either no. of neurons or shape of layer"

        if n is None:
            self.n = reduce(mul, shape)  # No. of neurons product of shape.
        else:
            self.n = n  # No. of neurons provided.

        if shape is None:
            self.shape = [self.n]  # Shape is equal to the size of the layer.
        else:
            self.shape = shape  # Shape is passed in as an argument.

        assert self.n == reduce(
            mul, self.shape
        ), "No. of neurons and shape do not match"

        self.traces = traces  # Whether to record synaptic traces.
        self.traces_additive = (
            traces_additive  # Whether to record spike traces additively.
        )
        self.register_buffer("s", torch.ByteTensor())  # Spike occurrences.
        self.register_buffer("mem_v", torch.ByteTensor())

        self.sum_input = sum_input  # Whether to sum all inputs.

        self.device_name = mem_device['device_name']
        self.c2c_variation = mem_device['c2c_variation']
        self.d2d_variation = mem_device['d2d_variation']
        self.stuck_at_fault = mem_device['stuck_at_fault']
        self.retention_loss = mem_device['retention_loss']

        if self.traces:
            self.register_buffer("x", torch.Tensor()) # Firing traces.
            self.register_buffer("mem_x", torch.Tensor())  # Memristor-based firing traces.
            self.register_buffer(
                "tc_trace", torch.tensor(tc_trace)
            )  # Time constant of spike trace decay.
            self.register_buffer(
                "trace_scale", torch.tensor(trace_scale)
            )  # Scaling factor for spike trace.
            self.register_buffer(
                "trace_decay", torch.empty_like(self.tc_trace)
            )  # Set in compute_decays.

            if self.c2c_variation:
                self.register_buffer("normal_absolute", torch.Tensor())
                self.register_buffer("normal_relative", torch.Tensor())

            if self.d2d_variation in [1, 2]:
                self.register_buffer("Gon_d2d", torch.Tensor())
                self.register_buffer("Goff_d2d", torch.Tensor())
            if self.d2d_variation in [1, 3]:
                self.register_buffer("Pon_d2d", torch.Tensor())
                self.register_buffer("Poff_d2d", torch.Tensor())

            if self.stuck_at_fault:
                self.register_buffer("SAF0_mask", torch.Tensor())
                self.register_buffer("SAF1_mask", torch.Tensor())
                
            if self.retention_loss == 2:
                self.register_buffer("mem_v_threshold", torch.Tensor())
                self.register_buffer("mem_loss_time", torch.Tensor())    

        if self.sum_input:
            self.register_buffer("summed", torch.FloatTensor())  # Summed inputs.

        self.dt = None
        self.batch_size = None
        self.trace_decay = None
        self.learning = learning

        if self.device_name != 'trace':
            with open('../../memristor_device_info.json', 'r') as f:
                self.memristor_info_dict = json.load(f)

            assert self.device_name in self.memristor_info_dict.keys(), "Invalid Memristor Device!"

    @abstractmethod
    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Abstract base class method for a single simulation step.

        :param x: Inputs to the layer.
        """
        if self.traces:
            if self.device_name == 'trace':
                # Decay and set spike traces.
                self.x *= self.trace_decay

                if self.traces_additive:
                    self.x += self.trace_scale * self.s.float()
                else:
                    self.x.masked_fill_(self.s.bool(), self.trace_scale)

            else:
                mem_info = self.memristor_info_dict[self.device_name]
                delta_t = mem_info['delta_t']
                k_off = mem_info['k_off']
                k_on = mem_info['k_on']
                v_off = mem_info['v_off']
                v_on = mem_info['v_on']
                alpha_off = mem_info['alpha_off']
                alpha_on = mem_info['alpha_on']
                P_off = mem_info['P_off']
                P_on = mem_info['P_on']
                G_off = mem_info['G_off']
                G_on = mem_info['G_on']
                sigma_relative = mem_info['sigma_relative']
                sigma_absolute = mem_info['sigma_absolute']
                retention_loss_tau = mem_info['retention_loss_tau']
                retention_loss_beta = mem_info['retention_loss_beta']

                trans_ratio = 1 / (G_off - G_on)

                self.mem_v = self.s.float()

                self.mem_v[self.mem_v == 0] = mem_info['vinput_neg']
                self.mem_v[self.mem_v == 1] = mem_info['vinput_pos']

                if self.d2d_variation in [1, 3]:
                    self.mem_x = torch.where(self.mem_v >= v_off, \
                                             self.mem_x + delta_t * (k_off * (self.mem_v / v_off - 1) ** alpha_off) * ( \
                                                         1 - self.mem_x) ** (self.Poff_d2d), self.mem_x)
                        
                    self.mem_x = torch.where(self.mem_v <= v_on, \
                                             self.mem_x + delta_t * (k_on * (self.mem_v / v_on - 1) ** alpha_on) * ( \
                                                 self.mem_x) ** (self.Pon_d2d),self.mem_x)
                    #self.mem_x[self.mem_v >= v_off] = self.mem_x[self.mem_v >= v_off] + delta_t*(k_off*(self.mem_v[self.mem_v >= v_off]/v_off-1)**alpha_off) * torch.pow((1-self.mem_x[self.mem_v >= v_off]),(self.Poff_d2d[self.mem_v >= v_off]))
                    #self.mem_x[self.mem_v <= v_on] = self.mem_x[self.mem_v <= v_on]+delta_t*(k_on*(self.mem_v[self.mem_v <= v_on]/v_on-1)**alpha_on)*torch.pow((self.mem_x[self.mem_v <= v_on]),(self.Pon_d2d[self.mem_v <= v_on]))

                else:
                    #self.mem_x = torch.where(self.mem_v > 0, \
                                             #self.mem_x + delta_t * (k_off * (self.mem_v / v_off - 1) ** alpha_off) * ( \
                                                     #1 - self.mem_x) ** (P_off), \
                                             #self.mem_x + delta_t * (k_on * (self.mem_v / v_on - 1) ** alpha_on) * ( \
                                                 #self.mem_x) ** (P_on))
                    self.mem_x = torch.where(self.mem_v >= v_off, \
                                             self.mem_x + delta_t * (k_off * (self.mem_v / v_off - 1) ** alpha_off) * ( \
                                                         1 - self.mem_x) ** (P_off), self.mem_x)
                        
                    self.mem_x = torch.where(self.mem_v <= v_on, \
                                             self.mem_x + delta_t * (k_on * (self.mem_v / v_on - 1) ** alpha_on) * ( \
                                                 self.mem_x) ** (P_on),self.mem_x)
                    #self.mem_x[self.mem_v >= v_off] = self.mem_x[self.mem_v >= v_off] + \
                                  #delta_t*(k_off*(self.mem_v[self.mem_v >= v_off]/v_off-1)**alpha_off)*(1-self.mem_x[self.mem_v >= v_off])**(P_off)
                    #self.mem_x[self.mem_v <= v_on] = self.mem_x[self.mem_v <= v_on] + \
                                 #delta_t*(k_on*(self.mem_v[self.mem_v <= v_on]/v_on-1)**alpha_on)*(self.mem_x[self.mem_v <= v_on])**(P_on)
                self.mem_x = torch.clamp(self.mem_x, min=0, max=1)

                # Retention Loss
                if self.retention_loss == 1:
                    # G(t) = G(0) * e^(- t*tau)^beta                      
                    self.mem_x = G_off * self.mem_x + G_on * (1 - self.mem_x)
                    self.mem_x *= torch.exp(torch.tensor(-(1/4 * delta_t * retention_loss_tau) ** retention_loss_beta))
                    self.mem_x = torch.clamp(self.mem_x, min = G_on, max = G_off)
                    self.mem_x = (self.mem_x - G_on) * trans_ratio
                    # tau = 0.012478 , beta = 1.066  or  tau = 0.01245 , beta = 1.073
                    
                if self.retention_loss == 2:
                    # dG(t)/dt = - tau^beta * beta * G(t) * t ^ (beta - 1)                    
                    self.mem_v_threshold = (self.mem_v > v_on) & (self.mem_v < v_off)
                    self.mem_loss_time[self.mem_v_threshold] += delta_t
                    self.mem_loss_time[~self.mem_v_threshold] = 0     
                    self.mem_x = G_off * self.mem_x + G_on * (1 - self.mem_x)
                    self.mem_x -= self.mem_x * delta_t * retention_loss_tau ** retention_loss_beta * retention_loss_beta * self.mem_loss_time ** (retention_loss_beta - 1)
                    self.mem_x = torch.clamp(self.mem_x, min = G_on, max = G_off)
                    self.mem_x = (self.mem_x - G_on) * trans_ratio 

                if self.c2c_variation:
                    self.normal_relative.normal_(mean=0., std=sigma_relative)
                    self.normal_absolute.normal_(mean=0., std=sigma_absolute)

                    device_v = torch.mul(self.mem_x, self.normal_relative) + self.normal_absolute
                    self.x2 = self.mem_x + device_v

                    self.x2 = torch.clamp(self.x2, min=0, max=1)

                else:
                    self.x2 = self.mem_x

                if self.stuck_at_fault:
                    self.x2.masked_fill_(self.SAF0_mask, 0)

                    self.x2.masked_fill_(self.SAF1_mask, 1)

                if self.d2d_variation in [1, 2]:
                    self.x = self.Goff_d2d * self.x2 + self.Gon_d2d * (1 - self.x2)
                else:
                    self.x = G_off * self.x2 + G_on * (1 - self.x2)

                self.x = (self.x - G_on) * trans_ratio


        if self.sum_input:
            # Add current input to running sum
            self.summed += x.float()


    def reset_state_variables(self) -> None:
        # language=rst
        """
        Abstract base class method for resetting state variables.
        """
        self.s.zero_()
        self.mem_v.zero_()

        if self.traces:
            self.x.zero_()  # Spike traces.
            self.mem_x.zero_()  # Memristor-based spike traces.

            if self.c2c_variation:
                self.normal_relative.zero_()
                self.normal_absolute.zero_()

        if self.sum_input:
            self.summed.zero_()  # Summed inputs.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Abstract base class method for setting decays.
        """
        self.dt = torch.tensor(dt)
        if self.traces:
            
            self.trace_decay = torch.exp(
                -self.dt / self.tc_trace
            )  # Spike trace decay (per timestep).


    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        self.batch_size = batch_size
        self.s = torch.zeros(
            batch_size, *self.shape, device=self.s.device, dtype=torch.bool
        )
        self.mem_v = torch.zeros(
            batch_size, *self.shape, device=self.mem_v.device, dtype=torch.bool
        )

        if self.traces:
            self.x = torch.zeros(batch_size, *self.shape, device=self.x.device)
            self.mem_x = torch.zeros(batch_size, *self.shape, device=self.mem_x.device)

            if self.c2c_variation:
                self.normal_relative = torch.zeros(batch_size, *self.shape, device=self.normal_relative.device)
                self.normal_absolute = torch.zeros(batch_size, *self.shape, device=self.normal_absolute.device)

            if self.d2d_variation in [1, 2]:
                G_off = self.memristor_info_dict[self.device_name]['G_off']
                G_on = self.memristor_info_dict[self.device_name]['G_on']
                Gon_sigma = self.memristor_info_dict[self.device_name]['Gon_sigma']
                Goff_sigma = self.memristor_info_dict[self.device_name]['Goff_sigma']

                # Initialize
                self.Gon_d2d = torch.zeros(*self.shape, device=self.Gon_d2d.device)
                self.Goff_d2d = torch.zeros(*self.shape, device=self.Goff_d2d.device)
                # Add d2d variation
                self.Gon_d2d.normal_(mean=G_on, std=Gon_sigma)
                self.Goff_d2d.normal_(mean=G_off, std=Goff_sigma)
                # Clipping
                self.Gon_d2d = torch.clamp(self.Gon_d2d, min=0)
                self.Goff_d2d = torch.clamp(self.Goff_d2d, min=0)

                self.Gon_d2d = torch.stack([self.Gon_d2d] * batch_size)
                self.Goff_d2d = torch.stack([self.Goff_d2d] * batch_size)

            if self.d2d_variation in [1, 3]:
                P_off = self.memristor_info_dict[self.device_name]['P_off']
                P_on = self.memristor_info_dict[self.device_name]['P_on']
                Pon_sigma = self.memristor_info_dict[self.device_name]['Pon_sigma']
                Poff_sigma = self.memristor_info_dict[self.device_name]['Poff_sigma']

                # Initialize
                self.Pon_d2d = torch.zeros(*self.shape, device=self.Pon_d2d.device)
                self.Poff_d2d = torch.zeros(*self.shape, device=self.Poff_d2d.device)
                # Add d2d variation
                self.Pon_d2d.normal_(mean=P_on, std=Pon_sigma)
                self.Poff_d2d.normal_(mean=P_off, std=Poff_sigma)
                # Clipping
                self.Pon_d2d = torch.clamp(self.Pon_d2d, min=0)
                self.Poff_d2d = torch.clamp(self.Poff_d2d, min=0)

                self.Pon_d2d = torch.stack([self.Pon_d2d] * batch_size)
                self.Poff_d2d = torch.stack([self.Poff_d2d] * batch_size)

            if self.stuck_at_fault:
                SAF_lambda = self.memristor_info_dict[self.device_name]['SAF_lambda']
                SAF_ratio = self.memristor_info_dict[self.device_name]['SAF_ratio'] # SAF0:SAF1
                SAF_delta = self.memristor_info_dict[self.device_name]['SAF_delta'] # measured %/h????

                # Add pre-deployment SAF #TODO:Change to architectural SAF with Poisson-distributed intra-crossbar and uniform-distributed inter-crossbar SAF
                random_tensor = torch.rand(*self.shape, device=self.SAF0_mask.device)
                self.SAF0_mask = random_tensor < ((SAF_ratio / (SAF_ratio + 1)) * SAF_lambda)
                self.SAF1_mask = (random_tensor >= ((SAF_ratio / (SAF_ratio + 1)) * SAF_lambda)) & (random_tensor < SAF_lambda)

            if self.retention_loss == 2:
                self.mem_v_threshold = torch.zeros(batch_size, *self.shape, device=self.mem_v_threshold.device)
                self.mem_loss_time = torch.zeros(batch_size, *self.shape, device=self.mem_loss_time.device)

        if self.sum_input:
            self.summed = torch.zeros(
                batch_size, *self.shape, device=self.summed.device
            )

    def train(self, mode: bool = True) -> "Nodes":
        # language=rst
        """
        Sets the layer in training mode.

        :param bool mode: Turn training on or off
        :return: self as specified in `torch.nn.Module`
        """
        self.learning = mode
        return super().train(mode)


class AbstractInput(ABC):
    # language=rst
    """
    Abstract base class for groups of input neurons.
    """


class Input(Nodes, AbstractInput):
    # language=rst
    """
    Layer of nodes with user-specified spiking behavior.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1,
        sum_input: bool = False,
        mem_device: dict = {},
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of input neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record decaying spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param mem_device: Memristor device to be used in learning.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
            mem_device=mem_device
        )

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        On each simulation step, set the spikes of the population equal to the inputs.

        :param x: Inputs to the layer.
        """
        # Set spike occurrences to input values.
        self.s = x

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()


class McCullochPitts(Nodes):
    # language=rst
    """
    Layer of `McCulloch-Pitts neurons
    <http://wwwold.ece.utep.edu/research/webfuzzy/docs/kk-thesis/kk-thesis-html/node12.html>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = 1.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a McCulloch-Pitts layer of neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        self.v = x  # Voltages are equal to the inputs.
        self.s = self.v >= self.thresh  # Check for spiking neurons.

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = torch.zeros(batch_size, *self.shape, device=self.v.device)


class IFNodes(Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer(
            "reset", torch.tensor(reset, dtype=torch.float)
        )  # Post-spike reset voltage.
        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Integrate input voltages.
        self.v += (self.refrac_count <= 0).float() * x

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.reset)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.reset * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class LIFNodes(Nodes):
    # language=rst
    """
    Layer of `leaky integrate-and-fire (LIF) neurons
    <http://web.archive.org/web/20190318204706/http://icwww.epfl.ch/~gerstner/SPNM/node26.html#SECTION02311000000000000000>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of LIF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer(
            "rest", torch.tensor(rest, dtype=torch.float)
        )  # Rest voltage.
        self.register_buffer(
            "reset", torch.tensor(reset, dtype=torch.float)
        )  # Post-spike reset voltage.
        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay, dtype=torch.float)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.zeros(*self.shape)
        )  # Set in compute_decays.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        if lbound is None:
            self.lbound = None  # Lower bound of voltage.
        else:
            self.lbound = torch.tensor(
                lbound, dtype=torch.float
            )  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages.
        self.v = self.decay * (self.v - self.rest) + self.rest

        # Integrate inputs.
        x.masked_fill_(self.refrac_count > 0, 0.0)

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        self.v += x  # interlaced

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class BoostedLIFNodes(Nodes):
    # Same as LIFNodes, faster: no rest, no reset, no lbound
    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = 13.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of LIF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay, dtype=torch.float)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.zeros(*self.shape)
        )  # Set in compute_decays.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "refrac_count", torch.tensor(0)
        )  # Refractory period counters.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages.
        self.v *= self.decay

        # Integrate inputs.
        if x is not None:
            x.masked_fill_(self.refrac_count > 0, 0.0)

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        if x is not None:
            self.v += x

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, 0)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(0)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = torch.zeros(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class CurrentLIFNodes(Nodes):
    # language=rst
    """
    Layer of `current-based leaky integrate-and-fire (LIF) neurons
    <http://web.archive.org/web/20190318204706/http://icwww.epfl.ch/~gerstner/SPNM/node26.html#SECTION02313000000000000000>`_.
    Total synaptic input current is modeled as a decaying memory of input spikes multiplied by synaptic strengths.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        tc_i_decay: Union[float, torch.Tensor] = 2.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of synaptic input current-based LIF neurons.
        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param tc_i_decay: Time constant of synaptic input current decay.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("reset", torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.empty_like(self.tc_decay)
        )  # Set in compute_decays.
        self.register_buffer(
            "tc_i_decay", torch.tensor(tc_i_decay)
        )  # Time constant of synaptic input current decay.
        self.register_buffer(
            "i_decay", torch.empty_like(self.tc_i_decay)
        )  # Set in compute_decays.

        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("i", torch.FloatTensor())  # Synaptic input currents.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and current.
        self.v = self.decay * (self.v - self.rest) + self.rest
        self.i *= self.i_decay

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Integrate inputs.
        self.i += x
        self.v += (self.refrac_count <= 0).float() * self.i

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.i.zero_()  # Synaptic input currents.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.i_decay = torch.exp(
            -self.dt / self.tc_i_decay
        )  # Synaptic input current decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.i = torch.zeros_like(self.v, device=self.i.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class AdaptiveLIFNodes(Nodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds. A neuron's voltage threshold is increased
    by some constant each time it spikes; otherwise, it is decaying back to its default value.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        thresh: Union[float, torch.Tensor] = -52.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of LIF neurons with adaptive firing thresholds.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param thresh: Spike threshold voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("reset", torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.empty_like(self.tc_decay, dtype=torch.float32)
        )  # Set in compute_decays.
        self.register_buffer(
            "theta_plus", torch.tensor(theta_plus)
        )  # Constant threshold increase on spike.
        self.register_buffer(
            "tc_theta_decay", torch.tensor(tc_theta_decay)
        )  # Time constant of adaptive threshold decay.
        self.register_buffer(
            "theta_decay", torch.empty_like(self.tc_theta_decay)
        )  # Set in compute_decays.

        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("theta", torch.zeros(*self.shape))  # Adaptive thresholds.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.
        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.refrac_count <= 0).float() * x

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta += self.theta_plus * self.s.float().sum(0)

        # voltage clipping to lowerbound
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.theta_decay = torch.exp(
            -self.dt / self.tc_theta_decay
        )  # Adaptive threshold decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class DiehlAndCookNodes(Nodes):
    # language=rst
    """
    Layer of leaky integrate-and-fire (LIF) neurons with adaptive thresholds (modified for Diehl & Cook 2015
    replication).
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1,
        sum_input: bool = False,
        mem_device: dict = {},
        thresh: Union[float, torch.Tensor] = -52.0,
        rest: Union[float, torch.Tensor] = -65.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        lbound: float = None,
        one_spike: bool = True,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Diehl & Cook 2015 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param mem_device: Memristor device to be used in learning.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        :param one_spike: Whether to allow only one spike per timestep.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
            mem_device=mem_device
        )

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("reset", torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.empty_like(self.tc_decay)
        )  # Set in compute_decays.
        self.register_buffer(
            "theta_plus", torch.tensor(theta_plus)
        )  # Constant threshold increase on spike.
        self.register_buffer(
            "tc_theta_decay", torch.tensor(tc_theta_decay)
        )  # Time constant of adaptive threshold decay.
        self.register_buffer(
            "theta_decay", torch.empty_like(self.tc_theta_decay)
        )  # Set in compute_decays.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer("theta", torch.zeros(*self.shape))  # Adaptive thresholds.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.
        self.one_spike = one_spike  # One spike per timestep.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages and adaptive thresholds.
        self.v = self.decay * (self.v - self.rest) + self.rest
        if self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        self.v += (self.refrac_count <= 0).float() * x

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        # Refractoriness, voltage reset, and adaptive thresholds.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)
        if self.learning:
            self.theta += self.theta_plus * self.s.float().sum(0)

        # Choose only a single neuron to spike.
        if self.one_spike:
            if self.s.any():
                _any = self.s.view(self.batch_size, -1).any(1)
                ind = torch.multinomial(
                    self.s.float().view(self.batch_size, -1)[_any], 1
                )
                _any = _any.nonzero()
                self.s.zero_()
                self.s.view(self.batch_size, -1)[_any, ind] = 1

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.theta_decay = torch.exp(
            -self.dt / self.tc_theta_decay
        )  # Adaptive threshold decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)


class IzhikevichNodes(Nodes):
    # language=rst
    """
    Layer of `Izhikevich neurons<https://www.izhikevich.org/publications/spikes.htm>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        excitatory: float = 1,
        thresh: Union[float, torch.Tensor] = 45.0,
        rest: Union[float, torch.Tensor] = -65.0,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Izhikevich neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param excitatory: Percent of excitatory (vs. inhibitory) neurons in the layer; in range ``[0, 1]``.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.lbound = lbound

        self.register_buffer("r", None)
        self.register_buffer("a", None)
        self.register_buffer("b", None)
        self.register_buffer("c", None)
        self.register_buffer("d", None)
        self.register_buffer("S", None)
        self.register_buffer("excitatory", None)

        if excitatory > 1:
            excitatory = 1
        elif excitatory < 0:
            excitatory = 0

        if excitatory == 1:
            self.r = torch.rand(n)
            self.a = 0.02 * torch.ones(n)
            self.b = 0.2 * torch.ones(n)
            self.c = -65.0 + 15 * (self.r ** 2)
            self.d = 8 - 6 * (self.r ** 2)
            self.S = 0.5 * torch.rand(n, n)
            self.excitatory = torch.ones(n).byte()

        elif excitatory == 0:
            self.r = torch.rand(n)
            self.a = 0.02 + 0.08 * self.r
            self.b = 0.25 - 0.05 * self.r
            self.c = -65.0 * torch.ones(n)
            self.d = 2 * torch.ones(n)
            self.S = -torch.rand(n, n)

            self.excitatory = torch.zeros(n).byte()

        else:
            self.excitatory = torch.zeros(n).byte()

            ex = int(n * excitatory)
            inh = n - ex

            # init
            self.r = torch.zeros(n)
            self.a = torch.zeros(n)
            self.b = torch.zeros(n)
            self.c = torch.zeros(n)
            self.d = torch.zeros(n)
            self.S = torch.zeros(n, n)

            # excitatory
            self.r[:ex] = torch.rand(ex)
            self.a[:ex] = 0.02 * torch.ones(ex)
            self.b[:ex] = 0.2 * torch.ones(ex)
            self.c[:ex] = -65.0 + 15 * self.r[:ex] ** 2
            self.d[:ex] = 8 - 6 * self.r[:ex] ** 2
            self.S[:, :ex] = 0.5 * torch.rand(n, ex)
            self.excitatory[:ex] = 1

            # inhibitory
            self.r[ex:] = torch.rand(inh)
            self.a[ex:] = 0.02 + 0.08 * self.r[ex:]
            self.b[ex:] = 0.25 - 0.05 * self.r[ex:]
            self.c[ex:] = -65.0 * torch.ones(inh)
            self.d[ex:] = 2 * torch.ones(inh)
            self.S[:, ex:] = -torch.rand(n, inh)
            self.excitatory[ex:] = 0

        self.register_buffer("v", self.rest * torch.ones(n))  # Neuron voltages.
        self.register_buffer("u", self.b * self.v)  # Neuron recovery.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Voltage and recovery reset.
        self.v = torch.where(self.s, self.c, self.v)
        self.u = torch.where(self.s, self.u + self.d, self.u)

        # Add inter-columnar input.
        if self.s.any():
            x += torch.cat(
                [self.S[:, self.s[i]].sum(dim=1)[None] for i in range(self.s.shape[0])],
                dim=0,
            )

        # Apply v and u updates.
        self.v += self.dt * 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + x)
        self.v += self.dt * 0.5 * (0.04 * self.v ** 2 + 5 * self.v + 140 - self.u + x)
        self.u += self.dt * self.a * (self.b * self.v - self.u)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.u = self.b * self.v  # Neuron recovery.

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.u = self.b * self.v


class CSRMNodes(Nodes):
    """
    A layer of Cumulative Spike Response Model (Gerstner and van Hemmen 1992, Gerstner et al. 1996) nodes.
    It accounts for a model where refractoriness and adaptation were modeled by the combined effects
    of the spike after potentials of several previous spikes, rather than only the most recent spike.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        rest: Union[float, torch.Tensor] = -65.0,
        thresh: Union[float, torch.Tensor] = -52.0,
        responseKernel: str = "ExponentialKernel",
        refractoryKernel: str = "EtaKernel",
        tau: Union[float, torch.Tensor] = 1,
        res_window_size: Union[float, torch.Tensor] = 20,
        ref_window_size: Union[float, torch.Tensor] = 10,
        reset_const: Union[float, torch.Tensor] = 50,
        tc_decay: Union[float, torch.Tensor] = 100.0,
        theta_plus: Union[float, torch.Tensor] = 0.05,
        tc_theta_decay: Union[float, torch.Tensor] = 1e7,
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of Cumulative Spike Response Model nodes.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param rest: Resting membrane voltage.
        :param thresh: Spike threshold voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param theta_plus: Voltage increase of threshold after spiking.
        :param tc_theta_decay: Time constant of adaptive threshold decay.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "tau", torch.tensor(tau)
        )  # Time constant of Spike Response Kernel
        self.register_buffer(
            "reset_const", torch.tensor(reset_const)
        )  # Reset constant of refractory kernel
        self.register_buffer(
            "res_window_size", torch.tensor(res_window_size)
        )  # Window size for sampling incoming input current
        self.register_buffer(
            "ref_window_size", torch.tensor(ref_window_size)
        )  # Window size for sampling previous spikes
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer(
            "decay", torch.empty_like(self.tc_decay, dtype=torch.float32)
        )  # Set in compute_decays.
        self.register_buffer(
            "theta_plus", torch.tensor(theta_plus)
        )  # Constant threshold increase on spike.
        self.register_buffer(
            "tc_theta_decay", torch.tensor(tc_theta_decay)
        )  # Time constant of adaptive threshold decay.
        self.register_buffer(
            "theta_decay", torch.empty_like(self.tc_theta_decay)
        )  # Set in compute_decays.

        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "last_spikes", torch.ByteTensor()
        )  # Previous spikes occurrences in time window
        self.register_buffer("theta", torch.zeros(*self.shape))  # Adaptive thresholds.
        self.lbound = lbound  # Lower bound of voltage.

        self.responseKernel = responseKernel  # Type of spike response kernel used

        self.refractoryKernel = refractoryKernel  # Type of refractory kernel used

        self.register_buffer(
            "resKernel", torch.FloatTensor()
        )  # Vector of synaptic response kernel values over a window of time

        self.register_buffer(
            "refKernel", torch.FloatTensor()
        )  # Vector of refractory kernel values over a window of time

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages.
        self.v *= self.decay

        if self.learning:
            self.theta *= self.theta_decay

        # Integrate inputs.
        v = torch.einsum(
            "i,kij->kj", self.resKernel, x
        )  # Response due to incoming current
        v += torch.einsum(
            "i,kij->kj", self.refKernel, self.last_spikes
        )  # Refractoriness due to previous spikes
        self.v += v.view(x.size(0), *self.shape)

        # Check for spiking neurons.
        self.s = self.v >= self.thresh + self.theta

        if self.learning:
            self.theta += self.theta_plus * self.s.float().sum(0)

        # Add the spike vector into the first in first out matrix of windowed (ref) spike trains
        self.last_spikes = torch.cat(
            (self.last_spikes[:, 1:, :], self.s[:, None, :]), 1
        )

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).
        self.theta_decay = torch.exp(
            -self.dt / self.tc_theta_decay
        )  # Adaptive threshold decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.last_spikes = torch.zeros(batch_size, self.ref_window_size, *self.shape)

        resKernels = {
            "AlphaKernel": self.AlphaKernel,
            "AlphaKernelSLAYER": self.AlphaKernelSLAYER,
            "LaplacianKernel": self.LaplacianKernel,
            "ExponentialKernel": self.ExponentialKernel,
            "RectangularKernel": self.RectangularKernel,
            "TriangularKernel": self.TriangularKernel,
        }

        if self.responseKernel not in resKernels.keys():
            raise Exception(" The given response Kernel is not implemented")

        self.resKernel = resKernels[self.responseKernel](self.dt)

        refKernels = {"EtaKernel": self.EtaKernel}

        if self.refractoryKernel not in refKernels.keys():
            raise Exception(" The given refractory Kernel is not implemented")

        self.refKernel = refKernels[self.refractoryKernel](self.dt)

    def AlphaKernel(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = (1 / (self.tau ** 2)) * t * torch.exp(-t / self.tau)
        return torch.flip(kernelVec, [0])

    def AlphaKernelSLAYER(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = (1 / self.tau) * t * torch.exp(1 - t / self.tau)
        return torch.flip(kernelVec, [0])

    def LaplacianKernel(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = (1 / (self.tau * 2)) * torch.exp(-1 * torch.abs(t / self.tau))
        return torch.flip(kernelVec, [0])

    def ExponentialKernel(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = (1 / self.tau) * torch.exp(-t / self.tau)
        return torch.flip(kernelVec, [0])

    def RectangularKernel(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = 1 / (self.tau * 2)
        return torch.flip(kernelVec, [0])

    def TriangularKernel(self, dt):
        t = torch.arange(0, self.res_window_size, dt)
        kernelVec = (1 / self.tau) * (1 - (t / self.tau))
        return torch.flip(kernelVec, [0])

    def EtaKernel(self, dt):
        t = torch.arange(0, self.ref_window_size, dt)
        kernelVec = -self.reset_const * torch.exp(-t / self.tau)
        return torch.flip(kernelVec, [0])


class SRM0Nodes(Nodes):
    # language=rst
    """
    Layer of simplified spike response model (SRM0) neurons with stochastic threshold (escape noise). Adapted from
    `(Vasilaki et al., 2009) <https://intranet.physio.unibe.ch/Publikationen/Dokumente/Vasilaki2009PloSComputBio_1.pdf>`_.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -50.0,
        rest: Union[float, torch.Tensor] = -70.0,
        reset: Union[float, torch.Tensor] = -70.0,
        refrac: Union[int, torch.Tensor] = 5,
        tc_decay: Union[float, torch.Tensor] = 10.0,
        lbound: float = None,
        eps_0: Union[float, torch.Tensor] = 1.0,
        rho_0: Union[float, torch.Tensor] = 1.0,
        d_thresh: Union[float, torch.Tensor] = 5.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of SRM0 neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param rest: Resting membrane voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param tc_decay: Time constant of neuron voltage decay.
        :param lbound: Lower bound of the voltage.
        :param eps_0: Scaling factor for pre-synaptic spike contributions.
        :param rho_0: Stochastic intensity at threshold.
        :param d_thresh: Width of the threshold region.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer("rest", torch.tensor(rest))  # Rest voltage.
        self.register_buffer("reset", torch.tensor(reset))  # Post-spike reset voltage.
        self.register_buffer("thresh", torch.tensor(thresh))  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer(
            "tc_decay", torch.tensor(tc_decay)
        )  # Time constant of neuron voltage decay.
        self.register_buffer("decay", torch.tensor(tc_decay))  # Set in compute_decays.
        self.register_buffer(
            "eps_0", torch.tensor(eps_0)
        )  # Scaling factor for pre-synaptic spike contributions.
        self.register_buffer(
            "rho_0", torch.tensor(rho_0)
        )  # Stochastic intensity at threshold.
        self.register_buffer(
            "d_thresh", torch.tensor(d_thresh)
        )  # Width of the threshold region.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # Decay voltages.
        self.v = self.decay * (self.v - self.rest) + self.rest

        # Integrate inputs.
        self.v += (self.refrac_count <= 0).float() * self.eps_0 * x

        # Compute (instantaneous) probabilities of spiking, clamp between 0 and 1 using exponentials.
        # Also known as 'escape noise', this simulates nearby neurons.
        self.rho = self.rho_0 * torch.exp((self.v - self.thresh) / self.d_thresh)
        self.s_prob = 1.0 - torch.exp(-self.rho * self.dt)

        # Decrement refractory counters.
        self.refrac_count -= self.dt

        # Check for spiking neurons (spike when probability > some random number).
        self.s = torch.rand_like(self.s_prob) < self.s_prob

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v.masked_fill_(self.s, self.reset)

        # Voltage clipping to lower bound.
        if self.lbound is not None:
            self.v.masked_fill_(self.v < self.lbound, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.rest)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def compute_decays(self, dt) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super().compute_decays(dt=dt)
        self.decay = torch.exp(
            -self.dt / self.tc_decay
        )  # Neuron voltage decay (per timestep).

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        self.v = self.rest * torch.ones(batch_size, *self.shape, device=self.v.device)
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
