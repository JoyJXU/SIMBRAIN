from typing import Union
import torch
from torch.nn.parameter import Parameter
from spikingjelly.clock_driven.base import MemoryModule
from bpsr import surrogate as surrogate
import numpy as np
from scipy.optimize import curve_fit
import math

# from torch.profiler import record_function
# import pydevd
# pydevd.settrace(suspend=False, trace_only_current_thread=True)


class LIFNode(MemoryModule):
    def __init__(self, tau: Union[float, torch.Tensor] = 0.25,
                 v_threshold: float = 1., param_tau: bool = False,
                 surrogate_function: Union[surrogate.SurrogateFunctionBase,
                                           surrogate.MultiArgsSurrogateFunctionBase] = surrogate.Sigmoid()):
        """
        单步LIF神经元节点。
        math::
            V[t] = V[t-1] * \\tau * (1 - S[t-1]) + X[t]
            S[t] = (V[t] >= V_{th})

        :param tau: 神经元泄漏常数, 取值范围[0,1]
        :type tau: float, torch.Tensor
        例如::
            FRLIFNode(tau=0.5)
            FRLIFNode(tau=torch.rand(size=(100,)))          # 手动指定异构tau取值

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param param_tau: 是否将tau视作可学习的参数
        :type param_tau: bool

        :param surrogate_function: 梯度近似函数
        :type surrogate_function: bpsr.surrogate类
        """

        assert ((isinstance(tau, float) or isinstance(tau, int)) and 1. >= tau >= 0.) or \
               (isinstance(tau, torch.Tensor) and torch.logical_and(tau <= 1., tau >= 0.).all())
        assert isinstance(v_threshold, float)
        assert isinstance(param_tau, bool)
        assert isinstance(surrogate_function, surrogate.SurrogateFunctionBase) or \
               isinstance(surrogate_function, surrogate.MultiArgsSurrogateFunctionBase)
        super().__init__()

        self.register_memory('v', torch.as_tensor(0.))
        self.register_memory('spike', torch.as_tensor(0.))
        self.register_buffer('v_threshold', torch.as_tensor(v_threshold))
        self.surrogate_function = surrogate_function
        if isinstance(tau, float):
            tau = torch.as_tensor(tau)
        if isinstance(tau, Parameter):
            self.tau = tau
            param_tau = True
        elif param_tau:
            self.register_parameter('tau', Parameter(tau))
        else:
            self.register_buffer('tau', tau)
        self.param_tau = param_tau

    def extra_repr(self) -> str:
        return f'tau={self.tau.shape}, v_threshold={self.v_threshold}, param_tau={self.param_tau}'

    def reset(self):
        super().reset()
        if 'tau' in self._parameters:
            with torch.no_grad():
                self.tau.clamp_(0., 1.)

    def forward(self, x: torch.Tensor):
        """
        :param x: 单时间步突触电流, shape[N, ... Neuron Shape ...]
        :type x: torch.Tensor
        :return: torch.Tensor类型单时间步脉冲序列
        """
        quant = self.v_quant if hasattr(self, 'v_quant') else None
        self.v, self.spike = FunctionalLif.apply(self.v * self.tau * (1 - self.spike.detach()) + x,
                                                 self.v_threshold, self.surrogate_function, quant)
        return self.spike


class FunctionalLif(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        v, v_th, surrogate_function, quant = args
        if quant is None:
            spike = (v >= v_th).to(v)
        else:
            v_quant = quant(v)
            spike = (v_quant >= v_th).to(v_quant)
        ctx.save_for_backward(v)
        ctx.surrogate_function = surrogate_function
        ctx.v_th = v_th
        return v if quant is None else v_quant, spike

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not any(ctx.needs_input_grad):
            return None, None, None, None
        grad_v = grad_outputs[1] * ctx.surrogate_function.derivative(ctx.saved_tensors[0] - ctx.v_th)
        return grad_v, None, None, None


class MultiStepLIFNode(MemoryModule):
    def __init__(self, tau: Union[float, torch.Tensor] = 0.25,
                 v_threshold: float = 1., param_tau: bool = False,
                 surrogate_function: Union[surrogate.SurrogateFunctionBase,
                                           surrogate.MultiArgsSurrogateFunctionBase] = surrogate.Sigmoid()):
        """
        多步LIF神经元节点。
        math::
            V[t] = V[t-1] * \\tau * (1 - S[t-1]) + X[t]
            S[t] = (V[t] >= V_{th})

        :param tau: 神经元泄漏常数, 取值范围[0,1]
        :type tau: float, torch.Tensor
        例如::
            FRLIFNode(tau=0.5)
            FRLIFNode(tau=torch.rand(size=(100,)))          # 手动指定异构tau取值

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param param_tau: 是否将tau视作可学习的参数
        :type param_tau: bool

        :param surrogate_function: 梯度近似函数
        :type surrogate_function: bpsr.surrogate类
        """

        assert ((isinstance(tau, float) or isinstance(tau, int)) and 1. >= tau >= 0.) or \
               (isinstance(tau, torch.Tensor) and torch.logical_and(tau <= 1., tau >= 0.).all())
        assert isinstance(v_threshold, float)
        assert isinstance(param_tau, bool)
        assert isinstance(surrogate_function, surrogate.SurrogateFunctionBase) or \
               isinstance(surrogate_function, surrogate.MultiArgsSurrogateFunctionBase)
        super().__init__()

        self.register_buffer('v_threshold', torch.as_tensor(v_threshold))
        self.surrogate_function = surrogate_function
        if isinstance(tau, float):
            tau = torch.as_tensor(tau)
        if isinstance(tau, Parameter):
            self.tau = tau
            param_tau = True
        elif param_tau:
            self.register_parameter('tau', Parameter(tau))
        else:
            self.register_buffer('tau', tau)
        self.param_tau = param_tau

    def extra_repr(self) -> str:
        return f'tau={self.tau.shape}, v_threshold={self.v_threshold}, param_tau={self.param_tau}'

    def reset(self):
        if 'tau' in self._parameters:
            with torch.no_grad():
                self.tau.clamp_(0., 1.)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        :param x_seq: 多时间步突触电流, shape[T, N, ... Neuron Shape ...]
        :type x_seq: torch.Tensor
        :return: torch.Tensor类型多时间步脉冲序列
        """
        assert x_seq.dim() > 2
        quant = self.v_quant if hasattr(self, 'v_quant') else None
        self.v_seq, spike_seq = MultiStepFunctionalLif.apply(x_seq, self.tau, self.v_threshold, self.surrogate_function, quant)
        return spike_seq


class MultiStepFunctionalLif(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        x_seq, tau, v_th, surrogate_function, quant = args
        v_seq = torch.zeros_like(x_seq)
        spike_seq = torch.zeros_like(x_seq)
        if quant is None:
            for t in range(x_seq.shape[0]):
                v_seq[t] = x_seq[t, ...] if t == 0 else \
                    v_seq[t - 1] * tau * (1. - spike_seq[t - 1]) + x_seq[t, ...]
                spike_seq[t] = (v_seq[t] >= v_th).to(v_seq)
        else:
            v_seq_quant = torch.zeros_like(x_seq)
            for t in range(x_seq.shape[0]):
                v_seq[t] = x_seq[t, ...] if t == 0 else \
                    v_seq_quant[t - 1] * tau * (1. - spike_seq[t - 1]) + x_seq[t, ...]
                v_seq_quant[t] = quant(v_seq[t])
                spike_seq[t] = (v_seq_quant[t] >= v_th).to(v_seq)
        ctx.save_for_backward(v_seq, spike_seq, tau)
        ctx.surrogate_function = surrogate_function
        ctx.v_th = v_th
        ctx.T = x_seq.shape[0]
        ctx.tau_dim = tau.dim()
        return v_seq if quant is None else v_seq_quant, spike_seq

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not any(ctx.needs_input_grad):
            return None, None, None, None, None
        v_seq, spike_seq, tau = ctx.saved_tensors
        grad_x_seq, grad_tau = None, None
        grad_v = grad_outputs[1] * ctx.surrogate_function.derivative(v_seq - ctx.v_th)
        not_spike = (spike_seq == 0)
        for t in range(ctx.T - 2, -1, -1):
            grad_v[t, ...] += grad_v[t + 1, ...] * not_spike[t, ...] * tau
        if ctx.needs_input_grad[0]:
            grad_x_seq = grad_v
        if ctx.needs_input_grad[1]:
            grad_tau = (grad_v[1:, ...] * v_seq[:-1, ...] * not_spike[:-1, ...]).sum() if ctx.tau_dim == 0 \
                else (grad_v[1:, ...] * v_seq[:-1, ...] * not_spike[:-1, ...]).sum(dim=(0, 1))
        return grad_x_seq, grad_tau, None, None, None


class MultiStepRateLIFNode(MemoryModule):
    def __init__(self, tau: Union[float, torch.Tensor] = 0.25,
                 v_threshold: float = 1., param_tau: bool = False,
                 fitting_interval: int = 100, counting_batches: int = None):
        """
        基于速率编码反向传播的多步LIF神经元节点。
        math::
            V[t] = V[t-1] * \\tau * (1 - S[t-1]) + X[t]
            S[t] = (V[t] >= V_{th})

        :param tau: 神经元泄漏常数, 取值范围[0,1]
        :type tau: float, torch.Tensor
        例如::
            FRLIFNode(tau=0.5)
            FRLIFNode(tau=torch.rand(size=(100,)))          # 手动指定异构tau取值

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param param_tau: 是否将tau视作可学习的参数
        :type param_tau: bool

        :param fitting_interval: 进行Linear_Sigmoid拟合的间隔周期
        :type fitting_interval: int

        :param counting_batches: 进行FR统计的批数量
        :type counting_batches: int
        """

        assert ((isinstance(tau, float) or isinstance(tau, int)) and 1. >= tau >= 0.) or \
               (isinstance(tau, torch.Tensor) and torch.logical_and(tau <= 1., tau >= 0.).all())
        assert isinstance(v_threshold, float)
        assert isinstance(param_tau, bool)
        assert isinstance(fitting_interval, int)
        super().__init__()

        self.register_buffer('v_threshold', torch.as_tensor(v_threshold))
        if isinstance(tau, float):
            tau = torch.as_tensor(tau)
        if isinstance(tau, Parameter):
            self.tau = tau
            param_tau = True
        elif param_tau:
            self.register_parameter('tau', Parameter(tau))
        else:
            self.register_buffer('tau', tau)

        self.param_tau = param_tau
        self.fitting_interval = fitting_interval
        self.counting_batches = counting_batches
        self.call_count = 0
        self.register_buffer('fitting_param', torch.as_tensor([1., -self.v_threshold / 2]))
        self.fitting_data = None

    def extra_repr(self) -> str:
        return f'tau={self.tau.shape}, v_threshold={self.v_threshold}, param_tau={self.param_tau}'

    def reset(self):
        if 'tau' in self._parameters:
            with torch.no_grad():
                self.tau.clamp_(0., 1.)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        :param x_seq: 多时间步突触电流, shape[T, N, ... Neuron Shape ...]
        :type x_seq: torch.Tensor
        :return: torch.Tensor类型多时间步脉冲序列
        """
        assert x_seq.dim() > 2
        quant = self.v_quant if hasattr(self, 'v_quant') else None
        self.v_seq, spike_seq = \
            FunctionalRateLif.apply(x_seq, self.tau, self.v_threshold, self.fitting_param, quant)

        if self.training:
            with torch.no_grad():
                self.call_count += 1
                if self.counting_batches is None:
                    aver_sample = x_seq[0, ...].numel() / x_seq.shape[0]
                    self.counting_batches = min(1e4 // aver_sample, self.fitting_interval)

                if self.call_count > self.fitting_interval - self.counting_batches \
                        or self.fitting_data is None:
                    if self.fitting_data is None:
                        self.fitting_data = torch.zeros(x_seq.shape[0] + 1, device=x_seq.device, dtype=x_seq.dtype) * torch.nan
                        self.call_count = self.fitting_interval - self.counting_batches + 1
                        self.s_bar_unique = torch.linspace(0., 1., x_seq.shape[0] + 1).to(x_seq)
                    s_bar = spike_seq.mean(axis=0)
                    i_bar = x_seq.mean(axis=0)
                    i_bar_average = torch.zeros_like(self.s_bar_unique) * torch.nan
                    for i, fr_ in enumerate(self.s_bar_unique):
                        mask_ = s_bar == fr_
                        count_ = mask_.float().sum()
                        i_bar_average[i] = (i_bar * mask_).sum() / count_
                    i_bar_average = torch.where(torch.isnan(i_bar_average),
                                                self.fitting_data, i_bar_average)
                    self.fitting_data = torch.where(torch.isnan(self.fitting_data),
                                                    i_bar_average,
                                                    (self.fitting_data + i_bar_average) / 2)

                if self.call_count == self.fitting_interval:
                    x, y, p0 = self.fitting_data.cpu().numpy(), self.s_bar_unique.cpu().numpy(), self.fitting_param.cpu().numpy()
                    x, y = x[np.isfinite(x)], y[np.isfinite(x)]
                    try:
                        (gamma, beta), _ = curve_fit(linear_sigmoid, x, y, p0=p0)
                        self.fitting_param = torch.as_tensor([gamma, beta]).to(self.fitting_param)
                    except: pass
                    self.call_count = 0
        return spike_seq


def linear_sigmoid(x, gamma, beta):
    if isinstance(x, np.ndarray):
        kernel = np
    elif isinstance(x, torch.Tensor):
        kernel = torch
    else:
        kernel = math
    return 1. / (1. + kernel.exp(-gamma * x - beta))


@torch.jit.script
def jit_linear_sigmoid(x, gamma, beta):
    return torch.sigmoid(gamma * x + beta)


def partial_s_tau(s, alpha, sigma, k):
    if isinstance(s, np.ndarray):
        return alpha * np.power(s * (1. - s), k) * np.exp(-sigma * s)
    elif isinstance(s, torch.Tensor):
        return alpha * torch.pow(s * (1. - s), k) * torch.exp(-sigma * s)
    else:
        return alpha * math.pow(s * (1. - s), k) * math.exp(-sigma * s)


class FunctionalRateLif(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        x_seq, tau, v_th, fiiting_param, quant = args
        v_seq = torch.zeros_like(x_seq)
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v_seq[t] = x_seq[t, ...] if t == 0 else \
                v_seq[t - 1] * tau * (1. - spike_seq[t - 1]) + x_seq[t, ...]
            if quant is not None:
                v_seq[t] = quant(v_seq[t])
            spike_seq[t] = (v_seq[t] >= v_th).to(v_seq)
        ctx.save_for_backward(x_seq.mean(dim=0), spike_seq.mean(dim=0), fiiting_param)
        ctx.T = x_seq.shape[0]
        ctx.tau_dim = tau.dim()
        return v_seq, spike_seq

    @staticmethod
    def backward(ctx, *grad_outputs):
        if not any(ctx.needs_input_grad):
            return None, None, None, None, None
        i_bar, s_bar, fiiting_param = ctx.saved_tensors
        grad_x_seq, grad_tau = None, None
        grad_s_bar = grad_outputs[1].mean(dim=0)
        if ctx.needs_input_grad[0]:
            S = jit_linear_sigmoid(i_bar, fiiting_param[0], fiiting_param[1])
            grad_x_seq = fiiting_param[0] * grad_s_bar * S * (1. - S)
            grad_x_seq = grad_x_seq.unsqueeze(0).expand((ctx.T, ) + grad_x_seq.shape)
        if ctx.needs_input_grad[1]:
            partial_tau_ = partial_s_tau(s_bar, 3.4, 3.4, 1.16)
            grad_tau = (grad_s_bar * partial_tau_).sum() if ctx.tau_dim == 0 else \
                (grad_s_bar * partial_tau_).sum(axis=0)
        return grad_x_seq, grad_tau, None, None, None


if __name__ == '__main__':
    import itertools
    from spikingjelly.clock_driven import functional

    T = 10
    batch_size = 4
    neu_size = (32, 32, 3)

    for neu_, tau_, param_tau_, device_ in itertools.product(
            [LIFNode, MultiStepLIFNode, MultiStepRateLIFNode],
            [0.5, torch.rand(neu_size)],
            [False, True], ['cpu', 'cuda']):
        Layer = neu_(tau=tau_, param_tau=param_tau_).to(device_)
        for batch in range(10):
            if isinstance(Layer, LIFNode) and not isinstance(Layer, MultiStepLIFNode) :
                for t_ in range(T):
                    x_in = torch.randn((batch_size,) + neu_size, requires_grad=True).to(device_)
                    s_out = Layer(x_in) if t_ == 0 else s_out + Layer(x_in)
                s_out.sum().backward()
            else:
                x_in = torch.randn((T, batch_size) + neu_size, requires_grad=True).to(device_)
                s_out = Layer(x_in)
                s_out.sum().backward()
            functional.reset_net(Layer)

    print('Test Pass')
