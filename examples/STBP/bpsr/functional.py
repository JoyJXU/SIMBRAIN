from typing import Union, Optional, Literal, Iterable
import torch
from torch import nn
from math import sqrt
from bpsr import surrogate as surrogate


def bpsr_init(net: nn.Module,
              encoding: Optional[Union[nn.Module, Iterable[nn.Module]]] = None,
              decoding: Optional[Union[nn.Module, Iterable[nn.Module]]] = None,
              fr: float = 0.15, v_threshold: float = 1.,
              tau: Optional[float] = None, T: Optional[int] = None,
              init: Literal['uniform', 'normal'] = 'uniform',
              surrogate_function: Union[surrogate.SurrogateFunctionBase,
                                        surrogate.MultiArgsSurrogateFunctionBase] = surrogate.Sigmoid()):
    """
    网络突触连接初始化方法，其中weight为均匀分布或高斯分布，有着方差
    D(weight)=2\cdot \left\lbrace n_{in}f\cdot\Big[\frac{\Phi^{-1}(1-f)}{u_{th}}\Big]^2 + n_{out}E(\symscr{h}^2) \right\rbrace ^{-1} \label{eq:Dw3}
    bias为常数0
    注意，初始化仅对Linear及Conv层生效

    :param net: 待初始化网络或网络层
    :type net: nn.Module

    :param encoding: 脉冲编码层
    :type encoding: nn.Module or None

    :param decoding: 脉冲解码层
    :type decoding: nn.Module or None

    :param fr: 初始化输入发放率，取值范围(0,0.5)
    :type fr: float

    :param v_threshold: 神经元发放阈值
    :type v_threshold: float

    :param tau: 衰减参数或其期望值，如为None则忽视tau在迭代过程中的影响
    :type tau: float or None

    :param T: 时间步长，如为None则忽视T在迭代过程中的影响
    :type T: int or None

    :param init: 初始化参数分布，'uniform'或'normal'
    :type init: str

    :param surrogate_function: 梯度近似函数
    :type surrogate_function: bpsr.surrogate类
    """

    assert isinstance(net, nn.Module)
    assert 0.5 > fr > 0.
    assert isinstance(v_threshold, float)
    assert tau is None or (isinstance(tau, float) and 1. >= tau >= 0)
    assert T is None or (isinstance(T, int) and T > 0)
    assert isinstance(init, str) and init in ['uniform', 'normal']
    assert isinstance(surrogate_function, surrogate.SurrogateFunctionBase) or \
           isinstance(surrogate_function, surrogate.MultiArgsSurrogateFunctionBase)

    if encoding is None:
        encoding = []
    if decoding is None:
        decoding = []
    if not isinstance(encoding, nn.Module):
        encoding = nn.Sequential(*encoding)
    if not isinstance(decoding, nn.Module):
        decoding = nn.Sequential(*decoding)

    if not isinstance(fr, float):
        fr = float(fr)
    with torch.no_grad():
        Du = (v_threshold / torch.special.ndtri(1 - torch.as_tensor(fr))) ** 2
        gaussian_dist = torch.randn(10000) * sqrt(Du) - v_threshold
        Eh2 = (surrogate_function.derivative(gaussian_dist) ** 2).mean()
        for m in net.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
                fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(m.weight)
                if m in encoding.modules():
                    Dw = 1 / fan_in * Du
                elif m in decoding.modules():
                    Dw = 2 / (fan_in + fan_out)
                else:
                    Dw1 = Du / fan_in / fr
                    Dw2 = 1 / fan_out / Eh2
                    Dw = 2 / (1 / Dw1 + 1 / Dw2)
                    if tau is not None and T is not None:
                        k = tau ** 2 * fr
                        Dw *= k * T / (pow(1 + k, T) - 1)

                std = sqrt(Dw)
                if init == 'uniform':
                    bound = sqrt(3.) * std
                    torch.nn.init.uniform_(m.weight, -bound, bound)
                else:
                    torch.nn.init.normal_(m.weight, 0., std)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0.)
