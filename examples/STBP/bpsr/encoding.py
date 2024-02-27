from typing import Literal, Union
import numpy as np
import torch


class LCSampler:
    # floating window LC_ADC sampling
    def __init__(self, delta: float, window: Literal['floating', 'fixed'] = 'floating'):
        """
        时间序列的LC采样器。

        :param delta: LC采样的阈值
        :type delta: float

        :param window: 窗口类型，可以为 ”floating“, "fixed"
        :type window: str
        """
        assert isinstance(delta, float)
        assert isinstance(window, str) and window in ['floating', 'fixed']
        self.delta = delta
        self.window = window

    def encode(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        :param x: 时间序列信号, shape[T, ...]
        :type x: np.array或torch.Tensor
        :return: LC采样后的脉冲序列信号，np.array或torch.Tensor类型
        """
        kernel = np if isinstance(x, np.ndarray) else torch
        lc = kernel.zeros_like(x)
        origin = kernel.zeros_like(x[0, ...])
        for t in range(x.shape[0]):
            lc[t, ...] = kernel.trunc((x[t, ...] - origin) / self.delta)
            origin = kernel.where(lc[t, ...] == 0, origin, x[t, ...]) if self.window == 'floating' \
                else origin + lc[t, ...] * self.delta
        lc[lc == 0] = 0
        return lc

    def decode(self, lc: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        :param lc: 脉冲序列信号, shape[T, ...]
        :type lc: np.array或torch.Tensor
        :return: 从LC信号恢复的时间序列信号，np.array或torch.Tensor类型
        """
        kernel = np if isinstance(lc, np.ndarray) else torch
        return kernel.cumsum(lc * self.delta, axis=0)
