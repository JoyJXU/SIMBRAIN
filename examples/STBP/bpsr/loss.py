from typing import Union, Optional, Sequence
import torch
from torch.nn import Module


class LossScaleEstimator(Module):
    def __init__(self, start: bool = False,
                 lambd: Union[float, Sequence[float]] = 1.,
                 margin: Union[float, Sequence[float]] = 0.):
        """
        稀疏正则化的损失函数
        L=L_{task}+\lambda_s\sigma_s \cdot L_{sparse}
        \sigma_s:=E(L_{task})/E(L_{sparse})

        :param start: 尺度因子是否需要预热过程
        :type start: bool

        :param lambd: 稀疏正则化稀疏
        :type lambd: float

        :param margin: 目标稀疏函数
        :type margin: float
        """
        super().__init__()
        self.start = start

        self.lambd = lambd if isinstance(lambd, Sequence) else (lambd, )
        self.margin = margin if isinstance(margin, Sequence) else (margin, )
        self.register_buffer('norm_coff', torch.as_tensor([1. for _ in range(len(self.lambd))]))
        self.base_loss = []
        self.scale_loss = None

    def forward(self, base_loss: torch.Tensor,
                scale_loss: Union[torch.Tensor, Sequence[torch.Tensor]]
                ) -> torch.Tensor:
        """
        :param base_loss: 基准损失
        :type base_loss: torch.Tensor
        :param scale_loss: 待缩放损失
        :type scale_loss: torch.Tensor
        :return: torch.Tensor类型损失
        """
        if not isinstance(scale_loss, Sequence):
            scale_loss = (scale_loss, )
        n_loss = len(scale_loss)
        assert len(self.lambd) == n_loss == n_loss

        self.base_loss.append(float(base_loss))
        if self.scale_loss is None:
            if len(self.lambd) == 1 and n_loss > 1:
                self.lambd = [self.lambd[0] for _ in range(n_loss)]
            if len(self.margin) == 1 and n_loss > 1:
                self.margin = [self.margin[0] for _ in range(n_loss)]
            self.scale_loss = [list() for _ in range(n_loss)]
        for i, scale_loss_ in enumerate(scale_loss):
            self.scale_loss[i].append(float(scale_loss_))

        if self.start:
            return sum([norm_coff_ * lambda_ * torch.max(torch.as_tensor(0.), scale_loss_ - margin_)
                        for scale_loss_, lambda_, margin_, norm_coff_ in
                        zip(scale_loss, self.lambd, self.margin, self.norm_coff)]) + base_loss
        else:
            return base_loss

    def step(self, start: Optional[bool] = None):
        """
        进行尺度估计
        :param start: 是否预热完毕
        :type start: bool
        """
        self.norm_coff = torch.as_tensor(
            [sum(self.base_loss) / sum(scale_loss_) for scale_loss_ in self.scale_loss])
        self.base_loss.clear()
        self.scale_loss = [list() for _ in range(len(self.scale_loss))]
        if start is not None:
            self.start = start
