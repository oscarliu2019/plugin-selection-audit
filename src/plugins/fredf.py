"""FreDF：频域损失（ICLR 2025）的精简复现——一个**损失层**插件。

机理：MSE 假设各时间步的误差相互独立，但标签序列本身是强自相关的；
在频域上算损失可以绕开这一错配。FreDF 的原始实现只是 2 行代码：
``loss = (1-α)·MSE(pred, true) + α·L1(rFFT(pred) - rFFT(true))``。

本项目对应的特征直觉：`fr_topk_energy` / `fr_flatness` 越显示能量集中在少数频率，
频域损失越可能有利；接近白噪声的序列上它不应该有增益。

V100 注意事项：`torch.fft` 不支持 complex32，因此在 AMP(fp16) 下必须先 cast 回 float32
再做 FFT——本实现已强制 float32，Electricity/Traffic 开 AMP 时不会崩。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class FreDFLoss(nn.Module):
    """(1-α)·MSE + α·频域 L1；α 全矩阵固定，不逐数据集调。"""

    def __init__(self, alpha: float = 0.5, dim: int = 1) -> None:
        super().__init__()
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha 必须落在 [0,1]")
        self.alpha = float(alpha)
        self.dim = int(dim)

    def forward(self, pred: Tensor, true: Tensor) -> Tensor:
        mse = F.mse_loss(pred, true)
        if self.alpha == 0.0:
            return mse
        p32, t32 = pred.float(), true.float()  # V100: complex32 不可用，强制 fp32
        fp = torch.fft.rfft(p32, dim=self.dim)
        ft = torch.fft.rfft(t32, dim=self.dim)
        freq_l1 = (fp - ft).abs().mean()
        return (1.0 - self.alpha) * mse + self.alpha * freq_l1.to(mse.dtype)

    def extra_repr(self) -> str:  # pragma: no cover
        return f"alpha={self.alpha}"
