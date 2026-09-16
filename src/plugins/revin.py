"""RevIN：可逆实例归一化（ICLR 2022）的精简复现。

机理：用**当前输入窗口自身**的通道均值/方差把输入标准化，模型在标准化空间预测，
再把输出逆变换回原空间。它处理的是「窗口之间的分布漂移」，因此
`features.hz_mean_shift` / `features.ns_mean_drift` 越大，它越可能有用。

关键实现细节（审稿人会查）：
- 统计量只来自 x_enc（输入窗口），**不使用任何未来信息**，无泄漏；
- x_dec 的前 label_len 段是真实观测，必须用同一组统计量归一化，否则解码器输入尺度不一致；
- 默认 affine=False：不引入任何可学习参数，从而满足「零新增可调超参」。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import PluginWrapper, align_channels

Tensor = torch.Tensor


class RevINPlugin(PluginWrapper):
    plugin_name = "revin"
    kind = "io"

    def __init__(self, model: nn.Module, affine: bool = False, eps: float = 1e-5,
                 n_channels: int | None = None) -> None:
        super().__init__(model, affine=affine, eps=eps)
        self.eps = float(eps)
        self.affine = bool(affine)
        if self.affine:
            if n_channels is None:
                raise ValueError("affine=True 时必须给出 n_channels")
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))

    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor | None = None,
        x_dec: Tensor | None = None,
        x_mark_dec: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        mean = x_enc.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x_enc.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        x_n = (x_enc - mean) / std
        if self.affine:
            x_n = x_n * self.weight + self.bias

        dec_n = None
        if x_dec is not None:
            m_d, s_d = align_channels(mean, x_dec), align_channels(std, x_dec)
            dec_n = (x_dec - m_d) / s_d
            if self.affine:
                w = align_channels(self.weight.view(1, 1, -1), x_dec)
                b = align_channels(self.bias.view(1, 1, -1), x_dec)
                dec_n = dec_n * w + b

        out = self.model(x_n, x_mark_enc, dec_n, x_mark_dec, mask)

        if self.affine:
            w = align_channels(self.weight.view(1, 1, -1), out)
            b = align_channels(self.bias.view(1, 1, -1), out)
            out = (out - b) / (w + self.eps)
        return out * align_channels(std, out) + align_channels(mean, out)
