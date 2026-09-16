"""FreDF 的 √H 尺度归一化变体。

动机
----
FreDF 的损失是 ``(1−α)·mean(resid²) + α·mean|rFFT(resid)|``。两项对 horizon H 的
标度不同：Parseval 关系下 FFT 系数的模长随 √H 增长，而时域 MSE 与 H 无关。实测
（`internal/docs/fredf_tuning_evidence.md` §3，残差同分布、只改 H）：

    H         96      192     336     720
    freq/mse  86.5    122.5   162.3   237.7      → 比值 ∝ √H（237.7/86.5 = 2.75 ≈ √7.5）

后果是**固定 α 在跨 horizon 时并不表示固定的权衡**：α=0.5 在 H=720 上的实际频域
权重是 H=96 上的 2.75 倍。这解释了主矩阵里 fredf 增益随 horizon 单调恶化
（−1.57% / −3.01% / −3.96% / −3.60%），也解释了为什么 FreDF 官方脚本必须逐数据集
重调 α——它同时在补偿"数据集不同"和"H 不同"两件事。

本变体只做一件事：把频域项除以 √(H//2+1)（rfft 的频点数），使两项在 H 上同阶。
如果这一项修正能让**单个** α 跨 4 个 horizon 都可用，那它本身就是一个小而干净的
方法贡献：把一个"必须逐格调参的插件"变成"真正即插即用的插件"。

判定口径（写进论文时必须一致）
--------------------------------
- 主张成立：`fredf_sqrth` 在固定 α 下，跨 horizon 的增益标准差显著小于 `fredf`，
  且平均增益不低于 `fredf`。
- 主张不成立：如果两者的 horizon 依赖趋势相同，说明退化不是尺度问题而是
  优化/过拟合问题，必须据实报告并撤掉这条叙事。

注意：归一化常数只依赖 H，不依赖数据，因此**不引入任何可调超参、不构成调参优势**。
这一点必须在论文里明说，否则会被质疑"你给自己的变体多调了一个东西"。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class FreDFSqrtHLoss(nn.Module):
    """(1−α)·MSE + α·mean|rFFT(resid)| / √(H//2+1)。

    参数
    ----
    alpha: 频域项权重，语义与 `FreDFLoss` 完全一致，便于直接对照。
    dim:   做 FFT 的轴，1 = 时间轴（与 TSLib 的 [B, H, C] 布局一致）。
    """

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
        p32, t32 = pred.float(), true.float()  # V100: fp16 cuFFT 不支持非 2 的幂长度
        resid_f = torch.fft.rfft(p32, dim=self.dim) - torch.fft.rfft(t32, dim=self.dim)
        n_freq = resid_f.shape[self.dim]  # = H//2 + 1
        freq_l1 = resid_f.abs().mean() / (float(n_freq) ** 0.5)
        return (1.0 - self.alpha) * mse + self.alpha * freq_l1.to(mse.dtype)

    def extra_repr(self) -> str:  # pragma: no cover
        return f"alpha={self.alpha}, scale=1/sqrt(n_freq)"
