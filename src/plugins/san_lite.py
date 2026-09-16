"""SAN-lite：切片级自适应归一化（NeurIPS 2023 SAN）的**免训练**精简变体。

与 RevIN 的区别（也是它存在的理由）：
- RevIN 用整个输入窗口的**一组**统计量，隐含假设「窗口内平稳」；
- SAN 把窗口切成若干切片，用**每个切片自己的**统计量归一化，并需要预测**未来切片**的统计量。

原版 SAN 用一个额外网络两阶段训练来预测未来切片统计量。为严格满足本文的
「训练解耦 + 零新增可调超参」硬约束（尽调 §5.2 第 10 条），本文实现的是它的
**免训练代理**：对过去 K 个切片的统计量（均值、log 标准差）沿切片索引做
**指数加权平均**（EWMA，半衰期固定为 ``K/2``、斜率隐式为 0），把该常数作为全部
未来切片的统计量预测（``_ewma_extrapolate``，O(K) 闭式、无新增超参）。

方法描述口径（2026-09-14 code review 修复项，论文方法章节按这段写）：
默认路径就是 **EWMA**，不是岭回归。``__init__`` 的 ``stats_estimator`` 默认值是
``"ewma"``，``forward`` 默认走 ``_ewma_extrapolate``；三个跑全矩阵的 config
（``configs/matrix.yaml`` / ``matrix_p2.yaml`` / ``matrix_p2seed.yaml``）的 san_lite
``params`` 都没有设置 ``stats_estimator``，所以**已跑完的全矩阵结果全部来自 EWMA**。
「沿切片索引做岭回归线性外推」（``_ridge_extrapolate``）实测显著更差
（统计量误差折算 MSE 0.663 vs EWMA 0.313，41 个完整块上 0 胜 41 负），
已降级为 ``stats_estimator="ridge"`` 的**负面对照**（ablation），
论文里只能以「为什么免训练线性外推不可行」的证据出现，不能写成 SAN-lite 的方法本体。

因此在论文里必须诚实地称之为 `SAN-lite (training-free surrogate of SAN)`，
而不是 SAN 本身；它的作用是提供一个「比 RevIN 更细粒度、但同样零参数」的归一化插件。

切片长度规则（固定、不逐数据集调）：``P = max(4, min(slice_len, seq_len // 3))``。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import PluginWrapper, align_channels

Tensor = torch.Tensor


def _ridge_extrapolate(stats: Tensor, n_future: int, lam: float) -> Tensor:
    """对 (B, K, C) 的切片统计量沿切片索引做岭回归线性外推，返回 (B, n_future, C)。

    .. deprecated::
        **本函数已不再用于 SAN-lite 的前向**，仅作为论文中的负面对照（ablation）保留。

        实测（``tools/diag_san.py`` / ``tools/diag_san_estimators.py``）：在 seq_len=96、
        切片长 24 的默认设置下只有 **K=4 个观测切片**，用 4 个点估斜率再外推最多 30 个切片，
        纯粹在放大噪声。把它折算成「完美主干下仅由统计量误差造成的 MSE」，
        全局平均 0.663，而最笨的持续性基线是 0.338、EWMA 是 0.313——
        它比什么都不做还差一倍，导致 SAN-lite 在 41 个完整块上 0 胜 41 负、平均相对退化 -155%。

        结论：切片变换本身是对的（oracle 统计量下 MSE=0），坏的是「免训练线性外推」这个
        设计选择本身。这反过来解释了原版 SAN 为何**必须**带一个训练出来的统计量预测网络。
    """
    b, k, c = stats.shape
    device, dtype = stats.device, stats.dtype
    idx = torch.arange(k, device=device, dtype=dtype).view(1, k, 1)
    idx_c = idx - idx.mean()
    y_mean = stats.mean(dim=1, keepdim=True)
    num = (idx_c * (stats - y_mean)).sum(dim=1, keepdim=True)
    den = (idx_c**2).sum() + lam
    slope = num / den
    f_idx = torch.arange(k, k + n_future, device=device, dtype=dtype).view(1, n_future, 1)
    pred = y_mean + slope * (f_idx - idx.mean())
    # 数值护栏：把外推值夹在「观测切片统计量的邻域」内。
    # 免训练线性外推在长 horizon（如 720 步 = 30 个切片）上会发散——实测 iTransformer/ETTh1/336
    # 上不加护栏时训练损失会飙到 1e4 量级。夹紧区间由观测统计量自身的跨度定义，
    # 不引入任何需要调的超参。
    lo = stats.min(dim=1, keepdim=True).values
    hi = stats.max(dim=1, keepdim=True).values
    span = (hi - lo).clamp(min=1e-6)
    return pred.clamp(min=lo - span, max=hi + span)


def _ewma_extrapolate(stats: Tensor, n_future: int) -> Tensor:
    """沿切片索引做指数加权平均，把该常数作为全部未来切片的统计量预测。

    这是 ``tools/diag_san_estimators.py`` 在 5 数据集 × 4 horizon 上选出的最优免训练估计器
    （统计量误差折算 MSE 全局平均 0.313，优于整窗均值 0.331、持续性 0.338、
    岭回归 0.663）。

    半衰期固定为 ``K/2``（K = 观测切片数），因此**没有引入任何需要逐数据集调的超参**：
    越靠近预测起点的切片权重越大，但仍聚合全部历史切片以抑制方差。
    斜率被隐式设为 0——这正是修复的核心：切片均值序列近似随机游走，
    对它做趋势外推是负收益。
    """
    b, k, c = stats.shape
    device, dtype = stats.device, stats.dtype
    half_life = max(k / 2.0, 1e-6)
    # 权重 0.5**((K-1-i)/half_life)：i=k-1（最后一个切片）权重为 1
    age = torch.arange(k - 1, -1, -1, device=device, dtype=dtype).view(1, k, 1)
    w = torch.pow(torch.tensor(0.5, device=device, dtype=dtype), age / half_life)
    agg = (stats * w).sum(dim=1, keepdim=True) / w.sum()
    return agg.expand(b, n_future, c)


class SANLitePlugin(PluginWrapper):
    plugin_name = "san_lite"
    kind = "io"

    def __init__(self, model: nn.Module, slice_len: int = 24, ridge_lambda: float = 1.0,
                 eps: float = 1e-5, std_floor: float = 0.1,
                 stats_estimator: str = "ewma",
                 seq_len: int | None = None, pred_len: int | None = None) -> None:
        super().__init__(model, slice_len=slice_len, ridge_lambda=ridge_lambda,
                         std_floor=std_floor, stats_estimator=stats_estimator)
        self.cfg_slice_len = int(slice_len)
        self.lam = float(ridge_lambda)
        self.eps = float(eps)
        self.std_floor = float(std_floor)
        if stats_estimator not in ("ewma", "ridge"):
            raise ValueError(f"stats_estimator 必须是 ewma|ridge，收到 {stats_estimator!r}")
        self.stats_estimator = stats_estimator
        self.seq_len = seq_len
        self.pred_len = pred_len

    # ---- 内部工具 ----
    def _slice_len(self, seq_len: int) -> int:
        return max(4, min(self.cfg_slice_len, seq_len // 3))

    @staticmethod
    def _slice_stats(x: Tensor, p: int, eps: float, std_floor: float) -> tuple[Tensor, Tensor, int]:
        """把 (B,L,C) 按长度 p 切片（末尾对齐，丢掉最前面的余数），返回逐切片均值/log 标准差。

        标准差下限（数值护栏，实测必需）：ETTh1 等数据集里存在**完全恒定的切片**
        （实测 ETTh1 训练集中切片标准差最小值 = 0.0），若直接相除会放大 1e12 倍，
        训练损失会飙到 1e4 量级。因此把切片标准差夹在「窗口标准差的 std_floor 倍」以上，
        放大倍数被硬性限制在 1/std_floor 内。
        """
        b, l, c = x.shape
        k = l // p
        x_tail = x[:, l - k * p :, :].reshape(b, k, p, c)
        mean = x_tail.mean(dim=2)
        std = torch.sqrt(x_tail.var(dim=2, unbiased=False) + eps)
        win_std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + eps)
        std = torch.maximum(std, std_floor * win_std)
        return mean, torch.log(std), k

    # ---- 前向 ----
    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor | None = None,
        x_dec: Tensor | None = None,
        x_mark_dec: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        b, l, c = x_enc.shape
        p = self._slice_len(l)
        mean, logstd, k = self._slice_stats(x_enc.detach(), p, self.eps, self.std_floor)
        std = torch.exp(logstd)

        # 1) 输入：逐切片归一化（末尾对齐，前面的余数用第一个切片的统计量）
        rep_m = mean.repeat_interleave(p, dim=1)
        rep_s = std.repeat_interleave(p, dim=1)
        pad = l - k * p
        if pad > 0:
            rep_m = torch.cat([rep_m[:, :1].expand(b, pad, c), rep_m], dim=1)
            rep_s = torch.cat([rep_s[:, :1].expand(b, pad, c), rep_s], dim=1)
        x_n = (x_enc - rep_m) / rep_s

        dec_n = None
        if x_dec is not None:
            # 解码器输入统一用**最后一个切片**的统计量（它与预测起点最近）
            m_last = align_channels(mean[:, -1:, :], x_dec)
            s_last = align_channels(std[:, -1:, :], x_dec)
            dec_n = (x_dec - m_last) / s_last

        out = self.model(x_n, x_mark_enc, dec_n, x_mark_dec, mask)

        # 2) 输出：用 EWMA 聚合的未来切片统计量逆变换
        #    历史版本用 _ridge_extrapolate 做线性外推，实测 0 胜 41 负（见该函数 docstring）。
        #    stats_estimator="ridge" 仅供论文里复现那条负面对照。
        h = out.shape[1]
        n_future = int(math.ceil(h / p))
        if self.stats_estimator == "ridge":
            f_mean = _ridge_extrapolate(mean, n_future, self.lam)
            f_logstd = _ridge_extrapolate(logstd, n_future, self.lam)
        else:
            f_mean = _ewma_extrapolate(mean, n_future)
            f_logstd = _ewma_extrapolate(logstd, n_future)
        f_std = torch.exp(f_logstd.clamp(min=-8.0, max=8.0))
        o_m = align_channels(f_mean.repeat_interleave(p, dim=1)[:, :h, :], out)
        o_s = align_channels(f_std.repeat_interleave(p, dim=1)[:, :h, :], out)
        return out * o_s + o_m
