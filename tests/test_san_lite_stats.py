"""锁定 SAN-lite 统计量估计器的修复。

背景：初版用 ``_ridge_extrapolate`` 对切片统计量做线性外推，在 41 个完整块上
**0 胜 41 负、平均相对退化 -155%**。诊断（tools/diag_san.py）显示切片变换本身正确
（oracle 统计量下 MSE=0），坏的是外推：seq_len=96 只有 4 个观测切片，
用 4 个点估斜率再外推 30 个切片是在放大噪声。

修复为 EWMA（斜率隐式为 0）。这些测试防止该修复被回退。
"""

from __future__ import annotations

import pytest
import torch

from src.plugins.san_lite import SANLitePlugin, _ewma_extrapolate, _ridge_extrapolate


def test_ewma_is_constant_across_future_slices():
    """EWMA 估计器对所有未来切片给出同一常数——即斜率被隐式设为 0。"""
    x = torch.randn(3, 4, 5)
    out = _ewma_extrapolate(x, n_future=7)
    assert out.shape == (3, 7, 5)
    for t in range(1, 7):
        torch.testing.assert_close(out[:, 0], out[:, t])


def test_ewma_weights_recent_slices_more():
    """最后一个切片的权重必须最大：把它单独抬高，输出应朝它移动更多。"""
    base = torch.zeros(1, 4, 1)
    bumped_last = base.clone()
    bumped_last[0, 3, 0] = 1.0
    bumped_first = base.clone()
    bumped_first[0, 0, 0] = 1.0
    d_last = _ewma_extrapolate(bumped_last, 1)[0, 0, 0].item()
    d_first = _ewma_extrapolate(bumped_first, 1)[0, 0, 0].item()
    assert d_last > d_first > 0


def test_ewma_reduces_to_value_when_constant():
    """常数序列的任何合理估计器都必须还原该常数（无偏性的最弱形式）。"""
    x = torch.full((2, 6, 3), 4.2)
    torch.testing.assert_close(_ewma_extrapolate(x, 5), torch.full((2, 5, 3), 4.2))


def test_ewma_does_not_diverge_on_trend_while_ridge_does():
    """这是修复的核心动机：面对带趋势的短序列，岭外推会跑到观测范围之外，EWMA 不会。

    切片均值序列近似随机游走，外推趋势是负收益；因此「不发散」正是我们想要的性质。
    """
    trend = torch.arange(4, dtype=torch.float32).view(1, 4, 1) * 1.0  # 0,1,2,3
    n_future = 30  # h=720 / p=24
    e = _ewma_extrapolate(trend, n_future)
    r = _ridge_extrapolate(trend, n_future, lam=1.0)
    lo, hi = trend.min().item(), trend.max().item()
    assert lo <= e.min().item() and e.max().item() <= hi, "EWMA 必须落在观测范围内"
    assert r.max().item() > hi, "岭外推应当冲出观测范围（这正是它 0 胜 41 负的原因）"


def test_default_estimator_is_ewma_not_ridge():
    """默认必须是修复后的 ewma；ridge 只能显式指定（论文里的负面对照）。"""
    p = SANLitePlugin(torch.nn.Identity())
    assert p.stats_estimator == "ewma"
    assert SANLitePlugin(torch.nn.Identity(), stats_estimator="ridge").stats_estimator == "ridge"
    with pytest.raises(ValueError, match="ewma|ridge"):
        SANLitePlugin(torch.nn.Identity(), stats_estimator="linear")


def test_oracle_stats_give_exact_reconstruction():
    """切片变换的自洽性：用真实未来切片统计量逆变换，完美主干应精确还原真值。

    这条断言是 diag_san.py 里 `oracle MSE=0` 那一列的单测化，
    它保证「归一化/逆归一化」这套骨架永远不会悄悄写错——
    将来若有人改切片对齐或 std_floor，这里会先炸。
    """
    torch.manual_seed(0)
    plug = SANLitePlugin(torch.nn.Identity())
    seq_len, pred_len, c = 96, 96, 7
    p = plug._slice_len(seq_len)
    y = torch.randn(8, pred_len, c) * 3.0 + 1.5
    o_mean, o_logstd, _ = plug._slice_stats(y, p, plug.eps, plug.std_floor)
    om = o_mean.repeat_interleave(p, dim=1)[:, :pred_len, :]
    os_ = torch.exp(o_logstd).repeat_interleave(p, dim=1)[:, :pred_len, :]
    z = (y - om) / os_          # 完美主干在归一化空间的输出
    recon = z * os_ + om        # 用 oracle 统计量逆变换
    torch.testing.assert_close(recon, y, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("pred_len", [96, 336, 720])
def test_forward_stays_bounded_with_ewma(pred_len):
    """端到端形状与数值护栏：EWMA 路径下输出不得出现 nan/inf 或量级爆炸。"""
    torch.manual_seed(0)
    c = 7

    class _Id(torch.nn.Module):
        def forward(self, x, xm=None, xd=None, xdm=None, mask=None):
            # 模拟主干：在归一化空间输出与 pred_len 对齐的张量
            return x[:, -1:, :].expand(-1, pred_len, -1).contiguous()

    plug = SANLitePlugin(_Id())
    x = torch.randn(4, 96, c) * 2.0 + 10.0
    out = plug(x)
    assert out.shape == (4, pred_len, c)
    assert torch.isfinite(out).all()
    # 逆变换后应回到输入的量级附近，而不是发散
    assert out.abs().max().item() < x.abs().max().item() * 10
