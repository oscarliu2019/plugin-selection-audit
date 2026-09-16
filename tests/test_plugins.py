"""`src/plugins/` 自检：零侵入 + 可逆性 + 无未来信息泄漏 + 数值护栏。

用一个「假骨干」代替 TSLib 模型：它只做恒等/线性映射，因此插件的行为可以被解析验证。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.plugins import PLUGIN_KINDS, available, kind_of, make_criterion, wrap
from src.plugins.san_lite import _ridge_extrapolate

B, L, H, C = 4, 96, 24, 7


class EchoBackbone(nn.Module):
    """假骨干：输出 = 输入窗口最后 H 步（恒等式行为，便于解析验证插件的可逆性）。"""

    def __init__(self, pred_len: int = H) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.dummy = nn.Linear(1, 1)
        self.seen: torch.Tensor | None = None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        self.seen = x_enc.detach().clone()
        return x_enc[:, -self.pred_len :, :]


def _batch(seed: int = 0, shift: float = 0.0, scale: float = 1.0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, L, C, generator=g) * scale + shift
    x_dec = torch.randn(B, 48 + H, C, generator=g) * scale + shift
    return x, x_dec


# --------------------------------------------------------------------------- #
# 注册表
# --------------------------------------------------------------------------- #
def test_registry_is_consistent():
    # 主矩阵的 4 个臂必须始终在册；`fredf_sqrth` 等消融/公平性变体是额外注册的实现，
    # 它们通过 matrix 里的 `impl:` 字段被引用（见 config.plugin_impl），故只要求「⊇」。
    assert {"none", "revin", "san_lite", "fredf"} <= set(available())
    assert set(PLUGIN_KINDS) == set(available())
    assert kind_of("fredf") == "loss" and kind_of("revin") == "io"
    assert kind_of("fredf_sqrth") == "loss"


def test_wrap_unknown_plugin_raises():
    try:
        wrap(EchoBackbone(), "revin2")
    except KeyError as exc:
        assert "未知插件" in str(exc)
    else:
        raise AssertionError("应当抛 KeyError")


def test_wrap_ignores_unknown_params():
    """matrix.yaml 里的整块参数可以直接传进来，不认识的键必须被忽略而不是报错。"""
    m = wrap(EchoBackbone(), "revin", eps=1e-4, slice_len=48, 随便一个键=1)
    assert m.eps == 1e-4


def test_none_plugin_is_bit_exact_passthrough():
    x, x_dec = _batch()
    bb = EchoBackbone()
    y_raw = bb(x, None, x_dec, None)
    y_wrapped = wrap(bb, "none")(x, None, x_dec, None)
    assert torch.equal(y_raw, y_wrapped)
    assert wrap(bb, "none").extra_param_count() == 0


def test_all_io_plugins_preserve_shape_and_add_no_parameters():
    x, x_dec = _batch()
    for name in ("none", "revin", "san_lite", "fredf"):
        m = wrap(EchoBackbone(), name)
        out = m(x, None, x_dec, None)
        assert out.shape == (B, H, C), name
        assert torch.isfinite(out).all(), name
        assert m.extra_param_count() == 0, f"{name} 引入了额外参数，违反零参数约束"
        assert float(m.pop_aux_loss()) == 0.0


# --------------------------------------------------------------------------- #
# RevIN
# --------------------------------------------------------------------------- #
def test_revin_normalizes_encoder_input_to_zero_mean_unit_var():
    x, x_dec = _batch(shift=100.0, scale=5.0)
    bb = EchoBackbone()
    wrap(bb, "revin")(x, None, x_dec, None)
    seen = bb.seen
    assert seen is not None
    assert seen.mean(dim=1).abs().max() < 1e-4
    assert (seen.std(dim=1, unbiased=False) - 1.0).abs().max() < 1e-3


def test_revin_is_exactly_invertible_on_identity_backbone():
    """恒等骨干下 RevIN 必须还原原始尺度（这是「可逆」的定义）。"""
    x, x_dec = _batch(shift=100.0, scale=5.0)
    out = wrap(EchoBackbone(), "revin")(x, None, x_dec, None)
    assert torch.allclose(out, x[:, -H:, :], atol=1e-3)


def test_revin_uses_no_future_information():
    """改变 x_dec 的未来段不得影响归一化统计量（无泄漏）。"""
    x, x_dec = _batch()
    bb1, bb2 = EchoBackbone(), EchoBackbone()
    x_dec2 = x_dec.clone()
    x_dec2[:, -H:, :] += 1000.0
    o1 = wrap(bb1, "revin")(x, None, x_dec, None)
    o2 = wrap(bb2, "revin")(x, None, x_dec2, None)
    assert torch.allclose(o1, o2, atol=1e-6)


def test_revin_shifts_prediction_with_input_level():
    """输入整体平移 δ → 预测整体平移 δ（归一化插件的核心行为）。"""
    x, x_dec = _batch()
    o1 = wrap(EchoBackbone(), "revin")(x, None, x_dec, None)
    o2 = wrap(EchoBackbone(), "revin")(x + 7.0, None, x_dec + 7.0, None)
    assert torch.allclose(o2 - o1, torch.full_like(o1, 7.0), atol=1e-3)


def test_revin_affine_adds_exactly_2c_parameters():
    m = wrap(EchoBackbone(), "revin", affine=True, n_channels=C)
    assert m.extra_param_count() == 2 * C
    x, x_dec = _batch()
    assert m(x, None, x_dec, None).shape == (B, H, C)


def test_revin_handles_ms_mode_channel_mismatch():
    """MS 模式：输出通道数 < 输入通道数时统计量要正确对齐。"""
    class LastChannel(EchoBackbone):
        def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
            return x_enc[:, -H:, -1:]

    x, x_dec = _batch(shift=50.0)
    out = wrap(LastChannel(), "revin")(x, None, x_dec, None)
    assert out.shape == (B, H, 1)
    assert torch.allclose(out, x[:, -H:, -1:], atol=1e-3)


# --------------------------------------------------------------------------- #
# SAN-lite
# --------------------------------------------------------------------------- #
def test_san_lite_slice_normalization_and_slice_rule():
    x, x_dec = _batch(shift=10.0, scale=2.0)
    m = wrap(EchoBackbone(), "san_lite", slice_len=24)
    assert m._slice_len(96) == 24 and m._slice_len(36) == 12 and m._slice_len(9) == 4
    out = m(x, None, x_dec, None)
    assert out.shape == (B, H, C) and torch.isfinite(out).all()


def test_san_lite_tracks_piecewise_level_shift():
    """构造分段跳变序列：SAN-lite 的切片统计量应吃掉段内电平，输出仍在合理范围。"""
    t = torch.arange(L, dtype=torch.float32)
    level = (t // 24) * 5.0                       # 每 24 步跳一个台阶
    x = (level[None, :, None] + torch.randn(B, L, C) * 0.1)
    x_dec = torch.zeros(B, 48 + H, C)
    bb = EchoBackbone()
    out = wrap(bb, "san_lite", slice_len=24)(x, None, x_dec, None)
    assert bb.seen is not None
    # 归一化后每个切片近似零均值（吃掉了台阶）
    sliced = bb.seen.reshape(B, L // 24, 24, C).mean(dim=2).abs().max()
    assert float(sliced) < 0.5
    # 逆变换后的预测应当继续跟随上升的电平（线性外推），且有限
    assert torch.isfinite(out).all() and out.mean() > x[:, :24].mean()


def test_san_lite_std_floor_prevents_explosion():
    """常数切片（实测 ETTh1 存在 std=0 的切片）不得导致数值爆炸——这是曾经踩过的坑。"""
    x = torch.zeros(B, L, C)
    x[:, 48:, :] = torch.randn(B, 48, C) * 0.5   # 前半段完全恒定
    x_dec = torch.zeros(B, 48 + H, C)
    bb = EchoBackbone()
    out = wrap(bb, "san_lite", slice_len=24, std_floor=0.1)(x, None, x_dec, None)
    assert bb.seen is not None
    assert bb.seen.abs().max() < 1.0 / 0.1 + 1.0   # 放大倍数被 1/std_floor 限住
    assert torch.isfinite(out).all() and out.abs().max() < 1e3


def test_ridge_extrapolate_recovers_linear_trend():
    k, n_future = 8, 4
    slope = 0.5
    stats = (slope * torch.arange(k, dtype=torch.float32)).view(1, k, 1).repeat(2, 1, 3)
    pred = _ridge_extrapolate(stats, n_future, lam=0.0)
    expected = slope * torch.arange(k, k + n_future, dtype=torch.float32)
    # 夹紧护栏允许的范围内应当贴合真实线性延伸
    assert pred.shape == (2, n_future, 3)
    assert torch.allclose(pred[0, :, 0], expected, atol=1e-4)


def test_ridge_extrapolate_shrinks_with_lambda_and_clamps():
    stats = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(1, 4, 1)
    strong = _ridge_extrapolate(stats, 30, lam=1e6)
    assert torch.allclose(strong, torch.full_like(strong, 1.5), atol=1e-3)  # 斜率被压向 0
    far = _ridge_extrapolate(stats, 100, lam=0.0)
    assert float(far.max()) <= 3.0 + 3.0 + 1e-4                            # hi + span 护栏
    assert torch.isfinite(far).all()


def test_san_lite_uses_no_future_information():
    x, x_dec = _batch()
    x_dec2 = x_dec.clone()
    x_dec2[:, -H:, :] += 500.0
    o1 = wrap(EchoBackbone(), "san_lite")(x, None, x_dec, None)
    o2 = wrap(EchoBackbone(), "san_lite")(x, None, x_dec2, None)
    assert torch.allclose(o1, o2, atol=1e-6)


# --------------------------------------------------------------------------- #
# FreDF（loss 型）
# --------------------------------------------------------------------------- #
def test_fredf_is_a_loss_plugin_not_a_model_wrapper():
    x, x_dec = _batch()
    bb = EchoBackbone()
    assert torch.equal(wrap(bb, "fredf")(x, None, x_dec, None), bb(x, None, x_dec, None))
    assert isinstance(make_criterion("fredf", alpha=0.5), nn.Module)
    assert isinstance(make_criterion("none"), nn.MSELoss)
    assert isinstance(make_criterion("revin"), nn.MSELoss)


def test_fredf_zero_at_perfect_prediction_and_positive_otherwise():
    crit = make_criterion("fredf", alpha=0.5)
    y = torch.randn(B, H, C)
    assert float(crit(y.clone(), y)) == 0.0
    assert float(crit(y + 0.1, y)) > 0.0


def test_fredf_alpha_zero_equals_mse():
    y, p = torch.randn(B, H, C), torch.randn(B, H, C)
    assert torch.allclose(make_criterion("fredf", alpha=0.0)(p, y), nn.MSELoss()(p, y))


def test_fredf_prefers_spectrally_sparse_errors_at_equal_mse():
    """FreDF 的机理检验：在 **MSE 完全相同** 的两种残差下，频域 L1 会更重地惩罚
    宽带（白噪声型）残差，而对集中在少数频率的残差惩罚更轻。

    这正是频域损失的立论点——它把「误差谱的稀疏性」写进了目标函数；
    也解释了为什么本文预期 `fr_topk_energy` 高（能量集中）的数据集上它更可能有增益。
    """
    n, ch = 48, 3
    g = torch.Generator().manual_seed(0)
    t = torch.arange(n, dtype=torch.float32)
    true = torch.sin(2 * torch.pi * t / 12).view(1, n, 1).repeat(B, 1, ch)
    e_sparse = (0.2 * torch.sin(2 * torch.pi * t / 8)).view(1, n, 1).repeat(B, 1, ch)
    e_broad = torch.randn(B, n, ch, generator=g)
    e_broad = e_broad / e_broad.pow(2).mean().sqrt() * e_sparse.pow(2).mean().sqrt()

    mse = nn.MSELoss()
    fredf = make_criterion("fredf", alpha=0.9)
    assert float(mse(true + e_sparse, true)) == pytest.approx(
        float(mse(true + e_broad, true)), rel=1e-5)          # 两种残差的 MSE 完全一致
    assert float(fredf(true + e_sparse, true)) < 0.5 * float(fredf(true + e_broad, true))


def test_fredf_is_invariant_to_a_common_circular_shift():
    """pred 与 true 同时循环平移只会给谱乘上同一个相位因子 → 复数差的模不变。

    这条性质说明本实现用的是**复数差的 L1**（同时约束幅值与相位），
    而不是只比较幅值谱；论文里必须这样描述，否则与实现不符。
    """
    g = torch.Generator().manual_seed(1)
    true = torch.randn(B, H, C, generator=g)
    pred = true + 0.1 * torch.randn(B, H, C, generator=g)
    fredf = make_criterion("fredf", alpha=1.0)
    a = float(fredf(pred, true))
    b = float(fredf(torch.roll(pred, 5, dims=1), torch.roll(true, 5, dims=1)))
    assert a == pytest.approx(b, rel=1e-5)
    # 只平移 pred（真正的相位错位）则损失显著变大
    assert float(fredf(torch.roll(pred, 5, dims=1), true)) > 3 * a


def test_fredf_rejects_bad_alpha():
    try:
        make_criterion("fredf", alpha=1.5)
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("应当抛 ValueError")


def test_fredf_gradient_flows():
    crit = make_criterion("fredf", alpha=0.5)
    p = torch.randn(B, H, C, requires_grad=True)
    crit(p, torch.randn(B, H, C)).backward()
    assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
