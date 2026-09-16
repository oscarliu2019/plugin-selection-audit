"""逐窗口特征在退化输入（恒定通道、半段恒定、死通道）下的行为。

2026-09-14 code review：这些情形此前被静默编码成 0.0，与「真的无结构」不可区分，
门控器因此拿到自相矛盾的假特征（恒定窗口同时表现为「最强结构」和「无结构」）。
下游 HistGradientBoosting* 原生支持缺失值，所以如实报 NaN 是正确做法。
"""

from __future__ import annotations

import numpy as np
import pytest

from src.features_window import window_features


L, C = 96, 6


def _seasonal(n=L, c=C, period=24, seed=0):
    g = np.random.default_rng(seed)
    t = np.arange(n)[:, None]
    return np.sin(2 * np.pi * t / period) + 0.1 * g.standard_normal((n, c))


def test_flatness_column_renamed_to_dominant_period():
    """旧名 w_fr_flatness 实为主频周期，与 features.py 的 fr_flatness 同名不同义。"""
    f = window_features(_seasonal())
    assert "w_fr_flatness" not in f, "同名不同义的列必须已改名"
    assert "w_fr_dom_period" in f
    # 主频周期以采样点为单位，量级为 O(seq_len)，绝不是 (0,1] 的平坦度
    assert f["w_fr_dom_period"] > 1.0


def test_constant_channel_yields_nan_not_zero_structure():
    """全恒定窗口：ACF / 趋势必须是 NaN，而不是 0（0 = 白噪声，含义相反）。"""
    W = np.full((L, C), 3.14)
    f = window_features(W)
    for k, v in f.items():
        if k.startswith("w_sd_acf") or k in ("w_sd_trend_r2", "w_sd_slope_abs"):
            assert np.isnan(v), f"{k} 应为 NaN，实际 {v}"


def test_partially_constant_channel_does_not_poison_alive_channels():
    """一个死通道不应把整行特征拖成 NaN：其余通道仍要给出有限值。"""
    W = _seasonal()
    W[:, 0] = 1.0
    f = window_features(W)
    assert np.isfinite(f["w_sd_trend_r2"])
    assert np.isfinite(f["w_sd_acf1"])
    assert f["w_ch_n_alive"] == C - 1
    assert f["w_ch_alive_frac"] == pytest.approx((C - 1) / C)


def test_std_contrast_is_bounded_when_first_half_constant():
    """裸比值 s2/s1 在前半段恒定时爆到 1e6~1e8（实测约 1% 的窗口），必须改用有界对比度。"""
    W = _seasonal()
    W[: L // 2, :] = 0.0
    f = window_features(W)
    assert "w_ns_std_ratio" not in f, "无界比值列必须已被有界对比度取代"
    v = f["w_ns_std_contrast"]
    assert np.isfinite(v)
    assert -1.0 <= v <= 1.0
    # 前半段恒定 => s1=0 => 对比度趋近 +1
    assert v > 0.9


def test_std_contrast_sign_and_symmetry():
    """0 = 前后等方差；符号区分「哪半段更稳」，且两种情形对称。"""
    g = np.random.default_rng(7)
    flat = window_features(g.standard_normal((L, C)))["w_ns_std_contrast"]
    assert abs(flat) < 0.3, "等方差窗口的对比度应接近 0"

    W = _seasonal(seed=5)
    front_dead, back_dead = W.copy(), W.copy()
    front_dead[: L // 2, :] = 0.0
    back_dead[L // 2 :, :] = 0.0
    a = window_features(front_dead)["w_ns_std_contrast"]
    b = window_features(back_dead)["w_ns_std_contrast"]
    assert a > 0 > b
    assert a == pytest.approx(-b, abs=0.05)


def test_std_contrast_is_monotone_in_variance_ratio():
    """与原始比值同单调：后半段方差越大，对比度越大。"""
    vals = []
    for amp in (0.5, 1.0, 2.0, 4.0):
        g = np.random.default_rng(11)
        W = np.empty((L, C))
        W[: L // 2] = g.standard_normal((L // 2, C))
        W[L // 2 :] = amp * g.standard_normal((L - L // 2, C))
        vals.append(window_features(W)["w_ns_std_contrast"])
    assert vals == sorted(vals), f"应单调递增，实际 {vals}"


def test_all_constant_channels_report_nan_cross_channel_stats():
    """通道全死时跨通道结构未定义，必须报 NaN 而不是编造 0 / 1。"""
    f = window_features(np.zeros((L, C)))
    assert f["w_ch_n_alive"] == 0
    assert np.isnan(f["w_ch_corr_abs_mean"])
    assert np.isnan(f["w_ch_corr_mean"])
    assert np.isnan(f["w_ch_eff_rank"])


def test_dead_channels_do_not_shrink_effective_rank():
    """死通道贡献 0 奇异值会压低 eff_rank，被误读成「通道冗余→通道类插件有效」。

    正确行为：死通道被剔除，eff_rank 只反映存活通道的冗余度，
    因此加入死通道不应让 eff_rank 变小。
    """
    W = _seasonal(seed=1)
    base = window_features(W)["w_ch_eff_rank"]
    W2 = np.concatenate([W, np.zeros((L, 3))], axis=1)
    with_dead = window_features(W2)["w_ch_eff_rank"]
    assert with_dead == pytest.approx(base, rel=1e-9)


def test_channel_correlation_denominator_is_stable_across_windows():
    """w_ch_corr_abs_mean 的分母必须只由存活通道数决定，跨窗口可比。"""
    a = window_features(_seasonal(seed=2))
    b = window_features(_seasonal(seed=3))
    assert a["w_ch_n_alive"] == b["w_ch_n_alive"] == C
    assert np.isfinite(a["w_ch_corr_abs_mean"]) and np.isfinite(b["w_ch_corr_abs_mean"])
    assert -1.0 <= a["w_ch_corr_mean"] <= 1.0
    assert 0.0 <= a["w_ch_corr_abs_mean"] <= 1.0


def test_feature_keys_are_stable_across_degenerate_inputs():
    """无论输入多退化，特征名集合必须一致，否则拼不成一张表。"""
    keys = [
        set(window_features(x).keys())
        for x in (_seasonal(), np.zeros((L, C)), np.full((L, C), 2.0))
    ]
    assert keys[0] == keys[1] == keys[2]
