"""`src/features.py` 自检：每个统计量必须在「性质已知」的合成序列上给出正确排序。

这些测试同时是特征的**可解释性文档**：如果某个断言挂了，说明该特征不再度量
论文里声称的那个直觉，必须回去改实现或改论文表述。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from src import features as F


# --------------------------------------------------------------------------- #
# 1. 窗口级模式复杂度
# --------------------------------------------------------------------------- #
def test_permutation_entropy_ordering(white_noise, seasonal):
    """白噪声排列熵 ≈ 1，且显著高于强季节序列。"""
    pe_noise = F.permutation_entropy(white_noise[:, 0])
    pe_seas = F.permutation_entropy(seasonal[:, 0])
    assert 0.95 <= pe_noise <= 1.0
    assert pe_seas < pe_noise - 0.1
    assert 0.0 <= pe_seas <= 1.0


def test_permutation_entropy_monotone_series_is_low():
    """严格单调序列只有一种序数模式 → 熵 = 0。"""
    assert F.permutation_entropy(np.arange(500.0)) == pytest.approx(0.0, abs=1e-12)


def test_permutation_entropy_too_short_returns_nan():
    assert np.isnan(F.permutation_entropy(np.array([1.0, 2.0])))


def test_spectral_entropy_and_flatness_ordering(white_noise, seasonal):
    """谱熵/谱平坦度：白噪声接近 1，谐波序列明显更小。"""
    se_n, se_s = F.spectral_entropy(white_noise[:, 0]), F.spectral_entropy(seasonal[:, 0])
    sf_n, sf_s = F.spectral_flatness(white_noise[:, 0]), F.spectral_flatness(seasonal[:, 0])
    assert 0.9 <= se_n <= 1.0 and se_s < se_n
    assert 0.0 < sf_s < sf_n <= 1.0


def test_window_complexity_keys_and_ranges(white_noise, seasonal):
    cn = F.window_complexity(white_noise)
    cs = F.window_complexity(seasonal)
    assert set(cn) == {"perm_entropy", "perm_entropy_std", "spec_entropy",
                       "spec_entropy_std", "forecastability"}
    assert cn["perm_entropy"] > cs["perm_entropy"]
    # forecastability = 1 - 谱熵：季节序列必须更「可预测」
    assert cs["forecastability"] > cn["forecastability"]
    assert cn["forecastability"] == pytest.approx(1.0 - cn["spec_entropy"], abs=1e-9)


def test_window_complexity_cost_is_independent_of_length(seasonal):
    """窗口级统计只采样固定个数的窗口 → 序列长 10 倍，耗时不应线性增长。"""
    long_series = np.tile(seasonal, (10, 1))
    t0 = time.perf_counter()
    F.window_complexity(seasonal)
    t_short = time.perf_counter() - t0
    t0 = time.perf_counter()
    F.window_complexity(long_series)
    t_long = time.perf_counter() - t0
    assert t_long < max(t_short * 4.0, 0.5)


# --------------------------------------------------------------------------- #
# 2. 频域集中度
# --------------------------------------------------------------------------- #
def test_top_freq_energy_and_dominant_period(seasonal, white_noise):
    ratio_s, period_s = F.top_freq_energy_ratio(seasonal[:, 0], k=5)
    ratio_n, _ = F.top_freq_energy_ratio(white_noise[:, 0], k=5)
    assert ratio_s > 0.8 > ratio_n
    # 主频周期应落在 24 附近（rfft 长度反推有 ±1 bin 误差）
    assert 22.0 <= period_s <= 26.0


def test_power_spectrum_drops_dc():
    x = np.ones(256) * 5.0 + np.sin(2 * np.pi * np.arange(256) / 16)
    psd = F.power_spectrum(x)
    assert psd.size == 128  # rfft(256) = 129 项，去掉 DC
    assert int(np.argmax(psd)) == 16 - 1  # 周期 16 → 第 16 个频率 bin


# --------------------------------------------------------------------------- #
# 3. 非平稳度
# --------------------------------------------------------------------------- #
def test_rolling_drift_walk_vs_noise(random_walk, white_noise):
    d_walk = F.rolling_drift(random_walk[:, 0], 96)
    d_noise = F.rolling_drift(white_noise[:, 0], 96)
    assert d_walk["mean_drift"] > 5 * d_noise["mean_drift"]
    assert d_walk["range_drift"] > d_noise["range_drift"]


def test_rolling_drift_constant_series_is_zero():
    d = F.rolling_drift(np.ones(1000), 96)
    assert d == {"mean_drift": 0.0, "var_drift": 0.0, "range_drift": 0.0}


def test_adf_detects_unit_root(random_walk, white_noise):
    a_walk = F.adf_stat(random_walk[:, 0])
    a_noise = F.adf_stat(white_noise[:, 0])
    assert a_walk["adf_pvalue"] > 0.05 > a_noise["adf_pvalue"]
    assert a_noise["adf_stat"] < a_walk["adf_stat"]  # 越负 = 越平稳


# --------------------------------------------------------------------------- #
# 4. 通道相关谱
# --------------------------------------------------------------------------- #
def test_channel_spectrum_rank_one_vs_independent(rank_one, white_noise):
    s1 = F.channel_spectrum(rank_one)
    sn = F.channel_spectrum(white_noise)
    assert s1["eig_top1_ratio"] > 0.9            # 一个公共因子解释几乎全部方差
    assert s1["eff_rank_ratio"] < 0.35
    assert sn["eff_rank_ratio"] > 0.9            # 独立通道 → 有效秩 ≈ C
    assert s1["mean_abs_corr"] > sn["mean_abs_corr"]
    assert s1["n_channels"] == 8.0


def test_channel_spectrum_single_channel_degenerate():
    s = F.channel_spectrum(np.random.default_rng(0).standard_normal((100, 1)))
    assert s["eff_rank_ratio"] == 1.0 and s["n_channels"] == 1.0


# --------------------------------------------------------------------------- #
# 5. 趋势/季节强度与 ACF 衰减
# --------------------------------------------------------------------------- #
def test_stl_strengths(seasonal, random_walk, white_noise):
    s_seas = F.stl_strengths(seasonal[:2000, 0], PERIOD := 24)
    s_walk = F.stl_strengths(random_walk[:2000, 0], PERIOD)
    s_noise = F.stl_strengths(white_noise[:2000, 0], PERIOD)
    assert s_seas["seasonal_strength"] > 0.8
    assert s_seas["seasonal_strength"] > s_noise["seasonal_strength"]
    assert s_walk["trend_strength"] > s_noise["trend_strength"]
    for v in list(s_seas.values()) + list(s_walk.values()):
        assert 0.0 <= v <= 1.0


def test_acf_decay(random_walk, white_noise):
    a_walk = F.acf_decay(random_walk[:, 0], nlags=200)
    a_noise = F.acf_decay(white_noise[:, 0], nlags=200)
    assert a_walk["acf1"] > 0.95 > abs(a_noise["acf1"])
    assert a_walk["acf_e_folding"] > a_noise["acf_e_folding"]  # 衰减更慢
    assert a_walk["acf_decay_rate"] < a_noise["acf_decay_rate"]


def test_acf_fft_matches_naive_definition(seasonal):
    x = seasonal[:500, 0]
    ac = F.acf_fft(x, 10)
    xc = x - x.mean()
    naive = np.array([np.dot(xc[: len(xc) - k], xc[k:]) for k in range(11)]) / np.dot(xc, xc)
    assert np.allclose(ac, naive, atol=1e-10)


# --------------------------------------------------------------------------- #
# 6. 组合特征表
# --------------------------------------------------------------------------- #
def test_series_features_complete_and_finite(seasonal):
    f = F.series_features(seasonal, period=24)
    prefixes = {k.split("_")[0] for k in f}
    assert {"pc", "ns", "ch", "fr", "sd"} <= prefixes
    bad = {k: v for k, v in f.items() if not np.isfinite(v)}
    assert not bad, f"非有限特征: {bad}"


def test_horizon_features_are_horizon_dependent(random_walk):
    f96 = F.horizon_features(random_walk, seq_len=96, pred_len=96, period=24)
    f720 = F.horizon_features(random_walk, seq_len=96, pred_len=720, period=24)
    # 随机游走上，horizon 越长，输入窗口→输出窗口的均值漂移越大（归一化插件的作用对象）
    assert f720["hz_mean_shift"] > f96["hz_mean_shift"]
    assert f720["hz_ratio_seq"] == pytest.approx(720 / 96)
    assert f720["hz_h_over_period"] == pytest.approx(30.0)
    assert f96["hz_acf_at_h"] > f720["hz_acf_at_h"]


def test_dataset_rows_shape_and_shared_dataset_level_features(seasonal):
    rows = F.dataset_rows(seasonal, "synth", seq_len=96, horizons=[96, 336], period=24)
    assert len(rows) == 2
    assert rows[0]["pc_perm_entropy"] == rows[1]["pc_perm_entropy"]  # 数据集级特征复用
    assert rows[0]["hz_mean_shift"] != rows[1]["hz_mean_shift"]      # horizon 级特征不同


def test_feature_columns_excludes_keys():
    import pandas as pd

    df = pd.DataFrame({"dataset": ["a"], "pred_len": [96], "backbone": ["DLinear"],
                       "pc_perm_entropy": [0.9]})
    assert F.feature_columns(df) == ["pc_perm_entropy"]


def test_load_train_split_respects_tslib_boundaries(tmp_path):
    """ETTh1 训练段固定为 12*30*24 行，并按训练段统计量做 z-score（无泄漏）。"""
    import pandas as pd

    n = 12 * 30 * 24 + 500
    df = pd.DataFrame({"date": pd.date_range("2016-07-01", periods=n, freq="h"),
                       "a": np.arange(n, dtype=float), "b": np.arange(n, dtype=float) * -2})
    p = tmp_path / "ETTh1.csv"
    df.to_csv(p, index=False)
    X = F.load_train_split(p, "ETTh1", seq_len=96)
    assert X.shape == (12 * 30 * 24, 2)
    assert abs(X.mean()) < 1e-8 and abs(X.std() - 1.0) < 1e-6


def test_demo_table_ranks_three_regimes():
    """CLI 自检表：噪声最复杂、季节最可预测、随机游走最非平稳。"""
    df = F._demo()
    by = df.drop_duplicates("dataset").set_index("dataset")
    assert by.loc["synth_noise", "pc_perm_entropy"] > by.loc["synth_seasonal", "pc_perm_entropy"]
    assert by.loc["synth_seasonal", "fr_topk_energy"] > by.loc["synth_noise", "fr_topk_energy"]
    assert by.loc["synth_walk", "ns_mean_drift"] > by.loc["synth_noise", "ns_mean_drift"]
