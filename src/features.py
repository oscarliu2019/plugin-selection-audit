"""⭐ 复杂度 / 可预测性统计量：本项目的核心科学贡献之一。

设计契约
--------
1. **廉价**：所有统计量都是 O(n) 或 O(n log n)。刻意不使用样本熵/近似熵的朴素实现
   （O(n²)，Traffic 上会跑到分钟级），改用**排列熵 (Permutation Entropy)** 与
   **谱熵**——两者都是 O(n log n) 且对幅度单调变换不敏感。
2. **无泄漏**：所有特征只从**训练段**（TSLib 的 border1s[0]:border2s[0]）计算，
   并且只用训练段统计量做 z-score。这是审稿人必查的点（尽调 §5.2 第 7 条）。
3. **尺度无关**：先对每个通道做训练段 z-score，因此特征不随单位变化。
4. **与研究问题对齐**：特征分四组，每组对应一类插件的作用机理——
   - `pc_*`  窗口级模式复杂度 → 对应 Accuracy Law 的误差下界视角（还有没有提升空间）
   - `ns_*`  非平稳度        → 对应 RevIN / SAN / FAN 这类归一化插件该不该挂
   - `ch_*`  通道相关谱      → 对应 CCM / LIFT 这类通道插件该不该挂
   - `fr_*`  频域集中度      → 对应 FreDF 这类频域损失该不该挂
   - `sd_*`  趋势/季节强度、自相关衰减 → 通用可预测性
   - `hz_*`  horizon 条件特征（依赖 pred_len）→ 决定「同一数据集不同 horizon 该不该换插件」

用法
----
    python -m src.features --config configs/matrix.yaml --out artifacts/features.csv
    python -m src.features --demo          # 用合成序列自检（无需下载数据）
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

FloatArray = np.ndarray


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureConfig:
    """特征抽取的预算参数（全部为固定值，不做逐数据集调参）。"""

    window: int = 96          # 窗口级统计的窗口长度（与主表 look-back 一致）
    n_windows: int = 64       # 采样多少个窗口做窗口级统计（O(1) 相对序列长度）
    max_channels: int = 32    # 通道逐条统计时的采样上限（Traffic 862 通道时生效）
    max_len: int = 30000      # STL / ADF 等中等成本统计的序列截断长度
    pe_order: int = 3         # 排列熵嵌入维度 m（m!=6 种模式，n>>6 时估计稳定）
    top_k_freq: int = 5       # 主频集中度取 top-k
    acf_nlags: int = 200      # 自相关最大滞后
    seed: int = 26            # 通道/窗口采样的随机种子（保证可复现）


DEFAULT_CFG = FeatureConfig()


# --------------------------------------------------------------------------- #
# 单通道基础统计量
# --------------------------------------------------------------------------- #
def permutation_entropy(x: FloatArray, m: int = 3, tau: int = 1, normalize: bool = True) -> float:
    """排列熵：窗口内**序数模式**的香农熵。

    直觉：值越大说明局部升降模式越接近随机（可预测性低、模式复杂度高）；
    白噪声 ≈ 1，单调/正弦等强结构序列 → 明显小于 1。
    复杂度：O(n · m log m)（m 固定为 3，实际 O(n)）。
    """
    x = np.asarray(x, dtype=np.float64)
    span = (m - 1) * tau + 1
    if x.size < span + 1:
        return float("nan")
    emb = np.lib.stride_tricks.sliding_window_view(x, span)[:, :: tau]
    order = np.argsort(emb, axis=1, kind="stable")
    radix = m ** np.arange(m)
    codes = (order * radix).sum(axis=1)
    _, counts = np.unique(codes, return_counts=True)
    p = counts / counts.sum()
    h = float(-(p * np.log(p)).sum())
    return h / math.log(math.factorial(m)) if normalize else h


def power_spectrum(x: FloatArray) -> FloatArray:
    """去均值后的单边功率谱（丢掉 DC 项）。复杂度 O(n log n)。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if not np.isfinite(x).all() or x.size < 4:
        return np.zeros(1)
    spec = np.abs(np.fft.rfft(x)) ** 2
    return spec[1:] if spec.size > 1 else spec


def spectral_entropy(x: FloatArray) -> float:
    """谱熵（归一化到 [0,1]）：功率谱的香农熵。

    直觉：能量分散在所有频率 → 接近白噪声 → 熵≈1 → 难预测；
    能量集中在少数频率 → 熵小 → 易被频域方法（如 FreDF）利用。
    复杂度 O(n log n)。
    """
    psd = power_spectrum(x)
    s = psd.sum()
    if s <= 0 or psd.size < 2:
        return float("nan")
    p = psd / s
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(psd.size))


def spectral_flatness(x: FloatArray) -> float:
    """谱平坦度 = 几何均值/算术均值 ∈ (0,1]：1 表示白噪声，接近 0 表示强谐波结构。O(n log n)。"""
    psd = power_spectrum(x)
    psd = psd[psd > 0]
    if psd.size < 2:
        return float("nan")
    return float(np.exp(np.log(psd).mean()) / psd.mean())


def top_freq_energy_ratio(x: FloatArray, k: int = 5) -> tuple[float, float]:
    """(top-k 频率能量占比, 主频对应周期)。

    直觉：占比高 = 少数几个周期解释了大部分方差 → 频域损失/周期先验类插件有利可图。
    复杂度 O(n log n)。
    """
    psd = power_spectrum(x)
    s = psd.sum()
    if s <= 0 or psd.size < 2:
        return float("nan"), float("nan")
    k = min(k, psd.size)
    idx = np.argpartition(psd, -k)[-k:]
    ratio = float(psd[idx].sum() / s)
    n = 2 * psd.size + 1  # rfft 长度反推（近似即可，只用于报告主周期量级）
    dom = int(np.argmax(psd)) + 1
    period = float(n / dom)
    return ratio, period


def acf_fft(x: FloatArray, nlags: int) -> FloatArray:
    """基于 FFT 的自相关函数（lag 0..nlags）。复杂度 O(n log n)。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = x.size
    if n < 4 or x.std() == 0:
        return np.full(nlags + 1, np.nan)
    nfft = 1 << (2 * n - 1).bit_length()
    f = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(f * np.conjugate(f), nfft)[: nlags + 1]
    if ac[0] <= 0:
        return np.full(nlags + 1, np.nan)
    return ac / ac[0]


def acf_decay(x: FloatArray, nlags: int = 200) -> dict[str, float]:
    """自相关衰减速率。

    直觉：ACF 衰减慢 → 长程依赖强 → 长 horizon 仍有可用信号；
    衰减快（几个 lag 就掉到噪声水平）→ 长 horizon 基本靠均值/趋势，插件收益结构不同。
    返回 lag1 自相关、首次跌破 1/e 的 lag（归一化）、指数衰减常数。

    衰减常数只在「从 lag=1 起连续显著」的那一段上用 log|ACF| 线性拟合
    （显著性带 = 2/√n）；如果连显著段都不足 5 个 lag，说明一步之内就掉进噪声，
    直接取上界 1.0（越大 = 衰减越快），从而避免在白噪声上拟合纯噪声得到无意义的符号。
    复杂度 O(n log n)。
    """
    x = np.asarray(x, dtype=np.float64)
    ac = acf_fft(x, nlags)
    if not np.isfinite(ac).any():
        return {"acf1": float("nan"), "acf_e_folding": float("nan"), "acf_decay_rate": float("nan")}
    thr = 1.0 / math.e
    below = np.where(np.abs(ac[1:]) < thr)[0]
    e_fold = float((below[0] + 1) / nlags) if below.size else 1.0
    band = 2.0 / math.sqrt(max(x.size, 1))
    lags = np.arange(1, min(nlags, 50) + 1)
    mag = np.abs(ac[1 : lags.size + 1])
    sig = mag > band
    k = int(np.argmin(sig)) if not sig.all() else int(sig.size)  # 首个不显著 lag 之前的长度
    if k >= 5:
        slope = float(np.polyfit(lags[:k], np.log(mag[:k]), 1)[0])
        rate = float(np.clip(-slope, 0.0, 1.0))
    else:
        rate = 1.0
    return {"acf1": float(ac[1]), "acf_e_folding": e_fold, "acf_decay_rate": rate}


def rolling_drift(x: FloatArray, window: int) -> dict[str, float]:
    """非平稳度：滑窗均值/方差的漂移程度（不依赖任何检验假设的稳健度量）。

    直觉（这是 RevIN/SAN 的立论基础）：如果窗口间均值漂移相对整体波动很大，
    「按窗口自身统计量归一化」才有意义；若序列本来就平稳，归一化插件只会白扔信息。
    复杂度 O(n)（用累积和实现滑窗矩）。
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    w = max(4, min(window, n // 4))
    if n < 2 * w:
        return {"mean_drift": float("nan"), "var_drift": float("nan"), "range_drift": float("nan")}
    csum = np.concatenate([[0.0], np.cumsum(x)])
    csq = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(0, n - w + 1, max(1, w // 2))
    m = (csum[starts + w] - csum[starts]) / w
    v = np.maximum((csq[starts + w] - csq[starts]) / w - m**2, 0.0)
    gstd = float(x.std())
    if gstd <= 0:
        return {"mean_drift": 0.0, "var_drift": 0.0, "range_drift": 0.0}
    mean_drift = float(m.std() / gstd)
    sd = np.sqrt(v) + 1e-8
    var_drift = float(np.std(np.log(sd)) / (abs(np.mean(np.log(sd))) + 1.0))
    range_drift = float((m.max() - m.min()) / gstd)
    return {"mean_drift": mean_drift, "var_drift": var_drift, "range_drift": range_drift}


def adf_stat(x: FloatArray, maxlag: int = 12) -> dict[str, float]:
    """ADF 单位根检验统计量（固定 maxlag，因此是 O(n)）。

    直觉：统计量越接近 0（p 越大）→ 越像单位根/随机游走 → 非平稳 → 归一化插件更可能有用。
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except Exception:  # pragma: no cover - statsmodels 必装，仅防御
        return {"adf_stat": float("nan"), "adf_pvalue": float("nan")}
    x = np.asarray(x, dtype=np.float64)
    if x.size < 4 * maxlag or x.std() == 0:
        return {"adf_stat": float("nan"), "adf_pvalue": float("nan")}
    try:
        try:  # statsmodels>=0.15：显式关掉未来的 result_object 行为以静音 FutureWarning
            res = adfuller(x, maxlag=maxlag, regression="c", autolag=None, result_object=False)
        except TypeError:
            res = adfuller(x, maxlag=maxlag, regression="c", autolag=None)
        return {"adf_stat": float(res[0]), "adf_pvalue": float(res[1])}
    except Exception:
        return {"adf_stat": float("nan"), "adf_pvalue": float("nan")}


def stl_strengths(x: FloatArray, period: int) -> dict[str, float]:
    """STL 分解的趋势强度与季节强度（Wang et al. 的经典定义）。

    strength = max(0, 1 - Var(remainder) / Var(remainder + component))
    直觉：季节强度高 → 周期结构明显；趋势强度高 → 存在慢漂移（非平稳的另一面）。
    复杂度：STL 的 loess 实现约 O(n)，此处再对长序列截断到 cfg.max_len。
    """
    x = np.asarray(x, dtype=np.float64)
    period = int(max(2, period))
    if x.size < 3 * period or x.std() == 0:
        return {"trend_strength": float("nan"), "seasonal_strength": float("nan")}
    try:
        from statsmodels.tsa.seasonal import STL

        res = STL(x, period=period, robust=False).fit()
        r, s, t = res.resid, res.seasonal, res.trend
        vr = float(np.var(r))
        f_t = max(0.0, 1.0 - vr / max(float(np.var(r + t)), 1e-12))
        f_s = max(0.0, 1.0 - vr / max(float(np.var(r + s)), 1e-12))
        return {"trend_strength": float(min(f_t, 1.0)), "seasonal_strength": float(min(f_s, 1.0))}
    except Exception:
        return {"trend_strength": float("nan"), "seasonal_strength": float("nan")}


# --------------------------------------------------------------------------- #
# 多通道统计量
# --------------------------------------------------------------------------- #
def channel_spectrum(X: FloatArray) -> dict[str, float]:
    """通道相关矩阵的谱秩与能量集中度。

    直觉（对应 CCM / LIFT / CrossLinear 这类通道插件）：
    - 有效秩接近 1 → 通道高度冗余 → 通道混合/聚类类插件有大空间；
    - 有效秩接近 C → 通道近乎独立 → 通道插件几乎必然无效甚至有害。
    复杂度：相关矩阵 O(C²T)（BLAS matmul），特征值 O(C³)；C≤862 时秒级。
    """
    X = np.asarray(X, dtype=np.float64)
    c = X.shape[1]
    if c < 2:
        return {
            "eff_rank_ratio": 1.0,
            "eig_top1_ratio": 1.0,
            "eig_top5_ratio": 1.0,
            "participation_ratio": 1.0 / max(c, 1),
            "mean_abs_corr": float("nan"),
            "n_channels": float(c),
        }
    Z = (X - X.mean(0)) / (X.std(0) + 1e-8)
    corr = (Z.T @ Z) / X.shape[0]
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    eig = np.linalg.eigvalsh(corr)
    eig = np.clip(eig[::-1], 0.0, None)
    tot = eig.sum()
    if tot <= 0:
        tot = 1e-12
    p = eig / tot
    pnz = p[p > 0]
    eff_rank = float(np.exp(-(pnz * np.log(pnz)).sum()))  # 谱熵指数 = 有效秩
    off = corr[~np.eye(c, dtype=bool)]
    return {
        "eff_rank_ratio": eff_rank / c,
        "eig_top1_ratio": float(p[0]),
        "eig_top5_ratio": float(p[: min(5, c)].sum()),
        "participation_ratio": float((p.sum() ** 2) / (p**2).sum() / c),
        "mean_abs_corr": float(np.abs(off).mean()),
        "n_channels": float(c),
    }


def window_complexity(X: FloatArray, cfg: FeatureConfig = DEFAULT_CFG) -> dict[str, float]:
    """窗口级模式复杂度（Accuracy Law 视角：它与可达的最小 MSE 呈指数关系）。

    做法：随机采样 n_windows 个长度为 window 的窗口 × 采样若干通道，
    在每个 (窗口, 通道) 上算排列熵与谱熵，报告均值与跨窗口标准差。
    跨窗口标准差本身是一个有用信号：**异质性高 → 单一插件难以全局适用**。
    复杂度 O(n_windows · window · log window)，与序列长度无关。
    """
    X = np.asarray(X, dtype=np.float64)
    t, c = X.shape
    rng = np.random.default_rng(cfg.seed)
    w = min(cfg.window, t)
    ch = rng.choice(c, size=min(c, cfg.max_channels), replace=False)
    starts = (
        rng.choice(t - w + 1, size=min(cfg.n_windows, t - w + 1), replace=False)
        if t > w
        else np.array([0])
    )
    pes, ses = [], []
    for s in starts:
        seg = X[s : s + w, ch]
        pes.append(np.nanmean([permutation_entropy(seg[:, j], cfg.pe_order) for j in range(seg.shape[1])]))
        ses.append(np.nanmean([spectral_entropy(seg[:, j]) for j in range(seg.shape[1])]))
    pes_a, ses_a = np.asarray(pes, dtype=float), np.asarray(ses, dtype=float)
    return {
        "perm_entropy": float(np.nanmean(pes_a)),
        "perm_entropy_std": float(np.nanstd(pes_a)),
        "spec_entropy": float(np.nanmean(ses_a)),
        "spec_entropy_std": float(np.nanstd(ses_a)),
        "forecastability": float(1.0 - np.nanmean(ses_a)),
    }


# --------------------------------------------------------------------------- #
# 数据集级 / horizon 级特征
# --------------------------------------------------------------------------- #
def _subsample_channels(X: FloatArray, cfg: FeatureConfig) -> FloatArray:
    rng = np.random.default_rng(cfg.seed)
    c = X.shape[1]
    if c <= cfg.max_channels:
        return X
    idx = np.sort(rng.choice(c, size=cfg.max_channels, replace=False))
    return X[:, idx]


def series_features(
    X: FloatArray, period: int, cfg: FeatureConfig = DEFAULT_CFG
) -> dict[str, float]:
    """数据集级（与 horizon 无关）特征。输入 X: (T, C) 训练段、已按通道 z-score。"""
    X = np.asarray(X, dtype=np.float64)
    Xs = _subsample_channels(X, cfg)
    Xt = Xs[: cfg.max_len]

    feats: dict[str, float] = {}
    # 1) 窗口级模式复杂度
    for k, v in window_complexity(X, cfg).items():
        feats[f"pc_{k}"] = v
    # 2) 非平稳度
    drift = [rolling_drift(Xs[:, j], cfg.window) for j in range(Xs.shape[1])]
    for key in ("mean_drift", "var_drift", "range_drift"):
        feats[f"ns_{key}"] = float(np.nanmean([d[key] for d in drift]))
    adf = [adf_stat(Xt[:, j]) for j in range(Xt.shape[1])]
    feats["ns_adf_stat"] = float(np.nanmean([a["adf_stat"] for a in adf]))
    feats["ns_adf_pvalue"] = float(np.nanmean([a["adf_pvalue"] for a in adf]))
    feats["ns_adf_frac_nonstat"] = float(np.nanmean([a["adf_pvalue"] > 0.05 for a in adf]))
    # 3) 通道相关谱
    for k, v in channel_spectrum(X[: cfg.max_len]).items():
        feats[f"ch_{k}"] = v
    # 4) 频域集中度
    tk = [top_freq_energy_ratio(Xs[:, j], cfg.top_k_freq) for j in range(Xs.shape[1])]
    feats["fr_topk_energy"] = float(np.nanmean([a for a, _ in tk]))
    feats["fr_dominant_period"] = float(np.nanmedian([b for _, b in tk]))
    feats["fr_flatness"] = float(np.nanmean([spectral_flatness(Xs[:, j]) for j in range(Xs.shape[1])]))
    # 5) 趋势/季节强度 + ACF 衰减
    stl = [stl_strengths(Xt[:, j], period) for j in range(Xt.shape[1])]
    feats["sd_trend_strength"] = float(np.nanmean([s["trend_strength"] for s in stl]))
    feats["sd_seasonal_strength"] = float(np.nanmean([s["seasonal_strength"] for s in stl]))
    dec = [acf_decay(Xs[:, j], cfg.acf_nlags) for j in range(Xs.shape[1])]
    for key in ("acf1", "acf_e_folding", "acf_decay_rate"):
        feats[f"sd_{key}"] = float(np.nanmean([d[key] for d in dec]))
    feats["sd_period"] = float(period)
    feats["sd_length"] = float(X.shape[0])
    return feats


def horizon_features(
    X: FloatArray,
    seq_len: int,
    pred_len: int,
    period: int,
    cfg: FeatureConfig = DEFAULT_CFG,
) -> dict[str, float]:
    """horizon 条件特征：同一数据集在不同 pred_len 下会取到不同值。

    这组特征是「同一格换插件」的关键——例如 `hz_mean_shift` 度量的是
    「输入窗口均值 → 输出窗口均值」的漂移幅度，正是归一化类插件的作用对象；
    pred_len 越长，该漂移越大，因此插件的最优选择会随 horizon 变化。
    复杂度 O(n)（滑窗矩用累积和）+ O(n log n)（谱）。
    """
    X = np.asarray(X, dtype=np.float64)
    Xs = _subsample_channels(X, cfg)
    t = Xs.shape[0]
    feats: dict[str, float] = {"hz_pred_len": float(pred_len), "hz_ratio_seq": float(pred_len / seq_len)}

    shifts, vratios, acfh = [], [], []
    for j in range(Xs.shape[1]):
        x = Xs[:, j]
        need = seq_len + pred_len
        if t < need + 1:
            continue
        csum = np.concatenate([[0.0], np.cumsum(x)])
        csq = np.concatenate([[0.0], np.cumsum(x * x)])
        starts = np.arange(0, t - need + 1, max(1, pred_len // 2))

        def _mv(a: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray]:
            m = (csum[a + w] - csum[a]) / w
            v = np.maximum((csq[a + w] - csq[a]) / w - m**2, 0.0)
            return m, v

        m_in, v_in = _mv(starts, seq_len)
        m_out, v_out = _mv(starts + seq_len, pred_len)
        gstd = x.std() + 1e-8
        shifts.append(float(np.mean(np.abs(m_out - m_in)) / gstd))
        vratios.append(float(np.mean(np.log((v_out + 1e-6) / (v_in + 1e-6)))))
        ac = acf_fft(x, min(pred_len, max(4, t // 4)))
        acfh.append(float(ac[-1]) if np.isfinite(ac[-1]) else np.nan)

    feats["hz_mean_shift"] = float(np.nanmean(shifts)) if shifts else float("nan")
    feats["hz_logvar_ratio"] = float(np.nanmean(vratios)) if vratios else float("nan")
    feats["hz_acf_at_h"] = float(np.nanmean(acfh)) if acfh else float("nan")

    # 长程能量占比：周期 ≥ pred_len 的频率能量份额（长 horizon 上唯一可用的结构性信号）
    long_share = []
    for j in range(Xs.shape[1]):
        psd = power_spectrum(Xs[:, j][: cfg.max_len])
        if psd.size < 2:
            continue
        n = 2 * psd.size + 1
        periods = n / (np.arange(1, psd.size + 1))
        s = psd.sum()
        if s > 0:
            long_share.append(float(psd[periods >= pred_len].sum() / s))
    feats["hz_long_range_energy"] = float(np.nanmean(long_share)) if long_share else float("nan")
    feats["hz_h_over_period"] = float(pred_len / max(period, 1))
    return feats


# --------------------------------------------------------------------------- #
# 数据读取（严格复刻 TSLib 的切分规则，只取训练段）
# --------------------------------------------------------------------------- #
def load_train_split(csv_path: str | Path, dataset: str, seq_len: int) -> FloatArray:
    """读 CSV，按 TSLib 规则取**训练段**并用训练段统计量做 z-score。

    ETTh*: train = 12*30*24；ETTm*: train = 12*30*24*4；其余 custom: 前 70%。
    返回 (T_train, C) 的 float64 数组（丢掉 date 列）。
    """
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.lower() != "date"]
    data = df[cols].to_numpy(dtype=np.float64)
    if dataset in ("ETTh1", "ETTh2"):
        end = 12 * 30 * 24
    elif dataset in ("ETTm1", "ETTm2"):
        end = 12 * 30 * 24 * 4
    else:
        end = int(len(df) * 0.7)
    train = data[: min(end, len(data))]
    mu, sd = train.mean(0), train.std(0) + 1e-8
    return (train - mu) / sd


def dataset_rows(
    X: FloatArray,
    dataset: str,
    seq_len: int,
    horizons: list[int],
    period: int,
    cfg: FeatureConfig = DEFAULT_CFG,
) -> list[dict[str, Any]]:
    """对一个数据集产出 len(horizons) 行特征（数据集级特征在各行重复）。"""
    base = series_features(X, period, cfg)
    rows = []
    for h in horizons:
        row: dict[str, Any] = {"dataset": dataset, "pred_len": int(h)}
        row.update(base)
        row.update(horizon_features(X, seq_len, int(h), period, cfg))
        rows.append(row)
    return rows


def build_feature_table(
    config_path: str | Path | None = None,
    datasets: list[str] | None = None,
    include_optional: bool = False,
    cfg: FeatureConfig = DEFAULT_CFG,
) -> pd.DataFrame:
    """按 matrix.yaml 遍历本地已下载的数据集，产出 (dataset × pred_len) 特征表。"""
    from .config import REPO_ROOT, MatrixConfig

    mc = MatrixConfig.load(config_path)
    names = datasets or mc.dataset_names(include_optional=include_optional)
    rows: list[dict[str, Any]] = []
    for name in names:
        ds = mc.dataset(name)
        csv = REPO_ROOT / ds["root_path"] / ds["data_path"]
        if not csv.exists():
            print(f"[features] skip {name}: {csv} 不存在（先运行 scripts/download_data.py）")
            continue
        X = load_train_split(csv, name, mc.seq_len(name))
        print(f"[features] {name}: train shape={X.shape}")
        rows += dataset_rows(X, name, mc.seq_len(name), list(ds["horizons"]), int(ds["period"]), cfg)
    return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """特征列 = 除 key 列以外的全部数值列。"""
    keys = {"dataset", "pred_len", "backbone", "plugin", "seed"}
    return [c for c in df.columns if c not in keys and pd.api.types.is_numeric_dtype(df[c])]


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _demo() -> pd.DataFrame:
    """用三种性质迥异的合成序列自检：白噪声 / 强季节 / 随机游走。"""
    rng = np.random.default_rng(0)
    t = 4000
    tt = np.arange(t)
    noise = rng.standard_normal((t, 4))
    seasonal = np.sin(2 * np.pi * tt[:, None] / 24) + 0.1 * rng.standard_normal((t, 4))
    walk = np.cumsum(rng.standard_normal((t, 4)), axis=0)
    rows = []
    for name, X in [("synth_noise", noise), ("synth_seasonal", seasonal), ("synth_walk", walk)]:
        Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)
        rows += dataset_rows(Xz, name, 96, [96, 336], 24)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="计算复杂度/可预测性特征表")
    ap.add_argument("--config", default=None, help="configs/matrix.yaml 路径")
    ap.add_argument("--out", default=None, help="输出 CSV 路径")
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--include-optional", action="store_true")
    ap.add_argument("--demo", action="store_true", help="用合成序列自检，不读真实数据")
    args = ap.parse_args()

    df = _demo() if args.demo else build_feature_table(args.config, args.datasets, args.include_optional)
    if df.empty:
        print("[features] 没有产出任何行（数据是否已下载？）")
        return
    with pd.option_context("display.width", 200, "display.max_columns", 100):
        cols = ["dataset", "pred_len", "pc_perm_entropy", "pc_forecastability", "ns_mean_drift",
                "ns_adf_frac_nonstat", "ch_eff_rank_ratio", "fr_topk_energy",
                "sd_seasonal_strength", "sd_trend_strength", "hz_mean_shift"]
        shown = df[[c for c in cols if c in df.columns]]
        print(shown.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"[features] 写出 {args.out}  shape={df.shape}")


if __name__ == "__main__":
    main()
