"""⭐ 逐测试窗口特征：把插件选择的决策粒度从「数据集」下沉到「单个输入窗口」。

为什么必须有这个模块
--------------------
phase-1 的 selector 是失败的（LODO 准确率 0.381 < 多数类基线 0.405，所有特征
permutation importance ≈ 0）。根因**不是模型选错**，而是决策粒度错了：
`artifacts/features.csv` 是 32 行，但同一数据集的 4 个 horizon 特征完全相同，
全表只有 **8 个互异特征向量**。留一数据集交叉验证下，等价于「7 个训练样本预测第 8 个」，
任何模型都学不出东西。

把粒度下沉到单个测试窗口后，N 从 42 变成万级，而且特征真正随样本变化。
这同时把研究问题升级了：从"给这个数据集推荐一个插件"（弱、且实践价值低）
变成"给这个即将到来的窗口决定该不该挂、挂哪个"（instance-adaptive gating）。

无泄漏契约（审稿人必查）
------------------------
1. 特征只从**输入窗口** `X[i : i+seq_len]` 计算。推理时这段历史是已知的，
   因此使用它不构成泄漏——这与"用测试集标签调参"有本质区别。
2. z-score 只用**训练段**统计量（与 TSLib 的 StandardScaler 一致），
   绝不使用测试段的均值/方差。
3. 窗口切分与 TSLib `Dataset_*` 的 border 规则严格一致，第 i 行对应 `window_index == i`
   （即 `X[i : i+seq_len]`），下游 `selector_window.build_groups` **按 window_index 取值**
   对齐，不按行位置。
   ⚠️ 注意 `test` 段 TSLib 本身 `shuffle=False`，但 **`val` 段默认 `shuffle=True`**
   （`data_factory.py:26` 只判断 test），因此 `winerr/val/*.npy` 必须由 `runner` 用显式
   确定序 loader 落盘并打 `.order.json` 标记，否则顺序是随机置换。
   这是 2026-09-14 code review 发现的 P0，详见 `internal/docs/code_review_20260914.md`。
4. 同一数据集不同 horizon 的第 i 个输入窗口是**同一段历史**，
   所以特征只按 (dataset, seq_len) 算一次，按 horizon 截断即可（省 4 倍算力）。

用法
----
    python -m src.features_window --config configs/matrix_p2.yaml            # 全部数据集
    python -m src.features_window --datasets ETTh1 Exchange --out /tmp/f.parquet
    python -m src.features_window --demo                                     # 合成数据自检
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import REPO_ROOT, MatrixConfig
from .features import permutation_entropy, spectral_entropy, top_freq_energy_ratio

FloatArray = np.ndarray


# --------------------------------------------------------------------------- #
# 预算参数
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class WindowFeatureConfig:
    """逐窗口特征的预算参数。全部固定，不逐数据集调。"""

    max_channels: int = 8      # 每个窗口采样多少通道做逐通道统计（Electricity 321 通道时生效）
    n_halves: int = 2          # 窗口内漂移用前/后半段比较
    top_k_freq: int = 3        # 主频能量集中度取 top-k
    acf_lags: tuple[int, ...] = (1, 2, 4, 8, 24)   # 只取少数关键滞后，保持 O(L log L)
    seed: int = 26             # 通道采样种子


DEFAULT_WCFG = WindowFeatureConfig()

# 特征分组前缀 -> 对应的插件机理（写论文时按组做消融，比逐特征更有解释力）
FEATURE_GROUPS: dict[str, str] = {
    "w_pc": "窗口内模式复杂度（还有没有可提升空间）",
    "w_ns": "窗口内非平稳/分布漂移（RevIN / SAN 该不该挂）",
    "w_fr": "频域能量集中度（FreDF 这类频域损失该不该挂）",
    "w_sd": "趋势/自相关结构（通用可预测性）",
    "w_ch": "跨通道相关结构（通道类插件该不该挂）",
}


# --------------------------------------------------------------------------- #
# TSLib 的数据切分（必须与 data_provider 完全一致，否则下标对不上）
# --------------------------------------------------------------------------- #
def tslib_borders(dataset: str, n_rows: int, seq_len: int) -> dict[str, tuple[int, int]]:
    """复刻 TSLib `Dataset_ETT_hour/minute/Custom` 的 border1s/border2s。

    返回 {'train': (b1,b2), 'val': (b1,b2), 'test': (b1,b2)}，左闭右开，
    其中 b1 已经减去 seq_len（TSLib 的做法：让每段的第一个窗口能取到完整 look-back）。
    """
    if dataset in ("ETTh1", "ETTh2"):
        n_train, n_val, n_test = 12 * 30 * 24, 4 * 30 * 24, 4 * 30 * 24
    elif dataset in ("ETTm1", "ETTm2"):
        n_train, n_val, n_test = 12 * 30 * 24 * 4, 4 * 30 * 24 * 4, 4 * 30 * 24 * 4
    else:  # Dataset_Custom：7 / 2 / 1 切分
        n_train = int(n_rows * 0.7)
        n_test = int(n_rows * 0.2)
        n_val = n_rows - n_train - n_test
    b1 = [0, n_train - seq_len, n_train + n_val - seq_len]
    b2 = [n_train, n_train + n_val, n_train + n_val + n_test]
    return {"train": (b1[0], b2[0]), "val": (b1[1], b2[1]), "test": (b1[2], b2[2])}


def load_split_scaled(
    csv_path: str | Path, dataset: str, seq_len: int, split: str = "test"
) -> tuple[FloatArray, FloatArray]:
    """读 CSV，按 TSLib 规则切出指定段，并用**训练段**统计量做 z-score。

    返回 (scaled_split, train_mu_sd)；后者用于把"窗口均值离训练均值多远"这类
    分布漂移特征表达成无量纲量。
    """
    df = pd.read_csv(csv_path)
    cols = [c for c in df.columns if c.lower() != "date"]
    data = df[cols].to_numpy(dtype=np.float64)
    bd = tslib_borders(dataset, len(data), seq_len)

    tr_lo, tr_hi = bd["train"]
    train = data[tr_lo:tr_hi]
    mu, sd = train.mean(0), train.std(0) + 1e-8   # 只用训练段统计量，杜绝泄漏

    lo, hi = bd[split]
    lo, hi = max(lo, 0), min(hi, len(data))
    return (data[lo:hi] - mu) / sd, np.stack([mu, sd])


def n_windows(split_len: int, seq_len: int, pred_len: int) -> int:
    """TSLib `__len__`：len(split) - seq_len - pred_len + 1。"""
    return max(0, split_len - seq_len - pred_len + 1)


# --------------------------------------------------------------------------- #
# 单窗口特征
# --------------------------------------------------------------------------- #
# 数据已按训练段 z-score，尺度为 O(1)，故用统一的相对下限判定「恒定」。
# 绝对 epsilon（如 1e-8）在做比值时会放大出 1e8 量级的假离群值。
_EPS_REL = 1e-6


def _nanstd(v: Any) -> float:
    """全 NaN / 单点时返回 NaN 而不抛 RuntimeWarning。"""
    a = np.asarray(v, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.std(a)) if a.size else float("nan")


def _nanmean(v: Any) -> float:
    """全 NaN 时返回 NaN 而不抛 RuntimeWarning。"""
    a = np.asarray(v, dtype=np.float64)
    if a.size == 0 or not np.isfinite(a).any():
        return float("nan")
    return float(np.nanmean(a[np.isfinite(a)]))


def _linear_trend_r2(x: FloatArray) -> tuple[float, float]:
    """对窗口做一次线性拟合，返回 (斜率绝对值, R²)。R² 高 = 趋势主导。"""
    n = x.size
    t = np.arange(n, dtype=np.float64)
    t = (t - t.mean()) / (t.std() + 1e-12)
    xc = x - x.mean()
    denom = float(xc @ xc)
    # 恒定窗口（零值段 / 传感器卡死）没有「趋势」这个量。返回 0.0 会与
    # 「真的无趋势」不可区分；如实返回 NaN，下游 HistGradientBoosting 原生支持缺失值。
    if denom < 1e-18:
        return float("nan"), float("nan")
    slope = float(t @ xc) / n
    r2 = float((t @ xc) ** 2 / (n * denom))
    return abs(slope), min(max(r2, 0.0), 1.0)


def _acf_at(x: FloatArray, lags: tuple[int, ...]) -> dict[str, float]:
    """滞后自相关。恒定窗口返回 NaN（而非 0），否则与白噪声不可区分。"""
    xc = x - x.mean()
    denom = float(xc @ xc)
    if denom < 1e-18:
        return {f"acf{k}": float("nan") for k in lags}
    out = {}
    for k in lags:
        out[f"acf{k}"] = float(xc[k:] @ xc[:-k] / denom) if 0 < k < x.size else float("nan")
    return out


def window_features(
    W: FloatArray, wcfg: WindowFeatureConfig = DEFAULT_WCFG, ch_idx: FloatArray | None = None
) -> dict[str, float]:
    """单个输入窗口 (seq_len, C) 的特征。全部 O(L log L)，不含任何跨窗口信息。"""
    W = np.asarray(W, dtype=np.float64)
    L, C = W.shape
    if ch_idx is None:
        ch_idx = np.arange(min(C, wcfg.max_channels))
    sub = W[:, ch_idx]

    pe, se, tk, fl, sl, r2 = [], [], [], [], [], []
    acc: dict[str, list[float]] = {f"acf{k}": [] for k in wcfg.acf_lags}
    for j in range(sub.shape[1]):
        col = sub[:, j]
        pe.append(permutation_entropy(col, 3))
        se.append(spectral_entropy(col))
        t_k, flat = top_freq_energy_ratio(col, wcfg.top_k_freq)
        tk.append(t_k)
        fl.append(flat)
        s, r = _linear_trend_r2(col)
        sl.append(s)
        r2.append(r)
        for k, v in _acf_at(col, wcfg.acf_lags).items():
            acc[k].append(v)

    half = L // 2
    m1, m2 = sub[:half].mean(0), sub[half:].mean(0)
    s1, s2 = sub[:half].std(0), sub[half:].std(0)
    win_mu, win_sd = sub.mean(0), sub.std(0)

    feats: dict[str, float] = {
        # --- 模式复杂度：Accuracy Law 视角的「还有多少可压缩空间」 ---
        "w_pc_perm_entropy": _nanmean(pe),
        "w_pc_perm_entropy_ch_std": _nanstd(pe),
        "w_pc_spec_entropy": _nanmean(se),
        "w_pc_forecastability": 1.0 - _nanmean(se),
        # --- 非平稳 / 分布漂移：归一化类插件的作用机理 ---
        "w_ns_mean_shift": _nanmean(np.abs(m2 - m1)),
        # 有界对比度而非裸比值：s2/s1 在 s1→0（前半段恒定，实测约 1% 的窗口）时
        # 会爆到 1e6~1e8，主导任何分裂阈值 / 标准化，使这一维退化成
        # 「有没有恒定半段」的指示器。(s2-s1)/(s2+s1) 与比值同单调但落在 [-1, 1]：
        # 0 = 前后半段等方差，+1 = 前半段恒定，-1 = 后半段恒定。
        "w_ns_std_contrast": _nanmean((s2 - s1) / (s2 + s1 + _EPS_REL)),
        "w_ns_dist_from_train": _nanmean(np.abs(win_mu)),   # 训练段 z-score 后，0 = 与训练同均值
        "w_ns_scale_vs_train": _nanmean(win_sd),            # 1 = 与训练同尺度
        "w_ns_max_abs": float(np.nanmax(np.abs(sub))) if sub.size else 0.0,
        # --- 频域集中度：FreDF 的作用机理 ---
        "w_fr_topk_energy": _nanmean(tk),
        # top_freq_energy_ratio 的第 2 个返回值是**主频周期**（单位：采样点），不是谱平坦度。
        # 旧名 w_fr_flatness 与 src/features.py 的 fr_flatness（真正的 spectral_flatness）
        # 同名不同义，会让 w_fr 组的消融解释反向。2026-09-14 code review 改名。
        "w_fr_dom_period": _nanmean(fl),
        "w_fr_topk_ch_std": _nanstd(tk),
        # --- 趋势 / 自相关结构 ---
        "w_sd_trend_r2": _nanmean(r2),
        "w_sd_slope_abs": _nanmean(sl),
        **{f"w_sd_{k}": _nanmean(v) for k, v in acc.items()},
    }
    # --- 跨通道相关结构：通道类插件的作用机理（单通道时置 0） ---
    # 恒定（死）通道必须先剔除：它们会让 corrcoef 产生 NaN 行列（被 nanmean 静默丢掉，
    # 导致分母随窗口漂移、跨窗口不可比），并给 eff_rank 贡献 0 奇异值，
    # 被误读成「通道高度冗余 → 通道类插件有效」，而真实原因只是通道没有信号。
    alive = np.asarray(win_sd) > _EPS_REL
    n_alive = int(alive.sum())
    feats["w_ch_n_alive"] = float(n_alive)
    feats["w_ch_alive_frac"] = float(n_alive / sub.shape[1]) if sub.shape[1] else float("nan")
    if n_alive > 1:
        Za = sub[:, alive]
        Za = (Za - Za.mean(0)) / Za.std(0)
        R = np.corrcoef(Za.T)
        off = R[~np.eye(R.shape[0], dtype=bool)]
        ev = np.linalg.svd(Za / np.sqrt(max(L - 1, 1)), compute_uv=False) ** 2
        p = ev / ev.sum()
        feats["w_ch_corr_abs_mean"] = float(np.nanmean(np.abs(off)))
        feats["w_ch_corr_mean"] = float(np.nanmean(off))
        # 谱熵形式的「有效通道数」：越小说明通道越冗余，通道类插件越可能有效
        feats["w_ch_eff_rank"] = float(np.exp(-(p * np.log(p + 1e-18)).sum()))
    else:
        # 单通道或全部恒定：跨通道结构未定义，如实报 NaN 而不是编造 0 / 1
        feats["w_ch_corr_abs_mean"] = float("nan")
        feats["w_ch_corr_mean"] = float("nan")
        feats["w_ch_eff_rank"] = float("nan")
    return feats


# --------------------------------------------------------------------------- #
# 批量：一个数据集的全部测试窗口
# --------------------------------------------------------------------------- #
def dataset_window_features(
    csv_path: str | Path,
    dataset: str,
    seq_len: int,
    max_pred_len: int,
    wcfg: WindowFeatureConfig = DEFAULT_WCFG,
    stride: int = 1,
    verbose: bool = False,
    split: str = "test",
) -> pd.DataFrame:
    """算一个数据集**测试段**所有输入窗口的特征。

    `max_pred_len` 只用来决定要算多少个窗口：最小的 pred_len 对应最多的窗口数，
    而窗口 i 的输入与 pred_len 无关，所以按最小 pred_len 算一次、其余 horizon 截断即可。
    这里传入的应当是该数据集**最小**的 pred_len。
    """
    test, _ = load_split_scaled(csv_path, dataset, seq_len, split)
    n = n_windows(len(test), seq_len, max_pred_len)
    if n <= 0:
        raise ValueError(f"{dataset}/{split}: 窗口数为 0（len={len(test)}, seq={seq_len}, pred={max_pred_len}）")

    rng = np.random.default_rng(wcfg.seed)
    C = test.shape[1]
    ch_idx = (
        np.arange(C) if C <= wcfg.max_channels
        else np.sort(rng.choice(C, size=wcfg.max_channels, replace=False))
    )

    idx = np.arange(0, n, stride)
    rows, t0 = [], time.time()
    for k, i in enumerate(idx):
        f = window_features(test[i : i + seq_len], wcfg, ch_idx)
        f["window_index"] = int(i)
        rows.append(f)
        if verbose and (k + 1) % 2000 == 0:
            print(f"  [{dataset}] {k+1}/{len(idx)} ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.insert(0, "dataset", dataset)
    df.insert(1, "split", split)
    df["n_channels"] = C
    df["n_windows_total"] = n
    return df


def dataset_csv_path(cfg: MatrixConfig, name: str) -> Path:
    """数据集 CSV 的绝对路径。

    注意 yaml 里 ``root_path`` 已经含 ``data/`` 前缀（与 TSLib ``--root_path`` 语义一致，
    见 runner.py 的 ``root_path=REPO_ROOT/ds['root_path']``），因此**不能**再拼一次
    ``paths.data_root``，否则会得到 ``data/data/ETT-small``。
    """
    d = cfg.dataset(name)
    return (REPO_ROOT / d["root_path"] / d["data_path"]).resolve()


def build_window_feature_table(
    cfg: MatrixConfig,
    datasets: list[str] | None = None,
    stride: int = 1,
    verbose: bool = True,
    splits: tuple[str, ...] = ("val", "test"),
) -> pd.DataFrame:
    """遍历配置里的数据集 × 数据段，产出逐窗口特征长表。

    默认同时算 val 与 test：门控器必须**在验证窗口上训练、在测试窗口上评估**，
    否则就是用测试标签选插件（见 runner 里 winerr/val 的说明）。
    """
    names = datasets or cfg.dataset_names()
    out = []
    for name in names:
        d = cfg.dataset(name)
        csv = dataset_csv_path(cfg, name)
        seq_len = cfg.seq_len(name)          # ILI 是 36，不能用全局 96
        pmin = min(int(h) for h in d["horizons"])
        if verbose:
            print(f"[wfeat] {name}: seq_len={seq_len} min_pred={pmin} file={csv.name}", flush=True)
        for sp in splits:
            out.append(
                dataset_window_features(csv, name, seq_len, pmin, stride=stride,
                                        verbose=verbose, split=sp)
            )
    return pd.concat(out, ignore_index=True)


def window_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("w_")]


# --------------------------------------------------------------------------- #
# 自检 / CLI
# --------------------------------------------------------------------------- #
def _demo() -> pd.DataFrame:
    """合成三段性质截然不同的序列，检查特征能把它们区分开（无需真实数据）。"""
    rng = np.random.default_rng(0)
    L, C = 96, 4
    pure_sine = np.stack([np.sin(np.arange(L) * 2 * np.pi / 24 + p) for p in range(C)], 1)
    white = rng.standard_normal((L, C))
    trend = np.tile(np.linspace(0, 5, L)[:, None], (1, C)) + 0.05 * rng.standard_normal((L, C))
    rows = []
    for nm, W in [("sine", pure_sine), ("white", white), ("trend", trend)]:
        f = window_features(W)
        f["kind"] = nm
        rows.append(f)
    df = pd.DataFrame(rows).set_index("kind")
    keep = ["w_pc_perm_entropy", "w_fr_topk_energy", "w_sd_trend_r2", "w_sd_acf1", "w_ch_eff_rank"]
    print(df[keep].round(4).to_string())
    # 正弦的频域集中度必须高于白噪声；趋势的 R² 必须最高；白噪声排列熵最高
    assert df.loc["sine", "w_fr_topk_energy"] > df.loc["white", "w_fr_topk_energy"]
    assert df.loc["trend", "w_sd_trend_r2"] > df.loc["sine", "w_sd_trend_r2"]
    assert df.loc["white", "w_pc_perm_entropy"] > df.loc["sine", "w_pc_perm_entropy"]
    print("[demo] 区分性断言全部通过")
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="逐测试窗口特征提取（纯 CPU，可与 GPU 训练并行）")
    ap.add_argument("--config", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--stride", type=int, default=1, help=">1 时对窗口下采样（快速试跑用）")
    ap.add_argument("--out", default=None, help="默认写到 <artifacts_dir>/window_features.parquet")
    ap.add_argument("--splits", nargs="*", default=["val", "test"],
                    help="要算特征的数据段；门控训练需要 val，评估需要 test")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        _demo()
        return 0

    cfg = MatrixConfig.load(args.config)
    t0 = time.time()
    df = build_window_feature_table(cfg, args.datasets, stride=args.stride,
                                    splits=tuple(args.splits))
    out = Path(args.out) if args.out else cfg.path("artifacts_dir") / "window_features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    if out.suffix == ".parquet":
        df.to_parquet(tmp, index=False)
    else:
        df.to_csv(tmp, index=False)
    tmp.replace(out)
    print(f"[wfeat] {len(df)} 行 × {len(window_feature_columns(df))} 特征 -> {out} "
          f"({time.time()-t0:.0f}s)")
    print(df.groupby(["dataset", "split"]).size().to_string())
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
