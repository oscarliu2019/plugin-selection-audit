#!/usr/bin/env python3
"""把 oracle 审计的逐组表画成论文主图。

两张图，都只吃 `artifacts/p2/selector_window_oracle_audit.csv`（零 GPU，秒级）：

* `fig_halflife_vs_horizon.{png,pdf}` —— **论文主图**。
  左：最优臂身份的相关半衰期（窗口数） vs 预测视界 H，双对数。
  参考线 y=H 是「相关长度刚好够用」的临界，y=H/10 作为量级参照；
  「有多少组落在 y=H/10 以下」由 `frac_below_H_over_10` 现算，**分母只含半衰期
  有定义的组**（半衰期在 excess_abs1<=0 或衰减曲线被截断时是 NaN，见修复项 #5），
  未定义组数一并返回，正文必须交代。
  右：同一批点的比值 半衰期/H vs H —— 半衰期的绝对值随 H 的增长远追不上 H 本身，
  所以比值单调塌陷。这张图一眼说明「H 越长越无望」。

* `fig_decay_curves.{png,pdf}` —— 分层衰减曲线。
  三个面板分别按 backbone / dataset / horizon 分层，
  纵轴是滞后一致率**超出 chance** 的量（0 = 与独立重抽无异），横轴是绝对滞后（对数）。
  竖虚线标出各层的 H，用来直观展示「曲线早在 H 之前就压到 0」。
  只画**全组都有支撑**的滞后点（长滞后处短序列组是 NaN，混着画等于换了组集合，
  见修复项 #6），支撑不全的滞后断线，每个滞后的有效组数随返回值一起给出。

设计约束：
* 图里全部用英文标签（论文是英文的），中文只出现在本文件的注释与终端输出里；
* 不用 seaborn、不设自定义颜色循环之外的花哨样式，保证在只有 matplotlib 的远端也能跑；
* 所有数值都从 CSV 现算，不接受手输参数，避免图与表不一致。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402
import pandas as pd                       # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.selector_window import (LAGS_ABS, stratify_audit,  # noqa: E402
                                 strata_verdict, summarize_strata)

MARKERS = ("o", "s", "^", "v", "D", "P", "X", "*")


def jitter_factor(bi: int, di: int, n_datasets: int, n_backbones: int = 4,
                  amp: float = 0.055) -> float:
    """把 (backbone, dataset) 下标映射成 x 轴上的一点抖动，避免同一 H 上的点完全重叠。

    两个约束：
    1. 必须**只依赖下标**：早期版本写的是 ``hash(f"{b}{d}")``，而 CPython 的 str hash 受
       PYTHONHASHSEED 随机化影响，同一份数据每次画出来的图都不一样——图表就不可复现了，
       审稿人对着两版图会看到点在动。
    2. 每个 (backbone, dataset) 组合都要拿到**互不相同**的偏移。早期版本用 ``% 7``，
       于是 4×8=32 个组合只有 7 个位置，H=96 那一列上 4 个点仍然精确重叠、看不出有几个。
       这里把组合下标线性铺满 [-amp, +amp]。
    """
    total = max(int(n_backbones) * int(n_datasets), 1)
    k = bi * int(n_datasets) + di
    frac = (k / (total - 1) - 0.5) * 2.0 if total > 1 else 0.0   # -1 .. +1
    return 1.0 + amp * frac


def fig_halflife_vs_horizon(df: pd.DataFrame, out: Path, dpi: int = 200) -> dict:
    # 2026-09-14 code review 修复项 #5：half_life_over_H / persist_half_life_windows 可以是
    # NaN（src/selector_window.py 的 evaluate_argmin_structure 在 excess_abs1 <= 0 或衰减
    # 曲线被截断时把 persist_half_life_windows 置 NaN）。这些「未定义」的组画不出点，
    # 也不该进入任何比例的分母。这里先把「有定义」的子集固定下来，后面所有统计都用它。
    defined = df["half_life_over_H"].notna() & df["persist_half_life_windows"].notna()
    dfd = df[defined]
    n_defined, n_undefined = int(defined.sum()), int((~defined).sum())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    backbones = sorted(df["backbone"].unique())
    datasets = sorted(df["dataset"].unique())
    cmap = plt.get_cmap("tab10")
    colors = {b: cmap(i % 10) for i, b in enumerate(backbones)}
    marks = {d: MARKERS[i % len(MARKERS)] for i, d in enumerate(datasets)}

    ax = axes[0]
    for bi, b in enumerate(backbones):
        for di, d in enumerate(datasets):
            # 只画半衰期**有定义**的组（修复项 #5）：NaN 点本来就画不出来，
            # 但以前它们仍被算进 n_points 与 frac_below_H_over_10 的分母。
            sub = dfd[(dfd["backbone"] == b) & (dfd["dataset"] == d)]
            if sub.empty:
                continue
            # 同一 (backbone,dataset) 的多个 horizon 在 x 上会重叠，加一点抖动便于看清。
            # 抖动量必须**由下标确定**：早期版本用了 hash(str)，而 CPython 的 str hash 带
            # PYTHONHASHSEED 随机化，同一份数据每次画出来的图都不一样，图表就不可复现了。
            jit = jitter_factor(bi, di, len(datasets), len(backbones))
            ax.scatter(sub["pred_len"] * jit, sub["persist_half_life_windows"],
                       s=30, color=colors[b], marker=marks[d], alpha=0.85,
                       edgecolors="none")
    hs = np.array(sorted(df["pred_len"].unique()), dtype=float)
    l1, = ax.plot(hs, hs, "k-", lw=1.3, label="half-life = H (break-even)")
    l2, = ax.plot(hs, hs / 10.0, "k--", lw=1.0, label="half-life = H/10")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _horizon_ticks(ax, hs)
    ax.set_xlabel("Forecast horizon H (windows)")
    ax.set_ylabel("Half-life of best-arm identity (windows)")
    ax.set_title("(a) Usable correlation length vs horizon")
    ax.grid(alpha=0.25, which="both")
    # 图例：颜色=骨干，形状=数据集。没有它们这张图不可读；但放在轴内会压住
    # ILI 的 H=24~60 那批点（实测过），所以统一放到整张图下方一行。
    ax.legend(handles=[l1, l2], fontsize=7.5, loc="upper left")
    h_bb = [plt.Line2D([], [], marker="o", ls="", color=colors[b], label=b)
            for b in backbones]
    h_ds = [plt.Line2D([], [], marker=marks[d], ls="", color="0.35", label=d)
            for d in datasets]

    ax = axes[1]
    # 同修复项 #5：中位数与散点都只用「半衰期有定义」的组
    g = dfd.groupby("pred_len")["half_life_over_H"]
    med = g.median().dropna()
    ax.scatter(dfd["pred_len"], dfd["half_life_over_H"], s=18, color="0.6",
               alpha=0.6, edgecolors="none", label="individual decision groups")
    ax.plot(med.index, med.values, "o-", color="crimson", lw=1.8,
            label="median per horizon")
    ax.axhline(1.0, color="k", lw=1.2)
    if len(med):
        ax.text(float(med.index.min()), 1.06, "half-life = H", fontsize=8, va="bottom")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _horizon_ticks(ax, hs)
    ax.set_xlabel("Forecast horizon H (windows)")
    ax.set_ylabel("Half-life / H")
    ax.set_title("(b) The ratio collapses as H grows")
    ax.legend(fontsize=7.5, loc="lower left")
    ax.grid(alpha=0.25, which="both")

    fig.legend(handles=h_bb + h_ds, loc="lower center", ncol=6, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01),
               title="color = backbone   ·   marker = dataset", title_fontsize=7.5)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    # 修复项 #5：
    # * n_points = **实际画出的点数**（NaN 组画不出点，不能算进去）；
    # * frac_below_H_over_10 只在 half_life_over_H 非 NaN 的组上算——
    #   旧写法 `(df["half_life_over_H"] < 0.1).mean()` 把 NaN 当 False 计入分母，
    #   于是这个比例被「未定义组」稀释，而 main() 把它打成「X% 的组半衰期 < H/10」；
    # * 有定义 / 未定义组数一起返回，供报告如实交代样本口径。
    return {"n_points": n_defined,
            "n_groups_total": int(len(df)),
            "n_groups_half_life_defined": n_defined,
            "n_groups_half_life_undefined": n_undefined,
            "median_ratio_by_H": {str(k): round(float(v), 4) for k, v in med.items()},
            "frac_below_H_over_10": (float((dfd["half_life_over_H"] < 0.1).mean())
                                     if n_defined else float("nan"))}


def _horizon_ticks(ax, hs: np.ndarray) -> None:
    """对数轴默认只标 10^2，会把 ILI 的 24/36/48/60 和主矩阵的 96~720 全糊在一起。"""
    ax.set_xticks(list(hs))
    ax.set_xticklabels([f"{int(h)}" for h in hs], fontsize=7.5)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def fig_decay_curves(df: pd.DataFrame, out: Path, dpi: int = 200) -> dict:
    """分层衰减曲线。**每条曲线只画「全组都有支撑」的滞后点。**

    2026-09-14 code review 修复项 #6：``excess_abs{L}`` 在 ``L >= n_test_windows`` 时是
    NaN（见 src/selector_window.py），而 ``sub[cols].mean()`` 按列跳过 NaN。于是同一条
    曲线上不同滞后点落在**不同组子集**上：lag 1~64 用全部组，lag 512/1024 只剩测试窗口
    足够长的组（ILI 这类短序列在长滞后处全 NaN），图例却仍标 ``n=len(sub)``。
    曲线尾部「压到 0」可能只是换了组集合造成的——而这恰恰是本图要论证的核心结论，
    也是 main() 打印的「最大滞后超出量」的来源。

    所以这里先统计每个滞后的非 NaN 组数，支撑不全的滞后置为 NaN
    （matplotlib 的 plot 会自然断线），并把每个滞后的有效组数一并返回。
    """
    lags = np.array(LAGS_ABS, dtype=float)
    cols = [f"excess_abs{L}" for L in LAGS_ABS]
    cuts = [("backbone", "(a) by backbone"), ("dataset", "(b) by dataset"),
            ("pred_len", "(c) by horizon H")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    info: dict = {}
    for ax, (key, title) in zip(axes, cuts):
        cmap = plt.get_cmap("tab10" if key != "dataset" else "tab20")
        info[key] = {}
        for i, (val, sub) in enumerate(df.groupby(key, sort=True)):
            n = int(len(sub))
            n_valid = sub[cols].notna().sum().to_numpy().astype(int)   # 每个滞后的有效组数
            full = n_valid == n                                        # 全组都有支撑
            y = np.where(full, sub[cols].mean().to_numpy(), np.nan)
            full_lags = lags[full]
            lab = (f"H={val}" if key == "pred_len" else f"{val}")
            lab += f" (n={n}, full support ≤{int(full_lags.max())})" if full_lags.size \
                else f" (n={n}, 无全支撑滞后)"
            ax.plot(lags, y, "o-", ms=3, lw=1.4, color=cmap(i % cmap.N), label=lab)
            if key == "pred_len":
                # 竖线标出这一层的 H：曲线在到达它之前就已经压到 0
                ax.axvline(float(val), color=cmap(i % cmap.N), ls=":", lw=0.9, alpha=0.7)
            # 最大**全支撑**滞后处的超出量：以前报的是最大滞后处那个被稀释的均值
            last = int(full_lags.max()) if full_lags.size else None
            info[key][str(val)] = {
                "n_groups": n,
                "last_full_support_lag": last,
                "excess_at_last_full_support_lag": (
                    round(float(y[full][-1]), 4) if full_lags.size else None),
                "n_valid_by_lag": {str(L): int(c) for L, c in zip(LAGS_ABS, n_valid)},
            }
        ax.axhline(0.0, color="k", lw=1.2)
        ax.set_xscale("log")
        ax.set_xlabel("Absolute lag (windows)")
        ax.set_title(title)
        if key == "pred_len":
            # 没有这一句，(c) 里那 8 条竖虚线就是无法解释的装饰；它们恰恰是本图的论点：
            # 每条曲线在到达同色竖线（= 该层唯一合法的决策延迟 H）之前就已经压到 0。
            ax.plot([], [], color="0.35", ls=":", lw=0.9,
                    label="dotted = legal delay H (same color)")
        ax.legend(fontsize=7, ncol=1)
        ax.grid(alpha=0.25, which="both")
    axes[0].set_ylabel("Lag agreement of best arm, excess over chance")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-csv", default="artifacts/p2/selector_window_oracle_audit.csv")
    ap.add_argument("--out-dir", default="artifacts/paper")
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    csv = Path(a.audit_csv)
    if not csv.exists():
        print(f"[figs] 缺少 {csv}；先跑 "
              f"`python -m src.selector_window --protocol inpool --oracle-audit`")
        return 1
    df = pd.read_csv(csv)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m1 = fig_halflife_vs_horizon(df, out_dir / "fig_halflife_vs_horizon", a.dpi)
    m2 = fig_decay_curves(df, out_dir / "fig_decay_curves", a.dpi)
    strata = stratify_audit(df)
    strata.to_csv(out_dir / "audit_strata.csv", index=False)
    s = summarize_strata(strata)
    s["verdict"] = strata_verdict(s)

    print(f"[figs] {len(df)} 个决策组 -> {out_dir}")
    # 修复项 #5：比例的分母是「半衰期有定义的组」，打印时必须把口径和未定义组数说清楚，
    # 否则读者（和照抄进论文的作者）会以为它是全部组的比例。
    frac = m1["frac_below_H_over_10"]
    print(f"[figs] fig_halflife_vs_horizon: 半衰期有定义的组 "
          f"{m1['n_groups_half_life_defined']}/{m1['n_groups_total']}"
          f"（未定义 {m1['n_groups_half_life_undefined']} 组，excess_abs1<=0 或曲线被截断，"
          f"不计入下面的分母）；其中 "
          + (f"{frac:.0%}" if frac == frac else "—")
          + " 的组半衰期 < H/10")
    print(f"[figs] 各 horizon 的半衰期/H 中位数（仅有定义的组）: {m1['median_ratio_by_H']}")
    # 修复项 #6：报最大**全支撑**滞后处的超出量，并带上该滞后与有效组数，
    # 不再报「不同滞后落在不同组子集上」的那个被稀释的均值。
    print("[figs] fig_decay_curves 各骨干在最大全支撑滞后处的超出量: "
          + ", ".join(f"{k}: lag<={v['last_full_support_lag']} -> "
                      f"{v['excess_at_last_full_support_lag']} (n={v['n_groups']})"
                      for k, v in m2["backbone"].items()))
    print("[figs] " + s["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
