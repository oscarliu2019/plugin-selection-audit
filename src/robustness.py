"""投稿前增量审计：指标稳健性、随机种子噪声、lead-time 剖面、成本收益。

为什么单独开一个模块而不是塞进 ``stats.py``
--------------------------------------------
``stats.py`` 负责的是论文主表那一套「块 × 方法」的假设检验，口径已经被
2026-09-14 的 code review 钉死（逐对 dropna / Holm / n_eff 效应量），不应该再往里
加分析维度。本模块做的是**围绕同一批产物的稳健性审计**：结论换个指标还成立吗？
所谓「增益」有没有大到超过随机种子噪声？最优臂在预测步长内部是否稳定？插件多花的
算力换回了什么？这些都复用 ``stats.py`` 的原语（``build_block_matrix`` /
``wilcoxon_pair`` / ``holm``），**绝不另起一套统计口径**——否则论文里会出现两个
互相解释不清的 p 值体系。

四项审计与它们各自回答的审稿意见
--------------------------------
1. ``metric_robustness``：主结论全部建立在 MSE 上。审稿人必问「换 MAE 呢」。
   本项目 844 格的 MAE 是和 MSE 同一次前向里累计出来的（``StreamingMetrics``），
   所以这项审计**零额外算力**。除了逐指标复算检验，还报出跨指标的
   逐块增益 Spearman 相关与符号一致率——指标之间不一致本身就是一个结论。
   同一张表里还带上 ``tail_stats``：均值与中位数在本数据上**符号相反**
   （少数块出现 $-100\\%$ 量级的回退），所以「用哪个估计量」和「过滤哪些块」
   一样是一个显著性杠杆，必须连分布形状与左尾一起报。

2. ``seed_stability``：208 个配置有 3 个种子。把「插件带来的块级增益」和
   「同一配置内换种子带来的波动」放在同一把尺子上量。如果多数 win 的幅度小于
   种子噪声，那么「插件有效」这句话在单次运行的粒度上是不可分辨的。

3. ``leadtime_profile``：``seg1_mse..seg4_mse`` 是**预测步长四分位**上的误差
   （见 ``runner.StreamingMetrics.result``：对 per-timestep MSE 做 ``array_split``），
   不是测试期的时间分段。于是可以问一个和逐窗口分析完全独立的问题：
   在同一个 (dataset, horizon, backbone) 格子内部，最优臂在近端和远端步长上
   是同一个吗？这条证据有两重价值：
   (a) **覆盖全部 844 格、包含 TimesNet**，而逐窗口那条链因为 val 数组的
   provenance 问题只覆盖 127/96 组；
   (b) 更重要的是，「按 lead-time 切换臂」是本文唯一**因果合法**的选择轴——
   lead-time 下标在决策时就已知，不需要等 $H$ 步反馈。所以这一项同时报出
   per-lead-time oracle 相对整格最优固定臂的 headroom：如果连这个上界都接近零，
   那么合法轴上也没有可做的事，论文的否定结论才算闭合。

4. ``cost_benefit``：插件不是免费的。用 ``train_seconds`` / ``extra_params`` /
   ``peak_mem_mib`` 对齐到同格 ``none``，给出「多付多少算力」与「换回多少误差」
   的配对比值。审稿人问「即便增益不显著，代价小也无妨」时需要这张表。

用法
----
    python -m src.robustness --results monday_final/p1_results.csv \\
        --out artifacts/robustness
    python -m src.robustness --demo      # 用 stats.demo_results 自检整条链路
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import skew, spearmanr

from .stats import ALPHA, BLOCK_KEYS, build_block_matrix, holm, wilcoxon_pair

SEG_COLS = ("seg1_mse", "seg2_mse", "seg3_mse", "seg4_mse")

#: 「灾难性回退」的两档阈值（相对增益百分点）。取 -10 / -25 是因为长时序预测里
#: 单个块 10% 的 MSE 恶化已经足够抹掉一整年的 benchmark 进展，25% 则是任何
#: 上线流程都会直接回滚的量级。阈值写死在这里以便论文与代码口径一致。
TAIL_THRESHOLDS = (-10.0, -25.0)


# --------------------------------------------------------------------------- #
# 通用：配对自举置信区间
# --------------------------------------------------------------------------- #
def paired_bootstrap_ci(
    treat: np.ndarray,
    control: np.ndarray,
    n_boot: int = 10_000,
    alpha: float = ALPHA,
    seed: int = 26,
) -> dict[str, float]:
    """对「相对增益」做**按块重抽样**的百分位自举置信区间。

    为什么必须按块重抽样而不是对增益值独立重抽样：块是配对单元，
    treat 和 control 在同一块内是强相关的，独立重抽样会破坏配对结构、
    把 CI 算得过窄。这里每次自举抽的是**块下标**，treat/control 同步取值。

    返回 mean/median 相对增益的点估计与 CI（单位：%，正 = 误差降低）。
    相对增益定义与 ``stats.wilcoxon_pair`` 完全一致：``(control-treat)/|control|``，
    保证论文里同一个数字不会因为算了两遍而对不上。
    """
    treat = np.asarray(treat, dtype=float)
    control = np.asarray(control, dtype=float)
    if treat.shape != control.shape:
        raise ValueError(f"treat/control 形状不一致: {treat.shape} vs {control.shape}")
    n = treat.size
    out = {
        "n_blocks": int(n),
        "mean_rel_gain_pct": float("nan"),
        "mean_ci_lo": float("nan"),
        "mean_ci_hi": float("nan"),
        "median_rel_gain_pct": float("nan"),
        "median_ci_lo": float("nan"),
        "median_ci_hi": float("nan"),
    }
    if n == 0:
        return out
    rel = (control - treat) / np.abs(control) * 100.0
    out["mean_rel_gain_pct"] = float(np.mean(rel))
    out["median_rel_gain_pct"] = float(np.median(rel))
    if n == 1:
        return out
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot = rel[idx]
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    means = boot.mean(axis=1)
    meds = np.median(boot, axis=1)
    out["mean_ci_lo"], out["mean_ci_hi"] = (float(np.percentile(means, lo_q)),
                                            float(np.percentile(means, hi_q)))
    out["median_ci_lo"], out["median_ci_hi"] = (float(np.percentile(meds, lo_q)),
                                                float(np.percentile(meds, hi_q)))
    return out


# --------------------------------------------------------------------------- #
# 通用：分布形状与尾部风险
# --------------------------------------------------------------------------- #
def tail_stats(rel: pd.Series, thresholds: Sequence[float] = TAIL_THRESHOLDS) -> dict[str, Any]:
    """描述逐块相对增益分布的**形状与左尾**，而不只是它的中心。

    为什么需要它：``mean`` 与 ``median`` 在这份数据上会给出**符号相反**的结论
    （\\arm{fredf} 的 MSE 均值 $-2.37\\%$、中位数 $+0.45\\%$），原因是少数块出现了
    量级极大的回退。只报中心估计量会把这个事实藏起来，而它恰恰是「估计量的选择
    本身就是一个显著性杠杆」这一结论的证据。因此这里同时报出：

    - ``skew``：分布偏度，解释均值/中位数为何分道扬镳；
    - ``frac_worse_than_*``：跌破各档阈值的块占比，即尾部风险的频率；
    - ``worst_block`` / ``worst_rel_gain_pct``：最坏的那一格是谁，便于论文点名；
    - ``mean_minus_median``：两个估计量的差，直接量化「换个估计量」的位移。

    ``rel`` 的索引应当是块标签（``build_block_matrix`` 的行索引），这样
    ``worst_block`` 才有意义。
    """
    v = np.asarray(rel, dtype=float)
    out: dict[str, Any] = {
        "skew": float("nan"), "mean_minus_median": float("nan"),
        "worst_block": None, "worst_rel_gain_pct": float("nan"),
        "best_block": None, "best_rel_gain_pct": float("nan"),
    }
    for t in thresholds:
        out[f"frac_worse_than_{abs(t):g}pct"] = float("nan")
    if v.size == 0:
        return out
    out["mean_minus_median"] = float(np.mean(v) - np.median(v))
    if v.size >= 3 and np.std(v) > 0:
        out["skew"] = float(skew(v))
    for t in thresholds:
        out[f"frac_worse_than_{abs(t):g}pct"] = float(np.mean(v < t))
    idx = list(rel.index) if isinstance(rel, pd.Series) else list(range(v.size))
    out["worst_block"] = str(idx[int(np.argmin(v))])
    out["worst_rel_gain_pct"] = float(np.min(v))
    out["best_block"] = str(idx[int(np.argmax(v))])
    out["best_rel_gain_pct"] = float(np.max(v))
    return out


# --------------------------------------------------------------------------- #
# 审计 1：指标稳健性（MSE vs MAE）
# --------------------------------------------------------------------------- #
def metric_robustness(
    results: pd.DataFrame,
    metrics: Sequence[str] = ("mse", "mae"),
    control: str = "none",
    alpha: float = ALPHA,
    n_boot: int = 10_000,
    seed: int = 26,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """逐指标复算「插件 vs 对照」的配对检验，并给出跨指标一致性。

    返回 ``(per_metric, agreement)``：

    - ``per_metric``：每个 (metric, method) 一行。检验统计量直接来自
      ``stats.wilcoxon_pair``（同一口径），额外补上自举 CI 与
      ``tail_stats`` 的分布形状/尾部风险。Holm 族按**指标内的方法数**取
      （每个指标是一次独立的报告，不跨指标合并校正，否则族规模翻倍、功效被
      无谓稀释；这个选择在论文里要写明）。
    - ``agreement``：每个 method 一行，报出两两指标之间逐块相对增益的
      Spearman ρ 与符号一致率。ρ 高但显著性结论相反，说明结论对指标敏感
      不是因为排序变了，而是因为幅度贴着零。
    """
    metrics = list(metrics)
    rows: list[dict[str, Any]] = []
    per_block: dict[tuple[str, str], pd.Series] = {}

    for metric in metrics:
        if metric not in results.columns:
            raise KeyError(f"结果表缺少指标列 {metric!r}；现有列: {sorted(results.columns)[:20]}…")
        mat = build_block_matrix(results, metric=metric, require_complete=False)
        if control not in mat.columns:
            raise KeyError(f"指标 {metric!r} 的矩阵里没有对照臂 {control!r}")
        methods = [c for c in mat.columns if c != control]
        stat_rows = []
        for m in methods:
            base = wilcoxon_pair(mat, m, control)
            pair = mat[[m, control]].dropna(axis=0, how="any")
            ci = paired_bootstrap_ci(pair[m].to_numpy(float), pair[control].to_numpy(float),
                                     n_boot=n_boot, alpha=alpha, seed=seed)
            rel = (pair[control] - pair[m]) / pair[control].abs() * 100.0
            per_block[(metric, m)] = rel
            stat_rows.append({"metric": metric, **base,
                              **{k: v for k, v in ci.items() if k != "n_blocks"},
                              **tail_stats(rel)})
        adj = holm({r["method"]: r["p_value"] for r in stat_rows}, alpha).set_index("comparison")
        for r in stat_rows:
            a = adj.loc[r["method"]]
            r["p_holm"] = float(a["p_holm_adjusted"])
            r["reject_H0"] = bool(a["reject_H0"])
            r["holm_family_size"] = len(stat_rows)
            # 「估计量杠杆」：均值与中位数是否给出相反符号的结论
            r["mean_median_sign_conflict"] = bool(
                np.isfinite(r["mean_rel_gain_pct"]) and np.isfinite(r["median_rel_gain_pct"])
                and np.sign(r["mean_rel_gain_pct"]) != np.sign(r["median_rel_gain_pct"]))
        rows.extend(stat_rows)

    per_metric = pd.DataFrame(rows)

    agree_rows: list[dict[str, Any]] = []
    for i, m1 in enumerate(metrics):
        for m2 in metrics[i + 1:]:
            methods = sorted({m for (mm, m) in per_block if mm == m1} &
                             {m for (mm, m) in per_block if mm == m2})
            for m in methods:
                a, b = per_block[(m1, m)], per_block[(m2, m)]
                common = a.index.intersection(b.index)
                av, bv = a.loc[common].to_numpy(float), b.loc[common].to_numpy(float)
                rho = float("nan")
                if common.size >= 3 and np.std(av) > 0 and np.std(bv) > 0:
                    rho = float(spearmanr(av, bv).statistic)
                sign_agree = (float(np.mean(np.sign(av) == np.sign(bv)))
                              if common.size else float("nan"))
                # 结论是否翻转：以两个指标各自的 Holm 判决为准
                v1 = per_metric[(per_metric.metric == m1) & (per_metric.method == m)]
                v2 = per_metric[(per_metric.metric == m2) & (per_metric.method == m)]
                r1 = bool(v1.reject_H0.iloc[0]) if len(v1) else False
                r2 = bool(v2.reject_H0.iloc[0]) if len(v2) else False
                agree_rows.append({
                    "method": m, "metric_a": m1, "metric_b": m2,
                    "n_common_blocks": int(common.size),
                    "spearman_rho": rho, "sign_agreement": sign_agree,
                    "reject_a": r1, "reject_b": r2, "verdict_flips": bool(r1 != r2),
                })
    return per_metric, pd.DataFrame(agree_rows)


# --------------------------------------------------------------------------- #
# 审计 2：随机种子噪声 vs 插件效应
# --------------------------------------------------------------------------- #
def seed_stability(
    results: pd.DataFrame,
    metric: str = "mse",
    control: str = "none",
    block_keys: Iterable[str] = BLOCK_KEYS,
    method_col: str = "plugin",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """把「插件增益」和「同配置换种子的波动」放到同一把尺子上。

    做法：
    1. 对每个 (block, arm)，若有 ≥2 个种子，算种子间标准差 ``seed_sd``；
    2. 块级增益 = ``mean_over_seeds(control) - mean_over_seeds(arm)``；
    3. 噪声基准取该块内 arm 与 control **两者种子标准差的合并值**
       ``sqrt(sd_arm² + sd_ctrl²)``——这是两个独立均值之差的标准差尺度，
       比只用单边 sd 更保守（不会人为放大「超过噪声」的比例）；
    4. 报出 ``|gain| < noise`` 的块占比，即「换个种子就可能翻转」的比例。

    注意：只用**同时**有多种子 arm 和多种子 control 的块，否则噪声估不出来；
    ``n_blocks_with_noise`` 会如实报出这个子集的大小，不拿单种子块凑数。
    """
    keys = list(block_keys)
    need = keys + [method_col, "seed", metric]
    missing = [c for c in need if c not in results.columns]
    if missing:
        raise KeyError(f"结果表缺列: {missing}")

    g = results.groupby(keys + [method_col])[metric]
    agg = g.agg(mean="mean", sd="std", n_seeds="nunique").reset_index()
    agg["block"] = agg[keys].astype(str).agg("|".join, axis=1)

    ctrl = agg[agg[method_col] == control].set_index("block")
    rows = []
    for m, sub in agg[agg[method_col] != control].groupby(method_col):
        sub = sub.set_index("block")
        common = sub.index.intersection(ctrl.index)
        for b in common:
            sa, sc = sub.loc[b], ctrl.loc[b]
            gain = float(sc["mean"] - sa["mean"])
            sd_a = float(sa["sd"]) if sa["n_seeds"] > 1 else float("nan")
            sd_c = float(sc["sd"]) if sc["n_seeds"] > 1 else float("nan")
            noise = (float(np.sqrt(sd_a**2 + sd_c**2))
                     if np.isfinite(sd_a) and np.isfinite(sd_c) else float("nan"))
            rows.append({
                "block": b, "method": m,
                "n_seeds_method": int(sa["n_seeds"]), "n_seeds_control": int(sc["n_seeds"]),
                "gain_abs": gain,
                "rel_gain_pct": gain / abs(float(sc["mean"])) * 100.0,
                "seed_sd_method": sd_a, "seed_sd_control": sd_c, "noise_scale": noise,
                "gain_over_noise": abs(gain) / noise if np.isfinite(noise) and noise > 0 else float("nan"),
                "within_noise": (bool(abs(gain) < noise) if np.isfinite(noise) else None),
            })
    detail = pd.DataFrame(rows)

    summary: dict[str, Any] = {"metric": metric, "control": control}
    if len(detail):
        usable = detail.dropna(subset=["noise_scale"])
        summary["n_block_arm_pairs"] = int(len(detail))
        summary["n_blocks_with_noise"] = int(len(usable))
        summary["median_seed_sd"] = float(
            pd.concat([usable.seed_sd_method, usable.seed_sd_control]).median())
        if len(usable):
            summary["frac_within_noise"] = float(usable.within_noise.mean())
            summary["median_gain_over_noise"] = float(usable.gain_over_noise.median())
            wins = usable[usable.gain_abs > 0]
            summary["n_wins"] = int(len(wins))
            summary["frac_wins_within_noise"] = (float(wins.within_noise.mean())
                                                 if len(wins) else float("nan"))
            by = usable.groupby("method").agg(
                n=("within_noise", "size"),
                frac_within_noise=("within_noise", "mean"),
                median_gain_over_noise=("gain_over_noise", "median"))
            summary["by_method"] = by.reset_index().to_dict("records")
    return detail, summary


# --------------------------------------------------------------------------- #
# 审计 3：lead-time 剖面上的最优臂稳定性
# --------------------------------------------------------------------------- #
def leadtime_profile(
    results: pd.DataFrame,
    control: str = "none",
    seg_cols: Sequence[str] = SEG_COLS,
    block_keys: Iterable[str] = BLOCK_KEYS,
    method_col: str = "plugin",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """在**预测步长四分位**上重做「谁是最优臂」。

    ``seg{i}_mse`` 的语义（见 ``runner.StreamingMetrics.result``）：先把测试集上
    per-timestep 的 MSE 求出来，再对长度为 ``pred_len`` 的这条曲线做 4 等分取均值。
    所以它刻画的是「近端 vs 远端预测步长」的误差结构，**不是**测试期的时间漂移。

    两个产出：
    - ``profile``：每个 (block, arm, quarter) 的误差与相对 ``control`` 的增益，
      用来看「插件的收益集中在近端还是远端」；
    - ``stability``：每个 block 一行，报出 4 个四分位各自的 argmin 臂、
      是否 4 个四分位同一个臂、以及「按整格 MSE 选出的臂」在多少个四分位上
      其实不是最优。后者直接量化了「一个格子只能选一个臂」这件事本身的损失。

    覆盖面是这项审计的价值所在：``seg*_mse`` 对全部 844 格都有，
    包含逐窗口链路缺席的 TimesNet。
    """
    keys = list(block_keys)
    seg_cols = [c for c in seg_cols if c in results.columns]
    if not seg_cols:
        raise KeyError(f"结果表缺少 lead-time 分段列，期望 {SEG_COLS}")

    agg = results.groupby(keys + [method_col])[list(seg_cols) + ["mse"]].mean().reset_index()
    agg["block"] = agg[keys].astype(str).agg("|".join, axis=1)

    long = agg.melt(id_vars=["block", method_col], value_vars=seg_cols,
                    var_name="quarter", value_name="seg_mse")
    long["quarter"] = long["quarter"].str.extract(r"seg(\d+)_mse").astype(int)
    ctrl = long[long[method_col] == control].set_index(["block", "quarter"])["seg_mse"]
    long = long.join(ctrl.rename("seg_mse_control"), on=["block", "quarter"])
    long["rel_gain_pct"] = ((long.seg_mse_control - long.seg_mse)
                            / long.seg_mse_control.abs() * 100.0)
    profile = long.rename(columns={method_col: "method"})

    stab_rows = []
    whole = agg.set_index(["block", method_col])["mse"]
    for b, sub in long.groupby("block"):
        piv = sub.pivot_table(index="quarter", columns=method_col, values="seg_mse")
        if piv.empty:
            continue
        # 每个四分位的 argmin 臂（该四分位上所有可用臂里误差最小的）
        per_q = {int(q): str(piv.loc[q].idxmin()) for q in piv.index if piv.loc[q].notna().any()}
        arms = list(per_q.values())
        try:
            whole_best = str(whole.loc[b].idxmin())
        except KeyError:
            whole_best = None
        n_q = len(arms)
        # ---- 「按 lead-time 切换臂」这条**合法**轴能拿到多少 ---------------- #
        # 四分位是 pred_len 的等分，所以 block MSE = 四个四分位 MSE 的均值
        # （已在实现中对 ``mse`` 逐格核对，误差 0）。因此可以直接把
        # 「逐四分位取最小」当作 per-lead-time oracle 的块级 MSE。
        per_q_oracle = float(piv.min(axis=1).mean()) if n_q else float("nan")
        col_means = piv.mean(axis=0)                      # 各臂的整格 MSE
        best_fixed = float(col_means.min()) if len(col_means) else float("nan")
        ctrl_mse = float(col_means.get(control, np.nan))
        stab_rows.append({
            "block": b,
            "n_quarters": n_q,
            "n_arms_available": int(piv.notna().any(axis=0).sum()),
            "argmin_by_quarter": ";".join(f"q{q}:{a}" for q, a in sorted(per_q.items())),
            "n_distinct_argmin": int(len(set(arms))),
            "constant_argmin": bool(n_q > 0 and len(set(arms)) == 1),
            "whole_horizon_best": whole_best,
            "n_quarters_where_whole_best_suboptimal": (
                int(sum(1 for a in arms if whole_best is not None and a != whole_best))),
            "per_leadtime_oracle_mse": per_q_oracle,
            "best_fixed_arm_mse": best_fixed,
            "control_mse": ctrl_mse,
            "leadtime_oracle_gain_vs_control_pct": (
                (ctrl_mse - per_q_oracle) / abs(ctrl_mse) * 100.0
                if np.isfinite(ctrl_mse) and ctrl_mse != 0 else float("nan")),
            "leadtime_oracle_gain_vs_best_fixed_pct": (
                (best_fixed - per_q_oracle) / abs(best_fixed) * 100.0
                if np.isfinite(best_fixed) and best_fixed != 0 else float("nan")),
        })
    stability = pd.DataFrame(stab_rows)

    summary: dict[str, Any] = {}
    if len(stability):
        multi = stability[stability.n_arms_available > 1]
        summary["n_blocks"] = int(len(stability))
        summary["n_blocks_multi_arm"] = int(len(multi))
        if len(multi):
            summary["frac_constant_argmin"] = float(multi.constant_argmin.mean())
            summary["mean_n_distinct_argmin"] = float(multi.n_distinct_argmin.mean())
            summary["frac_blocks_whole_best_suboptimal_somewhere"] = float(
                (multi.n_quarters_where_whole_best_suboptimal > 0).mean())
            # 这条轴的 headroom：注意它是**可部署**的（lead-time 下标在决策时已知），
            # 所以它的 oracle 上界才是论文真正关心的量——如果连上界都接近零，
            # 就不必再去实现一个 per-lead-time 选择器。
            gvc = multi.leadtime_oracle_gain_vs_control_pct.dropna()
            gvf = multi.leadtime_oracle_gain_vs_best_fixed_pct.dropna()
            if len(gvc):
                summary["leadtime_oracle_vs_control_mean_pct"] = float(gvc.mean())
                summary["leadtime_oracle_vs_control_median_pct"] = float(gvc.median())
            if len(gvf):
                summary["leadtime_oracle_vs_best_fixed_mean_pct"] = float(gvf.mean())
                summary["leadtime_oracle_vs_best_fixed_median_pct"] = float(gvf.median())
                summary["leadtime_oracle_vs_best_fixed_max_pct"] = float(gvf.max())
                summary["n_blocks_leadtime_oracle_beats_best_fixed"] = int((gvf > 1e-9).sum())
                summary["frac_blocks_leadtime_gain_over_0p5pp"] = float((gvf > 0.5).mean())
    if len(profile):
        by_q = (profile[profile.method != control]
                .groupby(["method", "quarter"]).rel_gain_pct
                .agg(["mean", "median", "size"]).reset_index())
        summary["gain_by_quarter"] = by_q.to_dict("records")
    return profile, stability, summary


# --------------------------------------------------------------------------- #
# 审计 4：成本收益
# --------------------------------------------------------------------------- #
def cost_benefit(
    results: pd.DataFrame,
    control: str = "none",
    metric: str = "mse",
    block_keys: Iterable[str] = BLOCK_KEYS,
    method_col: str = "plugin",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """插件的算力代价 vs 误差收益（同格配对，比值口径）。

    对齐到同一个块内的 ``control``，报出训练时长比、额外参数、峰值显存差，
    以及配对的相对增益。``per_method`` 汇总用中位数（比值分布长尾，均值不稳）。
    """
    keys = list(block_keys)
    cost_cols = [c for c in ("train_seconds", "extra_params", "n_params", "peak_mem_mib")
                 if c in results.columns]
    agg = results.groupby(keys + [method_col])[[metric] + cost_cols].mean().reset_index()
    agg["block"] = agg[keys].astype(str).agg("|".join, axis=1)
    ctrl = agg[agg[method_col] == control].set_index("block")

    rows = []
    for m, sub in agg[agg[method_col] != control].groupby(method_col):
        sub = sub.set_index("block")
        for b in sub.index.intersection(ctrl.index):
            a, c = sub.loc[b], ctrl.loc[b]
            r: dict[str, Any] = {"block": b, "method": m,
                                 "rel_gain_pct": float((c[metric] - a[metric]) / abs(c[metric]) * 100)}
            if "train_seconds" in cost_cols and float(c["train_seconds"]) > 0:
                r["train_time_ratio"] = float(a["train_seconds"]) / float(c["train_seconds"])
            if "extra_params" in cost_cols:
                r["extra_params"] = float(a["extra_params"])
            if "peak_mem_mib" in cost_cols:
                r["peak_mem_delta_mib"] = float(a["peak_mem_mib"]) - float(c["peak_mem_mib"])
            rows.append(r)
    detail = pd.DataFrame(rows)

    if not len(detail):
        return detail, pd.DataFrame()
    aggs: dict[str, Any] = {"n_blocks": ("rel_gain_pct", "size"),
                            "median_rel_gain_pct": ("rel_gain_pct", "median")}
    if "train_time_ratio" in detail:
        aggs["median_train_time_ratio"] = ("train_time_ratio", "median")
    if "extra_params" in detail:
        aggs["median_extra_params"] = ("extra_params", "median")
    if "peak_mem_delta_mib" in detail:
        aggs["median_peak_mem_delta_mib"] = ("peak_mem_delta_mib", "median")
    per_method = detail.groupby("method").agg(**aggs).reset_index()
    if "median_train_time_ratio" in per_method:
        # 「多花 1% 训练时间换到的误差降低百分点」——正数才算划得来
        per_method["gain_per_pct_extra_time"] = per_method.apply(
            lambda r: (r["median_rel_gain_pct"] / ((r["median_train_time_ratio"] - 1) * 100)
                       if r["median_train_time_ratio"] > 1 else float("nan")), axis=1)
    return detail, per_method


# --------------------------------------------------------------------------- #
# 自检用合成数据
# --------------------------------------------------------------------------- #
def demo_results_rich(seed: int = 26, n_seeds: int = 3) -> pd.DataFrame:
    """在 ``stats.demo_results`` 之上补出多种子、lead-time 分段与成本列。

    为什么需要它：``stats.demo_results`` 只有单种子、没有 ``seg*_mse``，
    用它跑 ``--demo`` 会让审计 2 和审计 3 静默返回空表——一个跑不满全部分支的
    自检等于没有自检。这里显式构造出四项审计都能被触发的结构：

    - 多种子：同配置加一个小幅高斯抖动，于是种子标准差可估；
    - lead-time：误差随预测步长单调上升（真实数据就是这样），并让
      ``san_lite`` 的相对劣势集中在近端、``fredf`` 集中在远端，
      这样 argmin 臂会在四分位之间发生切换，审计 3 有东西可报；
    - 成本：``san_lite`` 训练更慢，``fredf`` 略慢，用于审计 4。
    """
    from .stats import demo_results

    base = demo_results(seed)
    rng = np.random.default_rng(seed + 1)
    rows = []
    for _, r in base.iterrows():
        for si in range(n_seeds):
            mse = float(r["mse"]) * float(rng.normal(1.0, 0.01))
            mae = float(r["mae"]) * float(rng.normal(1.0, 0.01))
            # lead-time 剖面：整体随步长上升；插件的相对优劣按步长倾斜
            tilt = {"none": 0.0, "revin": -0.01, "san_lite": 0.06, "fredf": -0.05}.get(
                str(r["plugin"]), 0.0)
            segs = [mse * (0.7 + 0.2 * q) * (1.0 + tilt * (1.5 - q))
                    for q in range(4)]
            speed = {"none": 1.0, "revin": 1.05, "san_lite": 1.7, "fredf": 1.03}.get(
                str(r["plugin"]), 1.0)
            rows.append({
                **{k: r[k] for k in ("dataset", "pred_len", "backbone", "plugin")},
                "seed": 2021 + si, "mse": mse, "mae": mae,
                **{f"seg{i + 1}_mse": s for i, s in enumerate(segs)},
                "train_seconds": 30.0 * speed * float(rng.normal(1.0, 0.02)),
                "extra_params": 0.0,
                "peak_mem_mib": 100.0 + (5.0 if r["plugin"] != "none" else 0.0),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
def run_all(results: pd.DataFrame, out_dir: str | Path, control: str = "none",
            n_boot: int = 10_000, seed: int = 26) -> dict[str, Any]:
    """跑完四项审计，写 CSV + JSON，返回摘要（论文里引用的数字都从这里出）。"""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"control": control, "n_rows": int(len(results))}

    metrics = [m for m in ("mse", "mae") if m in results.columns]
    per_metric, agreement = metric_robustness(results, metrics, control,
                                              n_boot=n_boot, seed=seed)
    per_metric.to_csv(out / "metric_robustness.csv", index=False)
    agreement.to_csv(out / "metric_agreement.csv", index=False)
    summary["metric_robustness"] = per_metric.to_dict("records")
    summary["metric_agreement"] = agreement.to_dict("records")

    seed_detail, seed_sum = seed_stability(results, control=control)
    seed_detail.to_csv(out / "seed_stability.csv", index=False)
    summary["seed_stability"] = seed_sum

    try:
        profile, stability, lt_sum = leadtime_profile(results, control=control)
        profile.to_csv(out / "leadtime_profile.csv", index=False)
        stability.to_csv(out / "leadtime_argmin_stability.csv", index=False)
        summary["leadtime"] = lt_sum
    except KeyError as e:
        summary["leadtime"] = {"skipped": str(e)}

    cb_detail, cb_method = cost_benefit(results, control=control)
    cb_detail.to_csv(out / "cost_benefit.csv", index=False)
    cb_method.to_csv(out / "cost_benefit_by_method.csv", index=False)
    summary["cost_benefit"] = cb_method.to_dict("records")

    (out / "robustness_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[robust] 输出目录: {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="投稿前增量审计（指标/种子/lead-time/成本）")
    ap.add_argument("--results", default="monday_final/p1_results.csv")
    ap.add_argument("--out", default="artifacts/robustness")
    ap.add_argument("--control", default="none")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--demo", action="store_true", help="用 stats.demo_results 自检")
    args = ap.parse_args()

    if args.demo:
        df = demo_results_rich()
        print(f"[robust] demo 结果表 shape={df.shape}（含多种子 / lead-time 分段 / 成本列）")
    else:
        df = pd.read_csv(args.results)

    s = run_all(df, args.out, args.control, args.n_boot, args.seed)

    pm = pd.DataFrame(s["metric_robustness"])
    print("\n=== 审计 1：指标稳健性（各插件 vs 对照，Holm 按指标内校正）===")
    print(pm[["metric", "method", "n_blocks", "n_eff", "win", "loss",
              "median_rel_gain_pct", "median_ci_lo", "median_ci_hi",
              "p_value", "p_holm", "reject_H0"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n--- 估计量杠杆与左尾风险（均值 vs 中位数）---")
    tail_cols = ["metric", "method", "mean_rel_gain_pct", "median_rel_gain_pct",
                 "mean_minus_median", "mean_median_sign_conflict", "skew",
                 "frac_worse_than_10pct", "frac_worse_than_25pct",
                 "worst_block", "worst_rel_gain_pct"]
    print(pm[[c for c in tail_cols if c in pm.columns]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    ag = pd.DataFrame(s["metric_agreement"])
    if len(ag):
        print("\n--- 跨指标一致性 ---")
        print(ag.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    ss = s["seed_stability"]
    print("\n=== 审计 2：种子噪声 vs 插件效应 ===")
    for k in ("n_block_arm_pairs", "n_blocks_with_noise", "median_seed_sd",
              "frac_within_noise", "median_gain_over_noise", "n_wins",
              "frac_wins_within_noise"):
        if k in ss:
            print(f"  {k} = {ss[k]}")
    if "by_method" in ss:
        print(pd.DataFrame(ss["by_method"]).to_string(index=False,
                                                      float_format=lambda v: f"{v:.4g}"))

    lt = s.get("leadtime", {})
    print("\n=== 审计 3：lead-time 四分位上的最优臂稳定性 ===")
    for k in ("n_blocks", "n_blocks_multi_arm", "frac_constant_argmin",
              "mean_n_distinct_argmin", "frac_blocks_whole_best_suboptimal_somewhere",
              "leadtime_oracle_vs_control_mean_pct", "leadtime_oracle_vs_control_median_pct",
              "leadtime_oracle_vs_best_fixed_mean_pct",
              "leadtime_oracle_vs_best_fixed_median_pct",
              "leadtime_oracle_vs_best_fixed_max_pct",
              "n_blocks_leadtime_oracle_beats_best_fixed",
              "frac_blocks_leadtime_gain_over_0p5pp"):
        if k in lt:
            print(f"  {k} = {lt[k]}")
    if "gain_by_quarter" in lt:
        print(pd.DataFrame(lt["gain_by_quarter"]).to_string(
            index=False, float_format=lambda v: f"{v:.4g}"))

    cb = pd.DataFrame(s["cost_benefit"])
    if len(cb):
        print("\n=== 审计 4：成本收益 ===")
        print(cb.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
