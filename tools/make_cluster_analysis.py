#!/usr/bin/env python
"""把「按实验条件汇总」的结论重做成「以 dataset 为一级统计单位」的依赖感知推断。

审稿意见：论文把 (backbone x dataset x horizon x seed) 的每个组合当独立证据单位做
Wilcoxon / 计数 / bootstrap，但真正的抽样单位是 dataset，数量极少。本脚本不改论文，
只从已落盘产物重算：单位层级计数、dataset-level 效应表、以 dataset 为 cluster 的
bootstrap、结论级别 leave-one-dataset-out、dataset-level 配对检验（含 n 很小时的 p
下界）、以及 seed 噪声与 selector 增益的直接对比。

用法
----
    python tools/make_cluster_analysis.py
    python tools/make_cluster_analysis.py --boot 20000 --jobs 8
    python tools/make_cluster_analysis.py --out-dir artifacts/cluster
"""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, friedmanchisquare, rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RESULTS = "monday_final"
DEFAULT_ROBUST = "artifacts/robustness"
DEFAULT_OUT = "artifacts/cluster"

ARMS = ("revin", "san_lite", "fredf")
CONTROL = "none"

#: bootstrap 复制份数固定切成 8 块，保证结果与 --jobs 无关。
BOOT_CHUNKS = 8

#: 三个核心量：论文的口径原样保留，只是把统计单位换成 dataset。
PRIMARY = ("a_online_vs_bf", "b_oracle_vs_none", "c_gate_vs_bf")


# --------------------------------------------------------------------------- #
# 产物加载与三个核心量的长表
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Effect:
    """一个核心量：论文印的口径 + 它真正的统计单位。"""

    mid: str
    label: str
    paper_section: str
    source: str
    unit: str
    value_col: str
    note: str = ""


EFFECTS: tuple[Effect, ...] = (
    Effect("a_online_vs_bf",
           "online delayed-feedback selector vs best_fixed_val (%)",
           "§9.3 (tab:online)",
           "monday_final/p2/selector_window_online.csv (insufficient==False)",
           "decision group = (backbone, dataset, pred_len) @ seed 2021",
           "gain_online_vs_best_fixed",
           note="论文头条 -0.87 = 123 组的算术平均"),
    Effect("b_oracle_vs_none",
           "per-window oracle headroom vs none (%)",
           "§8 (oracle.headroom_127)",
           "monday_final/p2/selector_window_oracle_audit.csv",
           "decision group = (backbone, dataset, pred_len) @ seed 2021",
           "oracle_gain_real_pct",
           note="论文头条 +10.15 = 127 组的算术平均"),
    Effect("c_gate_vs_bf",
           "static complexity-feature gate vs best_fixed_val (%)",
           "§9.1 (tab:gate, in-pool reg)",
           "monday_final/p2/selector_window_inpool.csv",
           "decision group = (backbone, dataset, pred_len) @ seed 2021",
           "gain_gate_vs_best_fixed",
           note="论文头条 -0.67 = 96 组的算术平均"),
    Effect("b2_oracle_vs_bf",
           "per-window oracle headroom vs best_fixed_val (%), online support",
           "§8 派生量（论文未单印）",
           "monday_final/p2/selector_window_online.csv (insufficient==False)",
           "decision group = (backbone, dataset, pred_len) @ seed 2021",
           "gain_oracle_vs_best_fixed_derived",
           note="由 mse_best_fixed / mse_oracle 重算，参考臂换成可部署基线"),
    Effect("c2_gate_vs_none",
           "static complexity-feature gate vs none (%)",
           "§9.1 (tab:gate, in-pool reg)",
           "monday_final/p2/selector_window_inpool.csv",
           "decision group = (backbone, dataset, pred_len) @ seed 2021",
           "gain_gate_vs_none",
           note="论文头条 +1.50"),
)

EFFECT_BY_ID = {e.mid: e for e in EFFECTS}


def load_long(results: Path) -> pd.DataFrame:
    """三个核心量 + 两个派生量的统一长表，一行 = 一个 (metric, group)。"""
    online = pd.read_csv(results / "p2" / "selector_window_online.csv")
    feas = online[~online.insufficient].copy()
    feas["gain_oracle_vs_best_fixed_derived"] = (
        (feas.mse_best_fixed - feas.mse_oracle) / feas.mse_best_fixed * 100.0)
    oracle = pd.read_csv(results / "p2" / "selector_window_oracle_audit.csv")
    inpool = pd.read_csv(results / "p2" / "selector_window_inpool.csv")
    frames = {"a_online_vs_bf": feas, "b_oracle_vs_none": oracle,
              "c_gate_vs_bf": inpool, "b2_oracle_vs_bf": feas,
              "c2_gate_vs_none": inpool}
    out = []
    for mid, frame in frames.items():
        eff = EFFECT_BY_ID[mid]
        sub = frame[["key", "backbone", "dataset", "pred_len", eff.value_col]].copy()
        sub = sub.rename(columns={eff.value_col: "value"})
        sub["metric"] = mid
        out.append(sub)
    long = pd.concat(out, ignore_index=True)
    if long.value.isna().any():
        raise ValueError("核心量里出现 NaN，口径需先澄清而不是静默丢弃")
    return long


# --------------------------------------------------------------------------- #
# 1. 单位层级
# --------------------------------------------------------------------------- #
def unit_hierarchy(results: Path, long: pd.DataFrame) -> dict[str, Any]:
    p1 = pd.read_csv(results / "p1_results.csv")
    oracle = pd.read_csv(results / "p2" / "selector_window_oracle_audit.csv")
    online = pd.read_csv(results / "p2" / "selector_window_online.csv")
    inpool = pd.read_csv(results / "p2" / "selector_window_inpool.csv")
    lodo = pd.read_csv(results / "p2" / "selector_window_lodo.csv")
    feas = online[~online.insufficient]

    def by(frame: pd.DataFrame, col: str) -> dict[str, int]:
        return {str(k): int(v) for k, v in frame[col].value_counts().sort_index().items()}

    def layer(name: str, frame: pd.DataFrame, unit: str, source: str,
              extra: dict[str, Any] | None = None) -> dict[str, Any]:
        d = {"name": name, "n": int(len(frame)), "unit": unit, "source": source,
             "n_datasets": int(frame.dataset.nunique()),
             "n_backbones": int(frame.backbone.nunique()),
             "n_dataset_horizon_pairs":
                 int(frame.groupby(["dataset", "pred_len"]).ngroups),
             "groups_by_dataset": by(frame, "dataset"),
             "groups_by_backbone": by(frame, "backbone"),
             "groups_by_pred_len": by(frame, "pred_len")}
        if extra:
            d.update(extra)
        return d

    hierarchy = {
        "run_level": {
            "n_runs_phase1": int(len(p1)),
            "n_runs_ok": int((p1.status == "ok").sum()),
            "n_cells_phase1": int(p1.groupby(
                ["backbone", "dataset", "pred_len", "plugin"]).ngroups),
            "n_blocks_phase1": int(p1.groupby(
                ["backbone", "dataset", "pred_len"]).ngroups),
            "n_datasets": int(p1.dataset.nunique()),
            "datasets": sorted(p1.dataset.unique().tolist()),
            "n_dataset_horizon_pairs": int(p1.groupby(["dataset", "pred_len"]).ngroups),
            "n_backbones": int(p1.backbone.nunique()),
            "backbones": sorted(p1.backbone.unique().tolist()),
            "n_horizons": int(p1.pred_len.nunique()),
            "horizons": sorted(int(h) for h in p1.pred_len.unique()),
            "n_seeds": int(p1.seed.nunique()),
            "seeds": sorted(int(s) for s in p1.seed.unique()),
            "runs_by_dataset": by(p1, "dataset"),
            "cells_by_dataset": {
                str(k): int(v) for k, v in
                p1.groupby("dataset").apply(
                    lambda d: d.groupby(["backbone", "pred_len", "plugin"]).ngroups,
                    include_groups=False).sort_index().items()},
        },
        "group_levels": {
            "test_side_127": layer(
                "test-side decision groups (§8, tab:persist)", oracle,
                "(backbone, dataset, pred_len) @ seed 2021",
                "selector_window_oracle_audit.csv",
                {"arm_counts": {str(k): int(v) for k, v in
                                oracle.n_arms.value_counts().sort_index().items()}}),
            "online_feasible_123": layer(
                "online groups with a legal H-delay decision (§9.3, tab:online)",
                feas, "(backbone, dataset, pred_len) @ seed 2021",
                "selector_window_online.csv insufficient==False",
                {"excluded": {"n": int(online.insufficient.sum()),
                              "keys": online[online.insufficient].key.tolist()}}),
            "gating_96": layer(
                "gate-eligible groups (§9.1, tab:gate)", inpool,
                "(backbone, dataset, pred_len) @ seed 2021",
                "selector_window_inpool.csv"),
            "lodo_rows_76": layer(
                "rows of the existing selector LODO (§9.1 LODO row)", lodo,
                "(backbone, dataset, pred_len) @ seed 2021, dataset = held-out one",
                "selector_window_lodo.csv",
                {"n_train_datasets_values":
                 sorted(int(v) for v in lodo.n_train_datasets.unique())}),
        },
    }

    n_ds = hierarchy["run_level"]["n_datasets"]
    hierarchy["cluster_units"] = {
        "n_clusters_used_as_primary_unit": n_ds,
        "clusters": hierarchy["run_level"]["datasets"],
        "n_clusters_lodo_eligible": int(lodo.dataset.nunique()),
        "clusters_lodo_eligible": sorted(lodo.dataset.unique().tolist()),
        "groups_per_cluster_by_metric": {
            mid: {str(k): int(v) for k, v in
                  long[long.metric == mid].dataset.value_counts().sort_index().items()}
            for mid in EFFECT_BY_ID},
    }
    per_ds_127 = hierarchy["group_levels"]["test_side_127"]["groups_by_dataset"]
    hierarchy["notes"] = [
        f"实测 dataset 数为 {n_ds}（{', '.join(hierarchy['run_level']['datasets'])}），"
        "不是 7；审稿意见里的 7 对应的是 selector LODO 子集（ILI 结构性缺席）。",
        "127 / 123 / 96 组的统计单位都是 (backbone, dataset, pred_len)，seed 固定 2021，"
        f"因此 group 在 dataset 维度上高度依赖：test 侧每个 dataset 贡献 "
        f"{min(per_ds_127.values())}-{max(per_ds_127.values())} 组。",
        f"(dataset, pred_len) 组合数为 "
        f"{hierarchy['run_level']['n_dataset_horizon_pairs']}："
        f"{n_ds} 个 dataset x 4 个 horizon，ILI 用 24/36/48/60，"
        "其余用 96/192/336/720，两族 horizon 不重叠。",
    ]
    return hierarchy


# --------------------------------------------------------------------------- #
# 2. dataset-level 效应表
# --------------------------------------------------------------------------- #
def dataset_effects(long: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mid, eff in EFFECT_BY_ID.items():
        sub = long[long.metric == mid]
        for dataset, g in list(sub.groupby("dataset")) + [("ALL", sub)]:
            v = g.value.to_numpy()
            rows.append({
                "metric": mid, "label": eff.label, "paper_section": eff.paper_section,
                "unit": eff.unit, "source": eff.source,
                "dataset": dataset, "n_groups": int(len(v)),
                "n_backbones": int(g.backbone.nunique()),
                "n_horizons": int(g.pred_len.nunique()),
                "mean_pct": float(v.mean()), "median_pct": float(np.median(v)),
                "min_pct": float(v.min()), "max_pct": float(v.max()),
                "sd_pct": float(v.std(ddof=1)) if len(v) > 1 else float("nan"),
                "n_groups_positive": int((v > 0).sum()),
                "n_groups_negative": int((v < 0).sum()),
                "frac_groups_positive_pct": float((v > 0).mean() * 100.0),
                "dataset_mean_sign": int(np.sign(v.mean())),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. cluster bootstrap
# --------------------------------------------------------------------------- #
STATS: dict[str, Callable[[np.ndarray], float]] = {
    "pooled_mean": lambda v: float(np.mean(v)),
    "pooled_median": lambda v: float(np.median(v)),
}


def _boot_chunk(args: tuple[list[np.ndarray], int, int]) -> np.ndarray:
    """一块 bootstrap 复制：重抽 cluster（有放回），把被抽中的组拼起来再算统计量。"""
    clusters, n_rep, seed = args
    rng = np.random.default_rng(seed)
    k = len(clusters)
    out = np.empty((n_rep, 3), dtype=float)
    for i in range(n_rep):
        idx = rng.integers(0, k, size=k)
        pooled = np.concatenate([clusters[j] for j in idx])
        out[i, 0] = pooled.mean()
        out[i, 1] = np.median(pooled)
        out[i, 2] = np.mean([clusters[j].mean() for j in idx])
    return out


def cluster_bootstrap(long: pd.DataFrame, boot: int, seed: int,
                      jobs: int) -> pd.DataFrame:
    scopes = {"all_datasets": None, "lodo_eligible_7": "ILI"}
    rows: list[dict[str, Any]] = []
    for mid, eff in EFFECT_BY_ID.items():
        for scope, drop in scopes.items():
            sub = long[long.metric == mid]
            if drop is not None:
                sub = sub[sub.dataset != drop]
            clusters = [g.value.to_numpy() for _, g in sub.groupby("dataset")]
            pooled = sub.value.to_numpy()
            point = {"pooled_mean": float(pooled.mean()),
                     "pooled_median": float(np.median(pooled)),
                     "mean_of_dataset_means":
                         float(np.mean([c.mean() for c in clusters]))}
            reps = _run_bootstrap(clusters, boot, seed, jobs)
            for col, (name, pt) in enumerate(point.items()):
                draws = reps[:, col]
                lo, hi = np.percentile(draws, [2.5, 97.5])
                rows.append({
                    "metric": mid, "label": eff.label, "scope": scope,
                    "statistic": name, "unit_resampled": "dataset (cluster)",
                    "n_clusters": len(clusters), "n_groups": int(len(pooled)),
                    "point_pct": pt, "ci95_lo_pct": float(lo), "ci95_hi_pct": float(hi),
                    "ci95_width_pct": float(hi - lo),
                    "boot_se_pct": float(draws.std(ddof=1)),
                    "frac_boot_negative_pct": float((draws < 0).mean() * 100.0),
                    "frac_boot_positive_pct": float((draws > 0).mean() * 100.0),
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                    "n_boot": int(boot), "seed": int(seed),
                })
    return pd.DataFrame(rows)


def _run_bootstrap(clusters: list[np.ndarray], boot: int, seed: int,
                   jobs: int) -> np.ndarray:
    sizes = [boot // BOOT_CHUNKS] * BOOT_CHUNKS
    for i in range(boot - sum(sizes)):
        sizes[i] += 1
    seeds = [int(s.generate_state(1)[0])
             for s in np.random.SeedSequence(seed).spawn(BOOT_CHUNKS)]
    tasks = [(clusters, n, s) for n, s in zip(sizes, seeds) if n > 0]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            parts = list(ex.map(_boot_chunk, tasks))
    else:
        parts = [_boot_chunk(t) for t in tasks]
    return np.concatenate(parts, axis=0)


# --------------------------------------------------------------------------- #
# 4. 结论级别 leave-one-dataset-out
# --------------------------------------------------------------------------- #
def lodo_conclusions(long: pd.DataFrame, results: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mid, eff in EFFECT_BY_ID.items():
        sub = long[long.metric == mid]
        full_mean = float(sub.value.mean())
        full_median = float(sub.value.median())
        datasets = sorted(sub.dataset.unique())
        rows.append({
            "metric": mid, "label": eff.label,
            "semantics": "conclusion-level jackknife of the aggregate",
            "held_out": "NONE", "n_datasets_remaining": len(datasets),
            "n_groups_remaining": int(len(sub)),
            "mean_pct": full_mean, "median_pct": full_median,
            "delta_mean_vs_full_pp": 0.0, "delta_median_vs_full_pp": 0.0,
            "sign_flips_mean": False, "sign_flips_median": False,
            "n_datasets_mean_positive": int(
                sum(sub[sub.dataset == d].value.mean() > 0 for d in datasets)),
        })
        for d in datasets:
            rest = sub[sub.dataset != d]
            m, med = float(rest.value.mean()), float(rest.value.median())
            left = sorted(rest.dataset.unique())
            rows.append({
                "metric": mid, "label": eff.label,
                "semantics": "conclusion-level jackknife of the aggregate",
                "held_out": d, "n_datasets_remaining": len(left),
                "n_groups_remaining": int(len(rest)),
                "mean_pct": m, "median_pct": med,
                "delta_mean_vs_full_pp": m - full_mean,
                "delta_median_vs_full_pp": med - full_median,
                "sign_flips_mean": bool(np.sign(m) != np.sign(full_mean)),
                "sign_flips_median": bool(np.sign(med) != np.sign(full_median)),
                "n_datasets_mean_positive": int(
                    sum(rest[rest.dataset == x].value.mean() > 0 for x in left)),
            })
    rows.extend(_existing_lodo_rows(results))
    return pd.DataFrame(rows)


def _existing_lodo_rows(results: Path) -> list[dict[str, Any]]:
    """已有 selector_window_lodo.csv 算的是「门控跨数据集迁移」，不是结论级留一。"""
    lodo = pd.read_csv(results / "p2" / "selector_window_lodo.csv")
    out: list[dict[str, Any]] = []
    full = float(lodo.gain_gate_vs_best_fixed.mean())
    for d, g in list(lodo.groupby("dataset")) + [("ALL", lodo)]:
        v = g.gain_gate_vs_best_fixed.to_numpy()
        out.append({
            "metric": "c_lodo_gate_transfer_existing",
            "label": "gate trained on other datasets, evaluated on the held-out one (%)",
            "semantics": "training-level LODO (cross-dataset transfer of the gate)",
            "held_out": d, "n_datasets_remaining": int(g.n_train_datasets.max()),
            "n_groups_remaining": int(len(v)),
            "mean_pct": float(v.mean()), "median_pct": float(np.median(v)),
            "delta_mean_vs_full_pp": float(v.mean() - full),
            "delta_median_vs_full_pp": float("nan"),
            "sign_flips_mean": bool(np.sign(v.mean()) != np.sign(full)),
            "sign_flips_median": False,
            "n_datasets_mean_positive": int(v.mean() > 0),
        })
    return out


# --------------------------------------------------------------------------- #
# 5. 统计单位修正后的检验
# --------------------------------------------------------------------------- #
def min_two_sided_p(n: int) -> dict[str, float]:
    """n 个样本、方向完全一致时能达到的最小双侧 p（精确检验）。"""
    if n < 1:
        return {"wilcoxon": float("nan"), "sign": float("nan")}
    return {"wilcoxon": float(wilcoxon(np.arange(1.0, n + 1)).pvalue),
            "sign": float(binomtest(n, n, 0.5).pvalue)}


def holm(pvals: dict[str, float]) -> dict[str, float]:
    order = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(order)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (k, p) in enumerate(order):
        running = max(running, min(1.0, (m - i) * p))
        adj[k] = running
    return adj


def dataset_level_tests(long: pd.DataFrame, results: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_p: dict[str, float] = {}
    ds_wil_p: dict[str, float] = {}
    ds_sign_p: dict[str, float] = {}

    for mid, eff in EFFECT_BY_ID.items():
        sub = long[long.metric == mid]
        v = sub.value.to_numpy()
        gp = float(wilcoxon(v).pvalue)
        group_p[mid] = gp
        rows.append({
            "test_id": f"{mid}.group_level_wilcoxon", "metric": mid,
            "test": "Wilcoxon signed-rank vs 0 (two-sided)",
            "unit": eff.unit, "unit_level": "decision group (论文现用口径)",
            "n": int(len(v)), "point_estimate_pct": float(v.mean()),
            "median_pct": float(np.median(v)),
            "n_positive": int((v > 0).sum()), "n_negative": int((v < 0).sum()),
            "p_value": gp, "min_possible_two_sided_p": min_two_sided_p(len(v))["wilcoxon"],
            "note": "组内高度依赖：每个 dataset 贡献 15-16 组，n 被夸大约 16 倍",
        })
        per_ds = sub.groupby("dataset").value.mean()
        arr = per_ds.to_numpy()
        wp = float(wilcoxon(arr).pvalue)
        bt = binomtest(int((arr > 0).sum()), len(arr), 0.5)
        ds_wil_p[mid], ds_sign_p[mid] = wp, float(bt.pvalue)
        bounds = min_two_sided_p(len(arr))
        rows.append({
            "test_id": f"{mid}.dataset_level_wilcoxon", "metric": mid,
            "test": "Wilcoxon signed-rank vs 0 (two-sided, exact)",
            "unit": "dataset (先在 dataset 内取组均值)",
            "unit_level": "dataset (cluster, 本脚本新增)",
            "n": int(len(arr)), "point_estimate_pct": float(arr.mean()),
            "median_pct": float(np.median(arr)),
            "n_positive": int((arr > 0).sum()), "n_negative": int((arr < 0).sum()),
            "p_value": wp, "min_possible_two_sided_p": bounds["wilcoxon"],
            "note": "方向即使完全一致，p 也不可能低于 min_possible_two_sided_p",
        })
        rows.append({
            "test_id": f"{mid}.dataset_level_sign", "metric": mid,
            "test": "exact paired sign test vs 0 (two-sided)",
            "unit": "dataset (先在 dataset 内取组均值)",
            "unit_level": "dataset (cluster, 本脚本新增)",
            "n": int(len(arr)), "point_estimate_pct": float(arr.mean()),
            "median_pct": float(np.median(arr)),
            "n_positive": int((arr > 0).sum()), "n_negative": int((arr < 0).sum()),
            "p_value": float(bt.pvalue), "min_possible_two_sided_p": bounds["sign"],
            "note": "符号检验只用方向，不受 dataset 内组数不等的加权影响",
        })
        med_arr = sub.groupby("dataset").value.median().to_numpy()
        rows.append({
            "test_id": f"{mid}.dataset_level_wilcoxon_on_medians", "metric": mid,
            "test": "Wilcoxon signed-rank vs 0 (two-sided, exact)",
            "unit": "dataset (先在 dataset 内取组中位数)",
            "unit_level": "dataset (cluster, 本脚本新增)",
            "n": int(len(med_arr)), "point_estimate_pct": float(med_arr.mean()),
            "median_pct": float(np.median(med_arr)),
            "n_positive": int((med_arr > 0).sum()),
            "n_negative": int((med_arr < 0).sum()),
            "p_value": float(wilcoxon(med_arr).pvalue),
            "min_possible_two_sided_p": bounds["wilcoxon"],
            "note": "对 dataset 内聚合口径的敏感性检查",
        })

    for name, pv in (("group_level_wilcoxon", group_p),
                     ("dataset_level_wilcoxon", ds_wil_p),
                     ("dataset_level_sign", ds_sign_p)):
        fam = {k: v for k, v in pv.items() if k in PRIMARY}
        adj = holm(fam)
        for mid, p_adj in adj.items():
            rows.append({
                "test_id": f"{mid}.{name}_holm", "metric": mid,
                "test": f"Holm within the family of the three primary effects ({name})",
                "unit": "同上", "unit_level": name.split("_")[0],
                "n": int(len(fam)), "point_estimate_pct": float("nan"),
                "median_pct": float("nan"), "n_positive": -1, "n_negative": -1,
                "p_value": float(p_adj),
                "min_possible_two_sided_p": float("nan"),
                "note": "family = a/b/c 三个核心量",
            })

    rows.extend(_friedman_rows(results))
    return pd.DataFrame(rows)


def _friedman_rows(results: Path) -> list[dict[str, Any]]:
    """论文 §4.1 的 Friedman 用 44 个完整块；这里补一个 dataset 为块的版本。"""
    p1 = pd.read_csv(results / "p1_results.csv")
    blk = (p1.groupby(["backbone", "dataset", "pred_len", "plugin"])
           .mse.mean().unstack("plugin"))
    cols = [CONTROL, *ARMS]
    comp = blk[cols].dropna(axis=0, how="any")
    out: list[dict[str, Any]] = []
    res = friedmanchisquare(*[comp[c].to_numpy() for c in cols])
    out.append({
        "test_id": "friedman.block_level_complete44", "metric": "arms_vs_none",
        "test": "Friedman over 4 arms (论文现用口径)",
        "unit": "block = (backbone, dataset, pred_len)", "unit_level": "block",
        "n": int(len(comp)), "point_estimate_pct": float(res.statistic),
        "median_pct": float("nan"),
        "n_positive": -1, "n_negative": -1, "p_value": float(res.pvalue),
        "min_possible_two_sided_p": float("nan"),
        "note": f"44 个完整块只覆盖 {comp.reset_index().dataset.nunique()} 个 dataset，"
                "其中 ETTh1/Weather 各 10 块",
    })
    rel = comp.copy()
    for c in ARMS:
        rel[c] = (comp[CONTROL] - comp[c]) / comp[CONTROL] * 100.0
    rel[CONTROL] = 0.0
    per_ds = rel.reset_index().groupby("dataset")[cols].mean()
    res2 = friedmanchisquare(*[per_ds[c].to_numpy() for c in cols])
    ranks = np.apply_along_axis(rankdata, 1, -per_ds[cols].to_numpy())
    rank_txt = ", ".join(f"{c}={r:.2f}" for c, r in zip(cols, ranks.mean(axis=0)))
    cd = 2.569 * math.sqrt(4 * 5 / (6.0 * len(per_ds)))
    out.append({
        "test_id": "friedman.dataset_level", "metric": "arms_vs_none",
        "test": "Friedman over 4 arms, dataset 为块（本脚本新增）",
        "unit": "dataset (先在 dataset 内取相对增益均值)", "unit_level": "dataset",
        "n": int(len(per_ds)), "point_estimate_pct": float(res2.statistic),
        "median_pct": float("nan"), "n_positive": -1, "n_negative": -1,
        "p_value": float(res2.pvalue), "min_possible_two_sided_p": float("nan"),
        "note": f"平均秩（越小越好）{rank_txt}；k=4,N={len(per_ds)} 的 Nemenyi "
                f"CD={cd:.4f}，比整个秩幅还宽",
    })
    return out


# --------------------------------------------------------------------------- #
# 6. seed 噪声下界 vs selector 增益
# --------------------------------------------------------------------------- #
def _median_or_nan(s: pd.Series) -> float:
    """全 NaN 或空列的中位数返回 NaN，而不是让 numpy 抛 empty-slice 警告。"""
    return float(s.median()) if s.notna().any() else float("nan")


def seed_noise_vs_gain(results: Path, robust: Path,
                       long: pd.DataFrame) -> pd.DataFrame:
    p1 = pd.read_csv(results / "p1_results.csv")
    cell = (p1.groupby(["backbone", "dataset", "pred_len", "plugin"])
            .mse.agg(n_seeds="count", mean_mse="mean", sd_mse=lambda s: s.std(ddof=1))
            .reset_index())
    multi = cell[cell.n_seeds >= 2].copy()
    multi["rel_sd_pct"] = multi.sd_mse / multi.mean_mse * 100.0

    stab = pd.read_csv(robust / "seed_stability.csv")
    parts = stab.block.str.split("|", expand=True)
    stab["dataset"], stab["pred_len"], stab["backbone"] = (
        parts[0], parts[1].astype(int), parts[2])
    none_lvl = (cell[cell.plugin == CONTROL]
                .set_index(["backbone", "dataset", "pred_len"]).mean_mse)
    stab = stab.join(none_lvl.rename("none_mse"),
                     on=["backbone", "dataset", "pred_len"])
    stab["noise_scale_rel_pct"] = stab.noise_scale / stab.none_mse * 100.0

    a = long[long.metric == "a_online_vs_bf"]
    c = long[long.metric == "c_gate_vs_bf"]
    rows: list[dict[str, Any]] = []
    datasets = sorted(p1.dataset.unique())
    nan4 = (float("nan"),) * 4
    for d in datasets + ["ALL"]:
        m = multi if d == "ALL" else multi[multi.dataset == d]
        s = stab if d == "ALL" else stab[stab.dataset == d]
        q25, q50, q75, q90 = (np.percentile(m.rel_sd_pct, [25, 50, 75, 90])
                              if len(m) else nan4)
        row: dict[str, Any] = {
            "dataset": d, "has_seed_replicates": bool(len(m)),
            "n_cells_multi_seed": int(len(m)),
            "n_seeds_per_cell": int(m.n_seeds.max()) if len(m) else 1,
            "rel_sd_median_pct": float(q50), "rel_sd_q25_pct": float(q25),
            "rel_sd_q75_pct": float(q75),
            "rel_sd_iqr_pct": float(q75 - q25), "rel_sd_p90_pct": float(q90),
            "abs_sd_median_mse": float(m.sd_mse.median()) if len(m) else float("nan"),
            "n_block_arm_pairs_seed_stability": int(s.noise_scale.notna().sum()),
            "seed_noise_scale_median_mse": _median_or_nan(s.noise_scale),
            "seed_noise_scale_rel_median_pct": _median_or_nan(s.noise_scale_rel_pct),
        }
        for tag, frame in (("a", a), ("c", c)):
            g = frame if d == "ALL" else frame[frame.dataset == d]
            v = g.value.to_numpy()
            row[f"gain_{tag}_mean_pct"] = float(v.mean())
            row[f"gain_{tag}_median_abs_pct"] = float(np.median(np.abs(v)))
            ok = bool(len(m))
            diff_band = row["seed_noise_scale_rel_median_pct"]
            row[f"gain_{tag}_over_noise_median"] = (
                float(np.median(np.abs(v)) / q50) if ok else float("nan"))
            row[f"frac_{tag}_within_seed_noise_pct"] = (
                float((np.abs(v) < q50).mean() * 100.0) if ok else float("nan"))
            row[f"frac_{tag}_within_seed_noise_p90_pct"] = (
                float((np.abs(v) < q90).mean() * 100.0) if ok else float("nan"))
            row[f"frac_{tag}_within_diff_noise_pct"] = (
                float((np.abs(v) < diff_band).mean() * 100.0)
                if np.isfinite(diff_band) else float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# summary
# --------------------------------------------------------------------------- #
def build_summary(hier: dict[str, Any], eff: pd.DataFrame, boot: pd.DataFrame,
                  lodo: pd.DataFrame, tests: pd.DataFrame, noise: pd.DataFrame,
                  boot_n: int, seed: int, elapsed: float) -> dict[str, Any]:
    def e(mid: str, dataset: str, col: str) -> float:
        return float(eff[(eff.metric == mid) & (eff.dataset == dataset)][col].iloc[0])

    def b(mid: str, stat: str, scope: str, col: str) -> float:
        r = boot[(boot.metric == mid) & (boot.statistic == stat)
                 & (boot.scope == scope)]
        return float(r[col].iloc[0])

    def t(test_id: str, col: str = "p_value") -> float:
        return float(tests[tests.test_id == test_id][col].iloc[0])

    summary: dict[str, Any] = {
        "meta": {"n_boot": boot_n, "seed": seed, "runtime_seconds": round(elapsed, 2),
                 "primary_effects": list(PRIMARY)},
        "unit_hierarchy": {
            "n_datasets": hier["run_level"]["n_datasets"],
            "datasets": hier["run_level"]["datasets"],
            "n_dataset_horizon_pairs": hier["run_level"]["n_dataset_horizon_pairs"],
            "n_backbones": hier["run_level"]["n_backbones"],
            "n_horizons": hier["run_level"]["n_horizons"],
            "n_blocks_phase1": hier["run_level"]["n_blocks_phase1"],
            "n_cells_phase1": hier["run_level"]["n_cells_phase1"],
            "n_runs_phase1": hier["run_level"]["n_runs_phase1"],
            "n_groups_test_side": hier["group_levels"]["test_side_127"]["n"],
            "n_groups_online": hier["group_levels"]["online_feasible_123"]["n"],
            "n_groups_gating": hier["group_levels"]["gating_96"]["n"],
            "groups_per_dataset_test_side":
                hier["group_levels"]["test_side_127"]["groups_by_dataset"],
            "groups_per_dataset_online":
                hier["group_levels"]["online_feasible_123"]["groups_by_dataset"],
            "groups_per_dataset_gating":
                hier["group_levels"]["gating_96"]["groups_by_dataset"],
            "n_clusters_lodo_eligible": hier["cluster_units"]["n_clusters_lodo_eligible"],
        },
        "effects": {}, "cluster_bootstrap": {}, "lodo": {},
        "dataset_level_tests": {}, "seed_noise": {},
    }

    for mid in EFFECT_BY_ID:
        per_ds = eff[(eff.metric == mid) & (eff.dataset != "ALL")]
        best = per_ds.loc[per_ds.mean_pct.idxmax()]
        worst = per_ds.loc[per_ds.mean_pct.idxmin()]
        summary["effects"][mid] = {
            "pooled_mean_pct": e(mid, "ALL", "mean_pct"),
            "pooled_median_pct": e(mid, "ALL", "median_pct"),
            "n_groups": int(e(mid, "ALL", "n_groups")),
            "n_datasets": int(len(per_ds)),
            "n_datasets_mean_positive": int((per_ds.mean_pct > 0).sum()),
            "n_datasets_mean_negative": int((per_ds.mean_pct < 0).sum()),
            "n_groups_positive": int(e(mid, "ALL", "n_groups_positive")),
            "frac_groups_positive_pct": e(mid, "ALL", "frac_groups_positive_pct"),
            "best_dataset": str(best.dataset),
            "best_dataset_mean_pct": float(best.mean_pct),
            "worst_dataset": str(worst.dataset),
            "worst_dataset_mean_pct": float(worst.mean_pct),
            "dataset_mean_range_pp": float(best.mean_pct - worst.mean_pct),
        }
        for stat in ("pooled_mean", "pooled_median", "mean_of_dataset_means"):
            summary["cluster_bootstrap"][f"{mid}.{stat}"] = {
                "n_clusters": int(b(mid, stat, "all_datasets", "n_clusters")),
                "point_pct": b(mid, stat, "all_datasets", "point_pct"),
                "ci95_lo_pct": b(mid, stat, "all_datasets", "ci95_lo_pct"),
                "ci95_hi_pct": b(mid, stat, "all_datasets", "ci95_hi_pct"),
                "ci95_width_pct": b(mid, stat, "all_datasets", "ci95_width_pct"),
                "ci_excludes_zero": bool(boot[
                    (boot.metric == mid) & (boot.statistic == stat)
                    & (boot.scope == "all_datasets")].ci_excludes_zero.iloc[0]),
                "ci95_lo_pct_7clusters": b(mid, stat, "lodo_eligible_7", "ci95_lo_pct"),
                "ci95_hi_pct_7clusters": b(mid, stat, "lodo_eligible_7", "ci95_hi_pct"),
            }
        jack = lodo[(lodo.metric == mid) & (lodo.held_out != "NONE")]
        mi = jack.delta_mean_vs_full_pp.abs().idxmax()
        summary["lodo"][mid] = {
            "full_mean_pct": e(mid, "ALL", "mean_pct"),
            "min_mean_pct": float(jack.mean_pct.min()),
            "max_mean_pct": float(jack.mean_pct.max()),
            "any_sign_flip_mean": bool(jack.sign_flips_mean.any()),
            "any_sign_flip_median": bool(jack.sign_flips_median.any()),
            "most_influential_dataset": str(jack.loc[mi, "held_out"]),
            "most_influential_delta_pp": float(jack.loc[mi, "delta_mean_vs_full_pp"]),
            "most_influential_mean_pct": float(jack.loc[mi, "mean_pct"]),
        }
        summary["dataset_level_tests"][mid] = {
            "group_level_wilcoxon_n": int(
                tests[tests.test_id == f"{mid}.group_level_wilcoxon"].n.iloc[0]),
            "group_level_wilcoxon_p": t(f"{mid}.group_level_wilcoxon"),
            "dataset_level_n": int(
                tests[tests.test_id == f"{mid}.dataset_level_wilcoxon"].n.iloc[0]),
            "dataset_level_wilcoxon_p": t(f"{mid}.dataset_level_wilcoxon"),
            "dataset_level_wilcoxon_p_holm": t(f"{mid}.dataset_level_wilcoxon_holm")
            if mid in PRIMARY else float("nan"),
            "dataset_level_sign_p": t(f"{mid}.dataset_level_sign"),
            "dataset_level_wilcoxon_p_on_medians":
                t(f"{mid}.dataset_level_wilcoxon_on_medians"),
        }

    summary["p_value_floor"] = {
        f"n{n}": min_two_sided_p(n) for n in (5, 6, 7, 8)}
    summary["friedman"] = {
        "block_level_complete44_chi2": t("friedman.block_level_complete44",
                                         "point_estimate_pct"),
        "block_level_complete44_p": t("friedman.block_level_complete44"),
        "block_level_complete44_n": int(
            tests[tests.test_id == "friedman.block_level_complete44"].n.iloc[0]),
        "dataset_level_chi2": t("friedman.dataset_level", "point_estimate_pct"),
        "dataset_level_p": t("friedman.dataset_level"),
        "dataset_level_n": int(
            tests[tests.test_id == "friedman.dataset_level"].n.iloc[0]),
        "dataset_level_note": str(
            tests[tests.test_id == "friedman.dataset_level"].note.iloc[0]),
    }

    allrow = noise[noise.dataset == "ALL"].iloc[0]
    per_ds_noise = noise[(noise.dataset != "ALL") & noise.has_seed_replicates]
    summary["seed_noise"] = {
        "n_cells_multi_seed": int(allrow.n_cells_multi_seed),
        "rel_sd_median_pct": float(allrow.rel_sd_median_pct),
        "rel_sd_q25_pct": float(allrow.rel_sd_q25_pct),
        "rel_sd_q75_pct": float(allrow.rel_sd_q75_pct),
        "rel_sd_iqr_pct": float(allrow.rel_sd_iqr_pct),
        "rel_sd_p90_pct": float(allrow.rel_sd_p90_pct),
        "seed_noise_scale_median_mse": float(allrow.seed_noise_scale_median_mse),
        "seed_noise_scale_rel_median_pct":
            float(allrow.seed_noise_scale_rel_median_pct),
        "gain_a_mean_pct": float(allrow.gain_a_mean_pct),
        "gain_c_mean_pct": float(allrow.gain_c_mean_pct),
        "frac_a_within_seed_noise_pct": float(allrow.frac_a_within_seed_noise_pct),
        "frac_c_within_seed_noise_pct": float(allrow.frac_c_within_seed_noise_pct),
        "frac_a_within_diff_noise_pct": float(allrow.frac_a_within_diff_noise_pct),
        "frac_c_within_diff_noise_pct": float(allrow.frac_c_within_diff_noise_pct),
        "abs_mean_a_over_rel_sd_median": float(
            abs(allrow.gain_a_mean_pct) / allrow.rel_sd_median_pct),
        "abs_mean_c_over_rel_sd_median": float(
            abs(allrow.gain_c_mean_pct) / allrow.rel_sd_median_pct),
        "n_datasets_with_seed_replicates": int(len(per_ds_noise)),
        "datasets_with_seed_replicates": per_ds_noise.dataset.tolist(),
        "n_datasets_with_gain_a_inside_noise": int(
            (per_ds_noise.gain_a_mean_pct.abs()
             < per_ds_noise.rel_sd_median_pct).sum()),
        "n_datasets_with_gain_c_inside_noise": int(
            (per_ds_noise.gain_c_mean_pct.abs()
             < per_ds_noise.rel_sd_median_pct).sum()),
    }
    summary["cannot_compute"] = [
        "reviewer 提到的「只有 7 个 dataset」与产物不符：实测 8 个 dataset；"
        "7 只在 selector LODO 子集成立（ILI 结构性排除）。两种口径都已给出。",
        "dataset 级别的 seed 重复不完整：208/428 个 cell 有 3 个 seed，且只落在 "
        "ETTh1/ETTh2/Exchange/ILI 四个 dataset 上，另外四个 dataset 的 seed 噪声无法计算；"
        "window-level 三个核心量只有 seed 2021，所以无法做 (dataset, seed) 双层 bootstrap。",
        "per-lead-time 的可部署版本需要 validation 侧逐 lead-time 误差，仓库未落盘，"
        "因此本脚本不涉及 §7 的 lead-time 轴。",
    ]
    return summary


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="dataset 为一级统计单位的依赖感知推断")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--robust-dir", default=DEFAULT_ROBUST)
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--boot", type=int, default=10000, help="cluster bootstrap 复制次数")
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--jobs", type=int, default=1, help="bootstrap 并行进程数")
    args = ap.parse_args(argv)

    t0 = time.time()
    root = Path(args.root)
    results, robust = root / args.results_dir, root / args.robust_dir
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    long = load_long(results)
    hier = unit_hierarchy(results, long)
    eff = dataset_effects(long)
    boot = cluster_bootstrap(long, args.boot, args.seed, max(1, args.jobs))
    lodo = lodo_conclusions(long, results)
    tests = dataset_level_tests(long, results)
    noise = seed_noise_vs_gain(results, robust, long)
    summary = build_summary(hier, eff, boot, lodo, tests, noise,
                            args.boot, args.seed, time.time() - t0)

    (out / "unit_hierarchy.json").write_text(
        json.dumps(hier, indent=2, ensure_ascii=False))
    eff.to_csv(out / "dataset_effects.csv", index=False)
    boot.to_csv(out / "cluster_bootstrap.csv", index=False)
    lodo.to_csv(out / "lodo_conclusions.csv", index=False)
    tests.to_csv(out / "dataset_level_tests.csv", index=False)
    noise.to_csv(out / "seed_noise_vs_gain.csv", index=False)
    (out / "cluster_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"[out] {out}")
    for mid in PRIMARY:
        s, cb = summary["effects"][mid], summary["cluster_bootstrap"][f"{mid}.pooled_mean"]
        print(f"  {mid:<18} pooled mean {s['pooled_mean_pct']:+.3f}%  "
              f"cluster-boot 95% [{cb['ci95_lo_pct']:+.3f}, {cb['ci95_hi_pct']:+.3f}]  "
              f"datasets +{s['n_datasets_mean_positive']}/-"
              f"{s['n_datasets_mean_negative']}  "
              f"dataset-level Wilcoxon p="
              f"{summary['dataset_level_tests'][mid]['dataset_level_wilcoxon_p']:.4g}")
    print(f"  p 下界 n=7: {summary['p_value_floor']['n7']}")
    print(f"[time] {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
