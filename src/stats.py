"""统计检验流水线：Wilcoxon / Friedman+Nemenyi / CD 图 / win-tie-loss / Holm。

⚠️ 关于「块 (block) 的定义」——本项目最重要的统计设计决策
--------------------------------------------------------
Nemenyi 临界差 ``CD = q_α · sqrt(k(k+1)/(6N))``，其中 k = 方法数、N = 块数。
实算（α=0.05）：k=5、N=9（只用 9 个数据集当块）时 **CD = 2.03**，而 5 个方法的
平均秩最大间距只有 4.0 —— 也就是说**平均秩必须差 2.03 以上才显著，实际上不可能达到**。
尽调 §6.2 的端到端模拟已验证：N=36 时 Ours 与 Backbone 的秩差 0.917 < CD 1.017，
Nemenyi 判不显著，而同一批数据的 Wilcoxon p=0.0017 显著。

因此本项目：
1. **块 = (数据集 × 预测长度 × 骨干)**，而不是仅数据集。8 数据集 × 4 horizon × 4 骨干
   → **N = 128**（含 Traffic 则 144），此时 k=4 的 CD ≈ 0.41，检验才真正有功效；
2. **k=2 的比较（有插件 vs 无插件）一律用配对 Wilcoxon signed-rank**，不用 Friedman；
3. Friedman + Nemenyi + CD 图**只用于 k≥4 的多方法互比**；
4. 多个「插件 vs 对照」的成对比较用 **Holm** 逐步降序校正（比 Nemenyi 功效更高，
   选择理由在方法节写明，不做事后挑选）。

样本量口径（2026-09-14 code review 后统一，三个数不能混用）
----------------------------------------------------------
- ``n_blocks``：该**一对**方法都有值的块数（逐对 dropna；不同方法可以不同，见修复项 #1）；
- ``n_blocks_complete``：k 个方法全都有值的块数，**只有** Friedman/Nemenyi 用它；
- ``n_eff``：去掉零差值对之后真正进入 signed-rank 的对数，效应量 r = Z/√n_eff 必须用它
  （见修复项 #2）。
另外，分层（按骨干）检验一次报出 backbone × method 个比较，Holm 族必须涵盖全部层，
输出里 ``*_within_backbone`` = 层内探索性校正，``*_across_strata`` = 对外报告口径（修复项 #3）。

用法
----
    python -m src.stats --results results/results.csv --out artifacts/stats
    python -m src.stats --demo         # 用合成矩阵自检整条流水线（无需真实结果）
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range, wilcoxon

ALPHA = 0.05
BLOCK_KEYS = ("dataset", "pred_len", "backbone")


# --------------------------------------------------------------------------- #
# 基础工具
# --------------------------------------------------------------------------- #
def nemenyi_cd(k: int, n: int, alpha: float = ALPHA) -> float:
    """Nemenyi 临界差：CD = q_α · sqrt(k(k+1)/(6N))，q_α = q_studentized/sqrt(2)。"""
    q = float(studentized_range.ppf(1 - alpha, k, np.inf)) / math.sqrt(2.0)
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def cd_table(ks: Iterable[int] = (3, 4, 5, 6, 8, 10),
             ns: Iterable[int] = (9, 18, 36, 72, 128, 144)) -> pd.DataFrame:
    """CD 阈值表：定矩阵规模前必须先看这张表。"""
    ks, ns = list(ks), list(ns)
    return pd.DataFrame(
        [[round(nemenyi_cd(k, n), 3) for n in ns] for k in ks],
        index=pd.Index(ks, name="k_methods"),
        columns=pd.Index(ns, name="N_blocks"),
    )


def holm(pvalues: dict[str, float], alpha: float = ALPHA) -> pd.DataFrame:
    """Holm 逐步降序校正：返回按 p 升序的比较表（含校正阈值与是否拒绝）。"""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    rows, prev_reject = [], True
    running_max = 0.0
    for i, (name, p) in enumerate(items):
        thr = alpha / (m - i)
        running_max = max(running_max, p * (m - i))
        reject = bool(prev_reject and p <= thr)
        prev_reject = reject
        rows.append({"comparison": name, "p_raw": p, "holm_threshold": thr,
                     "p_holm_adjusted": min(running_max, 1.0), "reject_H0": reject})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 结果表 → 块 × 方法矩阵
# --------------------------------------------------------------------------- #
def build_block_matrix(
    results: pd.DataFrame,
    metric: str = "mse",
    method_col: str = "plugin",
    block_keys: Iterable[str] = BLOCK_KEYS,
    require_complete: bool = False,
) -> pd.DataFrame:
    """把长表结果转成「块 × 方法」矩阵（多 seed 先在块内取均值）。

    require_complete 的语义（2026-09-14 code review 修复项 #1）
    ----------------------------------------------------------
    旧实现默认 ``require_complete=True``，对**整张矩阵**做 ``dropna(how="any")``。
    这在本仓库会造成严重污染：``configs/matrix.yaml`` 的
    ``skip_rules.revin_on_internal_norm_backbones`` 只给 PatchTST/iTransformer/TimesNet
    保留了 ``{ETTh1, Weather} × {96, 720}`` 的 revin 验证子集，于是 128 个块里只有 44 个
    「四个方法都全」的完整块（其中 32 个来自 DLinear）。结果是 **revin 的剪枝规则决定了
    fredf / san_lite vs none 的检验样本**：这两个插件本身在全部 128 块上都有配对数据，
    却被砍到 44 块（功效掉 2/3，估计量偏向单一骨干）。
    配对 Wilcoxon 是 k=2 检验，只需要「这一对方法都存在」的块，逐对 dropna 才是正确口径
    （见 ``wilcoxon_pair``）。因此默认值改为 **False**，缺格以 NaN 保留在矩阵里。

    require_complete=True 只留给**真的要求同一块集合**的检验：Friedman 的秩是在每个块内
    跨全部 k 个方法算的，Nemenyi 的 CD 也依赖统一的 N，缺格块必须整块丢弃。
    注意「丢弃缺格块」并不能防止「某插件在难格子上跑挂了偷偷变成优势」——真正的防线是
    逐对报出实际使用的块数与骨干构成（``n_blocks`` / ``blocks_by_group``），
    让读者看到每个 p 值背后到底是哪些块。
    """
    keys = list(block_keys)
    agg = results.groupby(keys + [method_col])[metric].mean().reset_index()
    mat = agg.pivot_table(index=keys, columns=method_col, values=metric)
    if require_complete:
        before = len(mat)
        mat = mat.dropna(axis=0, how="any")
        if before != len(mat):
            print(f"[stats] 丢弃 {before - len(mat)} 个不完整块"
                  f"（仅 Friedman/Nemenyi 这类要求统一块集合的检验才这么做）")
    mat.index = pd.Index(["|".join(str(v) for v in idx) for idx in mat.index], name="block")
    return mat


def _block_group_labels(index: pd.Index, block_keys: Iterable[str] = BLOCK_KEYS,
                        group_key: str = "backbone") -> np.ndarray | None:
    """从 ``dataset|pred_len|backbone`` 形式的块名里取出分组标签（取不到就返回 None）。

    用于在配对检验的输出里记录「这 n 个块的骨干构成」——2026-09-14 code review 修复项 #1：
    论文必须能看出某个 p 值是不是几乎只来自一个骨干。
    """
    keys = list(block_keys)
    if group_key not in keys:
        return None
    pos = keys.index(group_key)
    parts = [str(b).split("|") for b in index]
    if not parts or any(len(p) != len(keys) for p in parts):
        return None            # 不是本项目的块命名（例如 horizon.py 传进来的 cell 矩阵）
    return np.array([p[pos] for p in parts])


# --------------------------------------------------------------------------- #
# k = 2：配对 Wilcoxon
# --------------------------------------------------------------------------- #
def wilcoxon_pair(mat: pd.DataFrame, method: str, control: str,
                  lower_is_better: bool = True, tol: float = 1e-6) -> dict[str, Any]:
    """配对 Wilcoxon signed-rank：method vs control（默认指标越小越好）。

    同时给出 win/tie/loss 与效应量 r = Z/sqrt(n_eff)（Z 由正态近似反推）。

    2026-09-14 code review 修复项 #1：**逐对 dropna**。
    只保留 method 与 control **都有值**的块——k=2 检验不需要第三个方法在场，
    旧口径（整张矩阵 dropna）会让 revin 的剪枝规则决定 fredf/san_lite 的样本。
    ``n_blocks`` = 实际进入本次检验的配对块数，``blocks_by_group`` 记录其骨干构成，
    避免论文里把 ``n_blocks`` 读成矩阵总行数（128）。

    2026-09-14 code review 修复项 #2：**效应量的样本量口径**。
    检验丢弃零差值对（``zero_method="wilcox"``），``stat`` 只由非零差值对构成，
    所以反推 Z 的 mu/sigma 与 r 都必须用去零后的 ``n_eff``；旧实现用含零差值的 n，
    只要有 tie 就系统性放大效应量（实测 rank_agg: tie=33/40 时 r 被抬到 0.86，真值 0.51）。
    这里把 tie 判定与「进入检验的对」用同一个 tol 口径，保证 tie + n_eff == n_blocks。
    """
    pair = mat[[method, control]].dropna(axis=0, how="any")
    a, b = pair[method].to_numpy(float), pair[control].to_numpy(float)
    d = a - b
    n = len(d)                                   # 配对块数（该对方法都存在的块）
    wins = int(np.sum(d < -tol)) if lower_is_better else int(np.sum(d > tol))
    losses = int(np.sum(d > tol)) if lower_is_better else int(np.sum(d < -tol))
    nonzero = np.abs(d) > tol                    # 与 win/tie/loss 完全同一口径
    n_eff = int(nonzero.sum())                   # 真正进入 signed-rank 检验的对数
    ties = n - n_eff
    groups = _block_group_labels(pair.index)
    by_group = ({str(g): int((groups == g).sum()) for g in sorted(set(groups))}
                if groups is not None else {})
    base = {"method": method, "control": control, "n_blocks": n, "n_eff": n_eff,
            "blocks_by_group": ";".join(f"{g}:{c}" for g, c in by_group.items())}
    if n == 0 or n_eff == 0:
        # 全平（或没有可配对的块）：没有可检验的信息，p=1、效应量 0
        return {**base, "stat": float("nan"), "p_value": 1.0,
                "win": wins, "tie": ties, "loss": losses,
                "median_diff": 0.0, "mean_rel_gain_pct": 0.0, "effect_r": 0.0}
    stat, p = wilcoxon(d[nonzero], zero_method="wilcox", alternative="two-sided")
    mu = n_eff * (n_eff + 1) / 4.0
    sigma = math.sqrt(n_eff * (n_eff + 1) * (2 * n_eff + 1) / 24.0)
    z = (float(stat) - mu) / sigma if sigma > 0 else 0.0
    rel = float(np.mean((b - a) / np.abs(b)) * 100.0)  # 正 = 相对对照组降低了多少 % 误差
    return {
        **base,
        "stat": float(stat), "p_value": float(p),
        "win": wins, "tie": ties, "loss": losses,
        "median_diff": float(np.median(d)), "mean_rel_gain_pct": rel,
        "effect_r": abs(z) / math.sqrt(n_eff),
    }


def wilcoxon_vs_control(mat: pd.DataFrame, control: str, alpha: float = ALPHA) -> pd.DataFrame:
    """所有方法逐一 vs 对照组，并做 Holm 校正（族 = 本次调用里的所有方法）。

    注意：每个方法用的是**自己那一对**的可配对块（见 wilcoxon_pair），因此不同方法的
    n_blocks 可以不同；这正是 2026-09-14 code review 修复项 #1 想要的行为。
    """
    methods = [c for c in mat.columns if c != control]
    rows = [wilcoxon_pair(mat, m, control) for m in methods]
    df = pd.DataFrame(rows)
    adj = holm({r["method"]: r["p_value"] for r in rows}, alpha).set_index("comparison")
    df = df.join(adj[["p_holm_adjusted", "reject_H0", "holm_threshold"]], on="method")
    return df.sort_values("p_value")


def win_tie_loss_table(mat: pd.DataFrame, control: str, tol: float = 1e-6,
                       group: pd.Series | None = None) -> pd.DataFrame:
    """win-tie-loss 统计表；给 group（如按骨干/数据集分组的标签）可分层统计。

    2026-09-14 code review 修复项 #1 的连带修正：矩阵现在允许含 NaN（缺格），
    所以每个方法只在「它和对照都有值」的块上计数，n 报的是该对的配对块数——
    否则 NaN 既不算 win 也不算 tie/loss，n 却把缺格块算进去，三者加起来对不上。
    """
    rows = []
    methods = [c for c in mat.columns if c != control]
    groups = {"ALL": np.ones(len(mat), dtype=bool)}
    if group is not None:
        for g in sorted(set(group)):
            groups[str(g)] = (group.to_numpy() == g)
    for gname, mask in groups.items():
        sub = mat[mask]
        for m in methods:
            pair = sub[[m, control]].dropna(axis=0, how="any")
            d = pair[m].to_numpy(float) - pair[control].to_numpy(float)
            if len(d) == 0:
                rows.append({"group": gname, "method": m, "n": 0, "win": 0, "tie": 0,
                             "loss": 0, "win_rate": float("nan"),
                             "mean_rel_gain_pct": float("nan")})
                continue
            rows.append({
                "group": gname, "method": m, "n": len(d),
                "win": int(np.sum(d < -tol)), "tie": int(np.sum(np.abs(d) <= tol)),
                "loss": int(np.sum(d > tol)),
                "win_rate": float(np.mean(d < -tol)),
                "mean_rel_gain_pct": float(
                    np.mean((pair[control] - pair[m]) / pair[control].abs()) * 100),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# k ≥ 4：Friedman + Nemenyi + CD 图
# --------------------------------------------------------------------------- #
def friedman_nemenyi(mat: pd.DataFrame, alpha: float = ALPHA) -> dict[str, Any]:
    """Friedman omnibus + Nemenyi 事后 + 平均秩 + CD 阈值。指标越小越好。

    这是**唯一真的要求「同一块集合」**的检验（秩在块内跨全部 k 个方法算，CD 依赖统一的 N），
    所以必须喂完整矩阵。2026-09-14 code review 修复项 #1：配对 Wilcoxon 已改成逐对 dropna，
    矩阵可能含 NaN，这里显式拦下来，避免 scipy 静默返回 nan 卡方。
    """
    if mat.isna().any().any():
        n_bad = int(mat.isna().any(axis=1).sum())
        raise ValueError(
            f"Friedman/Nemenyi 要求完整矩阵，但有 {n_bad} 个块缺方法；"
            f"请先 build_block_matrix(..., require_complete=True) 或 mat.dropna(how='any')")
    k, n = mat.shape[1], mat.shape[0]
    if k < 3:
        raise ValueError(f"Friedman 需要 k>=3，当前 k={k}；k=2 请用 wilcoxon_pair")
    chi2, p = friedmanchisquare(*[mat[c].to_numpy(float) for c in mat.columns])
    ranks = mat.rank(axis=1, ascending=True)
    avg_rank = ranks.mean().sort_values()
    import scikit_posthocs as sp

    nem = sp.posthoc_nemenyi_friedman(mat.to_numpy(float))
    nem.index = nem.columns = mat.columns
    return {
        "k": k, "n_blocks": n, "chi2": float(chi2), "p_value": float(p),
        "reject_H0": bool(p < alpha), "avg_rank": avg_rank, "nemenyi": nem,
        "cd": nemenyi_cd(k, n, alpha),
    }


def cd_diagram(avg_rank: pd.Series, nemenyi: pd.DataFrame, out_path: str | Path,
               title: str | None = None) -> Path:
    """画 CD 图（scikit-posthocs 的 critical_difference_diagram）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import scikit_posthocs as sp

    plt.figure(figsize=(8, 2.6))
    sp.critical_difference_diagram(avg_rank, nemenyi)
    plt.title(title or "Critical Difference Diagram (Nemenyi)")
    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    return out


# --------------------------------------------------------------------------- #
# 端到端流水线
# --------------------------------------------------------------------------- #
def run_pipeline(results: pd.DataFrame, out_dir: str | Path, control: str = "none",
                 metric: str = "mse", alpha: float = ALPHA) -> dict[str, Any]:
    """从结果长表跑完整套检验，产出 CSV + CD 图，并返回关键数字。

    2026-09-14 code review 修复项 #1：两套矩阵，各归其位——
    - ``mat``（可含缺格）用于所有 k=2 的配对 Wilcoxon / win-tie-loss，逐对取可配对块；
    - ``mat_complete``（整块完整）只给 Friedman + Nemenyi + CD 图。
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mat = build_block_matrix(results, metric=metric, require_complete=False)
    mat_complete = mat.dropna(axis=0, how="any")
    blocks = pd.DataFrame([b.split("|") for b in mat.index], columns=list(BLOCK_KEYS), index=mat.index)

    summary: dict[str, Any] = {"n_blocks": len(mat), "k_methods": mat.shape[1],
                               # 配对检验用 n_blocks（逐对），Friedman 用 n_blocks_complete
                               "n_blocks_complete": len(mat_complete),
                               "block_definition": " × ".join(BLOCK_KEYS)}

    cd_table().to_csv(out / "cd_threshold_table.csv")

    # 1) k=2：每个插件 vs 对照（逐对可配对块，n_blocks/n_eff/blocks_by_group 逐行报出）
    wil = wilcoxon_vs_control(mat, control, alpha)
    wil.to_csv(out / "wilcoxon_vs_control.csv", index=False)
    summary["wilcoxon"] = wil.to_dict("records")

    # 2) win-tie-loss（整体 + 按骨干分层：DLinear 无内置归一化，必须分层看）
    wtl = win_tie_loss_table(mat, control, group=blocks["backbone"])
    wtl.to_csv(out / "win_tie_loss.csv", index=False)

    # 3) k>=3：Friedman + Nemenyi + CD 图（只能用完整块）
    if mat_complete.shape[1] >= 3 and len(mat_complete) >= 3:
        fn = friedman_nemenyi(mat_complete, alpha)
        fn["avg_rank"].to_csv(out / "avg_ranks.csv", header=["avg_rank"])
        fn["nemenyi"].to_csv(out / "nemenyi_pvalues.csv")
        png = cd_diagram(fn["avg_rank"], fn["nemenyi"], out / "cd_diagram.png",
                         f"CD Diagram (Nemenyi, α={alpha}, N={fn['n_blocks']}, k={fn['k']})")
        summary["friedman"] = {kk: fn[kk] for kk in ("k", "n_blocks", "chi2", "p_value", "reject_H0", "cd")}
        summary["avg_rank"] = fn["avg_rank"].to_dict()
        summary["cd_diagram"] = str(png)

    # 4) 分层检验：每个骨干单独做一次（回答「插件的条件性」）
    #    2026-09-14 code review 修复项 #3：这张表同时报出 backbone × method 共 12 个比较，
    #    但 wilcoxon_vs_control 的 Holm 族只有「单个骨干内的 3 个插件」（m=3），
    #    跨骨干的层完全没有联合校正，而 README 要求这些分层 p 值写进论文。
    #    因此：层内校正列显式改名为 *_within_backbone（探索性参考），
    #    另加一列对全部 12 个原始 p 值做的族内 Holm（*_across_strata），作为对外报告的口径。
    strat = []
    for bk, idx in blocks.groupby("backbone").groups.items():
        sub = mat.loc[list(idx)]
        if len(sub) < 6:
            continue
        for r in wilcoxon_vs_control(sub, control, alpha).to_dict("records"):
            r = dict(r)
            r["p_holm_within_backbone"] = r.pop("p_holm_adjusted")
            r["reject_H0_within_backbone"] = r.pop("reject_H0")
            r["holm_threshold_within_backbone"] = r.pop("holm_threshold")
            strat.append({"backbone": bk, **r})
    if strat:
        sdf = pd.DataFrame(strat)
        keys = [f"{r['backbone']}|{r['method']}" for r in strat]
        fam = holm({k: r["p_value"] for k, r in zip(keys, strat)}, alpha).set_index("comparison")
        sdf["stratum_comparison"] = keys
        sdf = sdf.join(fam[["p_holm_adjusted", "reject_H0", "holm_threshold"]]
                       .rename(columns={"p_holm_adjusted": "p_holm_across_strata",
                                        "reject_H0": "reject_H0_across_strata",
                                        "holm_threshold": "holm_threshold_across_strata"}),
                       on="stratum_comparison")
        sdf["holm_family_size_across_strata"] = len(strat)
        sdf.to_csv(out / "wilcoxon_by_backbone.csv", index=False)
        summary["wilcoxon_by_backbone"] = sdf.to_dict("records")
        summary["holm_family_size_across_strata"] = len(strat)

    print(f"[stats] 块定义 = {summary['block_definition']}，N={summary['n_blocks']}"
          f"（完整块 {summary['n_blocks_complete']}），k={summary['k_methods']}")
    print(f"[stats] 输出目录: {out}")
    return summary


# --------------------------------------------------------------------------- #
# 自检 demo
# --------------------------------------------------------------------------- #
def demo_results(seed: int = 26) -> pd.DataFrame:
    """构造一个「插件增益条件依赖于骨干」的合成结果长表（正是本文要检出的现象）。"""
    rng = np.random.default_rng(seed)
    datasets = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather", "Electricity", "Exchange", "ILI"]
    horizons = [96, 192, 336, 720]
    backbones = ["DLinear", "PatchTST", "iTransformer", "TimesNet"]
    plugins = ["none", "revin", "san_lite", "fredf"]
    rows = []
    for ds in datasets:
        base_ds = rng.uniform(0.25, 0.6)
        for h in horizons:
            for bk in backbones:
                base = base_ds * (1 + 0.15 * horizons.index(h)) * rng.uniform(0.95, 1.05)
                for pl in plugins:
                    gain = 0.0
                    if pl == "revin":
                        gain = -0.03 if bk == "DLinear" else -0.001
                    elif pl == "san_lite":
                        gain = -0.02 if ds in ("Exchange", "ETTh2") else 0.005
                    elif pl == "fredf":
                        gain = -0.015 if ds in ("ETTm1", "Weather", "Electricity") else 0.002
                    rows.append({"dataset": ds, "pred_len": h, "backbone": bk, "plugin": pl,
                                 "seed": 2021, "mse": base + gain + rng.normal(0, 0.01),
                                 "mae": base * 0.8 + gain + rng.normal(0, 0.01)})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="统计检验流水线")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default="artifacts/stats")
    ap.add_argument("--control", default="none")
    ap.add_argument("--metric", default="mse")
    ap.add_argument("--demo", action="store_true", help="用合成结果自检")
    args = ap.parse_args()

    if args.demo:
        df = demo_results()
        print(f"[stats] demo 结果表 shape={df.shape}")
    else:
        df = pd.read_csv(args.results)

    print("\nCD 阈值表（α=0.05；注意 k=5,N=9 时 CD=2.03，几乎不可能显著）:")
    print(cd_table().to_string())

    s = run_pipeline(df, args.out, args.control, args.metric)
    print("\n=== Wilcoxon（各插件 vs 对照，Holm 校正）===")
    # n_blocks = 该对实际可配对的块数（可能小于矩阵总行数）；n_eff = 去零差值后进入检验的对数
    print(pd.DataFrame(s["wilcoxon"])[
        ["method", "n_blocks", "n_eff", "win", "tie", "loss", "mean_rel_gain_pct", "p_value",
         "p_holm_adjusted", "reject_H0", "effect_r", "blocks_by_group"]
    ].to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    if "friedman" in s:
        print("\n=== Friedman + Nemenyi（仅完整块）===")
        print(s["friedman"])
        print("平均秩:", {k: round(v, 3) for k, v in s["avg_rank"].items()})
    if "wilcoxon_by_backbone" in s:
        print(f"\n=== 分层 Wilcoxon（族规模 = {s['holm_family_size_across_strata']} 个"
              f" backbone × method 比较；层内校正仅供探索）===")
        print(pd.DataFrame(s["wilcoxon_by_backbone"])[
            ["backbone", "method", "n_blocks", "n_eff", "p_value",
             "p_holm_within_backbone", "p_holm_across_strata", "reject_H0_across_strata"]
        ].to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
