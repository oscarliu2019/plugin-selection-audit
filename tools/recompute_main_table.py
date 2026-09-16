#!/usr/bin/env python
"""主表口径复算：4 臂完整块 vs 逐对可配对块。

为什么需要这个脚本
------------------
2026-09-14 的 code review 发现 `src/stats.py` 的 `require_complete=True` 会把
「任一方法缺格」的块从**所有**配对比较里一起删掉，于是某一个方法的缺失会污染
其他方法的配对样本。

本仓库正好踩中：`configs/matrix.yaml` 的 `skip_rules.revin_on_internal_norm_backbones`
只给 PatchTST / iTransformer / TimesNet 保留了 `{ETTh1, Weather} × {96, 720}` 的
revin 验证子集，所以 128 个块里只有 44 个是「四臂俱全」的，而这 44 个里 32 个
（73%）来自 DLinear。fredf 和 san_lite 本身在**全部 128 块**上都有配对数据，
却被 revin 的剪枝规则连带砍到 44 块。

这个脚本直接从 phase-1 的 `results.csv` 原始 844 行复算两种口径，用来独立验证
修复前后的差异，不依赖 `src/stats.py` 自身的实现（避免自证）。

用法
----
    python tools/recompute_main_table.py results/results.csv
"""

from __future__ import annotations

import sys

import pandas as pd
from scipy.stats import wilcoxon

ARMS = ["revin", "san_lite", "fredf"]
CONTROL = "none"


def block_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """块 = backbone × dataset × pred_len；多 seed 取均值。"""
    return df.groupby(["backbone", "dataset", "pred_len", "plugin"]).mse.mean().unstack("plugin")


def summarize(blk: pd.DataFrame, arm: str, complete_only: bool) -> dict:
    sub = blk.dropna(how="any") if complete_only else blk
    s = sub[[arm, CONTROL]].dropna()
    if s.empty:
        return {}
    rel = (s[CONTROL] - s[arm]) / s[CONTROL] * 100
    n_eff = int((rel.abs() > 1e-12).sum())
    p = wilcoxon(s[CONTROL], s[arm], zero_method="wilcox").pvalue if n_eff else float("nan")
    return {
        "arm": arm,
        "n_blocks": len(s),
        "n_eff": n_eff,
        "mean_rel_gain_pct": rel.mean(),
        "median_rel_gain_pct": rel.median(),
        "p_value": p,
    }


def main(path: str) -> None:
    df = pd.read_csv(path)
    blk = block_matrix(df)
    comp = blk.dropna(how="any")
    n_dlinear = int((comp.reset_index().backbone == "DLinear").sum())

    print(f"块总数 {len(blk)}；各臂非缺失块数 {blk.notna().sum().to_dict()}")
    print(f"四臂完整块 {len(comp)}，其中 DLinear {n_dlinear} "
          f"({n_dlinear / max(len(comp), 1):.0%}) —— 这就是选择偏差的来源\n")

    rows = []
    for arm in ARMS:
        for complete_only, label in [(True, "旧口径：4臂完整块"), (False, "新口径：逐对可配对")]:
            r = summarize(blk, arm, complete_only)
            if r:
                rows.append({"口径": label, **r})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    print("\n=== 分骨干（全部可配对块）：解释为什么均值会翻号 ===")
    for arm in ARMS:
        s = blk[[arm, CONTROL]].dropna().reset_index()
        if s.empty:
            continue
        s["rel"] = (s[CONTROL] - s[arm]) / s[CONTROL] * 100
        print(f"\n{arm}:")
        for bb, g in s.groupby("backbone"):
            p = wilcoxon(g[CONTROL], g[arm], zero_method="wilcox").pvalue
            print(f"  {bb:13s} n={len(g):3d} 平均={g.rel.mean():+8.2f}% "
                  f"中位={g.rel.median():+7.2f}% p={p:.4g}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/results.csv")
