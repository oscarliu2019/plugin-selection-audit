"""horizon-aware 早停 / checkpoint 选择准则（附加贡献，**零算力增量**）。

动机
----
TSLib 的默认协议是「单一验证 MSE + patience=3」选 checkpoint。但验证 MSE 是
**所有 horizon 步的平均**，短 horizon 段先收敛、长 horizon 段还在改善时，
平均值会过早触发早停/选到偏向短 horizon 的 checkpoint。综述也批评过
「固定窗口平均指标掩盖长 horizon 退化」。

零算力增量的原因
----------------
所有准则都只是**同一批训练日志的不同读法**：`runs/<cell_id>/epochs.jsonl` 里已经
记录了每 epoch 的验证/测试分段 MSE（runner.StreamingMetrics 按 pred_len 均分 4 段）。
因此「N 骨干 × M 数据集 × K 准则」的成本 = 「N × M × 1 次训练」。

准则一览（全部只看**验证集**，测试集仅用于事后评估）
- `val_mse`          : TSLib 默认 = argmin 总体验证 MSE（基线）
- `last`             : 最后一个 epoch（没有早停）
- `rank_agg`         : 各 horizon 段内对 epoch 排名后取平均秩，argmin 平均秩
- `long_seg`         : argmin 最后一段（最长 horizon）的验证 MSE
- `worst_seg`        : argmin 各段中的最大值（minimax）
- `slope_penalized`  : argmin 验证 MSE × (1 + 正的段间退化斜率)

另有 `oracle_test`：直接看**测试集**选 epoch，是上界不是准则，只在描述性对比表里报
「距上界的差距」，**不进入配对 Wilcoxon 的假设族**（2026-09-14 code review 修复项 #4）。
5 个真准则 vs 基线的 p 值统一做 Holm 校正，见 `run()`。

用法
----
    python -m src.horizon --runs-dir runs --out artifacts/horizon
    python -m src.horizon --demo            # 用合成训练日志自检
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

Criterion = Callable[[pd.DataFrame], int]


# --------------------------------------------------------------------------- #
# 日志载入
# --------------------------------------------------------------------------- #
def load_epoch_logs(runs_dir: str | Path) -> pd.DataFrame:
    """读取 runs/<cell_id>/epochs.jsonl，展开分段指标为列，返回长表。"""
    rows: list[dict[str, Any]] = []
    for f in sorted(Path(runs_dir).glob("*/epochs.jsonl")):
        cell_id = f.parent.name
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            rec = {"cell_id": cell_id, "epoch": int(r["epoch"]),
                   "val_mse": float(r["val_mse"]), "test_mse": float(r["test_mse"])}
            for tag in ("val", "test"):
                for i, v in enumerate(r.get(f"{tag}_seg_mse", []) or []):
                    rec[f"{tag}_seg{i+1}_mse"] = float(v)
            rows.append(rec)
    return pd.DataFrame(rows)


def _seg_cols(df: pd.DataFrame, tag: str = "val") -> list[str]:
    return sorted([c for c in df.columns if c.startswith(f"{tag}_seg") and c.endswith("_mse")])


# --------------------------------------------------------------------------- #
# 准则
# --------------------------------------------------------------------------- #
def crit_val_mse(g: pd.DataFrame) -> int:
    """TSLib 默认：整体验证 MSE 最小的 epoch。"""
    return int(g.loc[g["val_mse"].idxmin(), "epoch"])


def crit_last(g: pd.DataFrame) -> int:
    """不早停，取最后一个 epoch（用于量化早停本身的价值）。"""
    return int(g["epoch"].max())


def crit_rank_agg(g: pd.DataFrame) -> int:
    """各 horizon 段内分别对 epoch 排名，再取平均秩最小者。

    直觉：让每个 horizon 段「一票」，避免短 horizon 段因数值量级小而被平均值淹没。
    """
    cols = _seg_cols(g, "val")
    if not cols:
        return crit_val_mse(g)
    ranks = g[cols].rank(axis=0, ascending=True)
    return int(g.loc[ranks.mean(axis=1).idxmin(), "epoch"])


def crit_long_seg(g: pd.DataFrame) -> int:
    """只看最后一段（最长 horizon）的验证误差。"""
    cols = _seg_cols(g, "val")
    if not cols:
        return crit_val_mse(g)
    return int(g.loc[g[cols[-1]].idxmin(), "epoch"])


def crit_worst_seg(g: pd.DataFrame) -> int:
    """minimax：各 horizon 段中最差的那一段最小化。"""
    cols = _seg_cols(g, "val")
    if not cols:
        return crit_val_mse(g)
    return int(g.loc[g[cols].max(axis=1).idxmin(), "epoch"])


def crit_slope_penalized(g: pd.DataFrame, lam: float = 1.0) -> int:
    """验证 MSE × (1 + λ·正的段间退化斜率)：惩罚「越到长 horizon 越糟」的 checkpoint。

    斜率用「归一化段 MSE 对段索引」的最小二乘斜率，故与量级无关，无需调参。
    """
    cols = _seg_cols(g, "val")
    if len(cols) < 2:
        return crit_val_mse(g)
    seg = g[cols].to_numpy(float)
    norm = seg / np.maximum(seg.mean(axis=1, keepdims=True), 1e-12)
    x = np.arange(seg.shape[1], dtype=float)
    x = x - x.mean()
    slope = (norm * x).sum(axis=1) / (x**2).sum()
    score = g["val_mse"].to_numpy(float) * (1.0 + lam * np.clip(slope, 0, None))
    return int(g["epoch"].to_numpy()[int(np.argmin(score))])


def crit_oracle(g: pd.DataFrame) -> int:
    """上界：直接看测试集（仅用于给出「最好能有多好」，不是可用准则）。"""
    return int(g.loc[g["test_mse"].idxmin(), "epoch"])


CRITERIA: dict[str, Criterion] = {
    "val_mse": crit_val_mse,
    "last": crit_last,
    "rank_agg": crit_rank_agg,
    "long_seg": crit_long_seg,
    "worst_seg": crit_worst_seg,
    "slope_penalized": crit_slope_penalized,
    "oracle_test": crit_oracle,
}

#: 上界型「准则」：直接看测试集选 epoch，不是可用方法，**不参与假设检验**。
#: 2026-09-14 code review 修复项 #4：把它当假设检验既无意义（H0「偷看测试集不更好」没人关心），
#: 又白白把 Holm 族规模从 5 抬到 6、削掉真正准则的功效。它只在描述性表里报「距上界的差距」。
ORACLE_CRITERIA = ("oracle_test",)


def hypothesis_criteria(baseline: str = "val_mse") -> list[str]:
    """真正进入假设检验的准则列表（排除基线自身与 oracle 上界）。"""
    return [c for c in CRITERIA if c != baseline and c not in ORACLE_CRITERIA]


# --------------------------------------------------------------------------- #
# 评估
# --------------------------------------------------------------------------- #
def select_epochs(logs: pd.DataFrame) -> pd.DataFrame:
    """对每个 cell × 每个准则给出选中的 epoch 与其测试 MSE。"""
    rows = []
    for cell_id, g in logs.groupby("cell_id"):
        g = g.sort_values("epoch").reset_index(drop=True)
        rec: dict[str, Any] = {"cell_id": cell_id, "n_epochs": len(g)}
        for name, fn in CRITERIA.items():
            ep = fn(g)
            rec[f"{name}_epoch"] = ep
            rec[f"{name}_test_mse"] = float(g.loc[g["epoch"] == ep, "test_mse"].iloc[0])
        rows.append(rec)
    return pd.DataFrame(rows)


def compare_criteria(sel: pd.DataFrame, baseline: str = "val_mse") -> pd.DataFrame:
    """准则对比表：平均测试 MSE、相对基线的相对增益、win/tie/loss、与 oracle 的差距。"""
    base = sel[f"{baseline}_test_mse"].to_numpy(float)
    oracle = sel["oracle_test_test_mse"].to_numpy(float)
    rows = []
    for name in CRITERIA:
        v = sel[f"{name}_test_mse"].to_numpy(float)
        d = v - base
        rows.append({
            "criterion": name,
            "mean_test_mse": float(v.mean()),
            "mean_rel_gain_vs_baseline_pct": float(np.mean((base - v) / base) * 100),
            "win": int(np.sum(d < -1e-9)), "tie": int(np.sum(np.abs(d) <= 1e-9)),
            "loss": int(np.sum(d > 1e-9)),
            "mean_gap_to_oracle_pct": float(np.mean((v - oracle) / base) * 100),
            "mean_epoch": float(sel[f"{name}_epoch"].mean()),
            "n_cells": int(len(v)),
        })
    return pd.DataFrame(rows).sort_values("mean_test_mse")


def run(logs: pd.DataFrame, out_dir: str | Path, baseline: str = "val_mse") -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sel = select_epochs(logs)
    cmp = compare_criteria(sel, baseline)
    sel.to_csv(out / "selected_epochs.csv", index=False)
    cmp.to_csv(out / "criteria_comparison.csv", index=False)

    # 与基线的配对 Wilcoxon（k=2 场景，遵循 stats.py 的规范）
    # 2026-09-14 code review 修复项 #4：这里一次性对同一批 cell 报出多个「准则 vs 基线」检验，
    # 旧实现只写原始 p_value —— 5 次检验下 α=0.05 的族错误率约 1-0.95^5 ≈ 0.23（含 oracle 6 次 ≈ 0.26）。
    # stats.py 的 docstring 早已声明「多个成对比较用 Holm 校正」，这里必须照做；
    # 同时剔除 oracle_test（上界，不是假设检验的对象，见 ORACLE_CRITERIA）。
    from .stats import holm, wilcoxon_pair

    mat = sel[[f"{c}_test_mse" for c in CRITERIA]].rename(
        columns={f"{c}_test_mse": c for c in CRITERIA})
    tested = hypothesis_criteria(baseline)
    tests = [wilcoxon_pair(mat, c, baseline) for c in tested]
    adj = holm({r["method"]: r["p_value"] for r in tests}).set_index("comparison")
    for r in tests:                                   # 族 = 这 5 个准则，逐行写回校正结果
        r["holm_family_size"] = len(tests)
        r["holm_threshold"] = float(adj.loc[r["method"], "holm_threshold"])
        r["p_holm_adjusted"] = float(adj.loc[r["method"], "p_holm_adjusted"])
        r["reject_H0"] = bool(adj.loc[r["method"], "reject_H0"])
    pd.DataFrame(tests).to_csv(out / "criteria_wilcoxon.csv", index=False)
    print(f"[horizon] {len(sel)} 个 cell，输出 -> {out}")
    return {"comparison": cmp.to_dict("records"), "wilcoxon": tests}


def main() -> None:
    ap = argparse.ArgumentParser(description="horizon-aware checkpoint 选择准则评估")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--logs-csv", default=None, help="直接给一个长表 CSV（列同 load_epoch_logs 输出）")
    ap.add_argument("--out", default="artifacts/horizon")
    ap.add_argument("--baseline", default="val_mse")
    ap.add_argument("--demo", action="store_true", help="用合成训练日志自检")
    args = ap.parse_args()

    if args.demo:
        from .synth import make_epoch_logs

        logs = make_epoch_logs()
        print(f"[horizon] demo 日志 shape={logs.shape}")
    elif args.logs_csv:
        logs = pd.read_csv(args.logs_csv)
    else:
        logs = load_epoch_logs(args.runs_dir)
        if logs.empty:
            print(f"[horizon] {args.runs_dir} 下没有 epochs.jsonl，先跑训练或用 --demo")
            return

    s = run(logs, args.out, args.baseline)
    print("\n=== 准则对比（测试 MSE 越小越好；oracle_test 是上界，不是可用准则）===")
    print(pd.DataFrame(s["comparison"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== 与基线的配对 Wilcoxon（Holm 校正，族 = 5 个准则；oracle_test 是上界，不参与检验）===")
    print(pd.DataFrame(s["wilcoxon"])[["method", "n_blocks", "n_eff", "win", "tie", "loss",
                                       "mean_rel_gain_pct", "p_value", "p_holm_adjusted",
                                       "reject_H0"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
