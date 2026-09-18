#!/usr/bin/env python
"""论文头条数字的机器校验：把 `paper/main.tex` 里印出来的每个数，重新从落盘产物算一遍。

为什么需要这个脚本
------------------
这篇论文的全部主张都是"某个数比另一个数小"，所以一个手抄错的小数点就足以让整篇
结论不自洽。实际发生过：正文 §12 把「test 内部切分门控 vs 固定臂」写成 `-0.55`，
而表 13 与 §9.2 是 `-0.53`——同一个量在同一篇论文里出现两个值。这类漂移不是靠通读
能稳定抓到的，必须机器化。

它做两件事，两件都必须过
--------------------------
1. **数值可复现**：每条 claim 带一个 `fn`，直接从 `monday_final/` 或
   `artifacts/robustness/` 的产物重算，和 claim 里登记的期望值比对（带容差）。
   这条守的是「论文里的数 == 产物里的数」。
2. **论文里真的印了这个数**：把 `paper/main.tex` 与 `paper/tables_robust/*.tex`
   归一化后做字面搜索。这条守的是「产物变了但论文忘了改」以及「论文改了但产物没跟上」。

只有第 1 条会漏掉「正文和表格互相不一致」这种情形（两处都不等于产物时才会报），
只有第 2 条会漏掉「论文和产物一起错」。两条一起才闭环。

用法
----
    python tools/verify_paper_numbers.py                     # 全量校验，失败则退出码 1
    python tools/verify_paper_numbers.py --only oracle       # 只跑 id 含 "oracle" 的
    python tools/verify_paper_numbers.py --json out.json     # 机器可读结果
    python tools/verify_paper_numbers.py --list              # 只列 claim 清单

投稿前请把它跑成绿的，并把输出贴进 rebuttal——「本文没有一个数字是手打的」这句话
需要有东西兜着。
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, spearmanr, wilcoxon

ROOT = Path(__file__).resolve().parents[1]

#: 落盘产物的默认位置。`--results-dir` / `--robust-dir` 可覆盖。
DEFAULT_RESULTS = "monday_final"
DEFAULT_ROBUST = "artifacts/robustness"

ARMS = ("revin", "san_lite", "fredf")
CONTROL = "none"


# --------------------------------------------------------------------------- #
# 产物加载
# --------------------------------------------------------------------------- #
class Artefacts:
    """惰性加载全部落盘产物。

    惰性是必要的：`--only` 常常只需要一两个文件，而 `window_features.parquet`
    这类产物加载一次要几秒。用 `cached_property` 保证同一次运行内只读一次。
    """

    def __init__(self, root: Path, results_dir: str = DEFAULT_RESULTS,
                 robust_dir: str = DEFAULT_ROBUST) -> None:
        self.root = root
        self.results = root / results_dir
        self.robust = root / robust_dir

    # ---- 原始结果 -------------------------------------------------------- #
    @cached_property
    def wr(self) -> dict[str, Any]:
        """`weekend_report.json`：门控/在线/半衰期/seed/FreDF 的汇总口径。"""
        return json.loads((self.results / "weekend_report.json").read_text())

    @cached_property
    def p1(self) -> pd.DataFrame:
        return pd.read_csv(self.results / "p1_results.csv")

    @cached_property
    def p3(self) -> pd.DataFrame:
        return pd.read_csv(self.results / "p3" / "fredf_alpha.csv")

    @cached_property
    def oracle_audit(self) -> pd.DataFrame:
        """逐决策组的 argmin 身份持续性审计（127 组），半衰期结论的原始表。"""
        return pd.read_csv(self.results / "p2" / "selector_window_oracle_audit.csv")

    @cached_property
    def strata(self) -> pd.DataFrame:
        return pd.read_csv(self.results / "paper" / "audit_strata.csv")

    def splitfix(self, source: str) -> pd.DataFrame:
        return pd.read_csv(self.results / "paper" / "splitfix"
                           / f"selector_window_split_{source}_splitfix.csv")

    @cached_property
    def coverage_gap(self) -> pd.DataFrame:
        return pd.read_csv(self.results / "paper" / "splitfix"
                           / "selector_window_coverage_gap_splitfix.csv")

    # ---- robustness 增量审计 --------------------------------------------- #
    @cached_property
    def rb_metric(self) -> pd.DataFrame:
        return pd.read_csv(self.robust / "metric_robustness.csv")

    @cached_property
    def rb_leadtime(self) -> pd.DataFrame:
        return pd.read_csv(self.robust / "leadtime_argmin_stability.csv")

    @cached_property
    def rb_agree(self) -> pd.DataFrame:
        return pd.read_csv(self.robust / "metric_agreement.csv")

    @cached_property
    def rb_cost(self) -> pd.DataFrame:
        return pd.read_csv(self.robust / "cost_benefit_by_method.csv")

    @cached_property
    def rb_summary(self) -> dict[str, Any]:
        return json.loads((self.robust / "robustness_summary.json").read_text())

    # ---- 取值助手 -------------------------------------------------------- #
    def j(self, dotted: str) -> float:
        """按点分路径从 `weekend_report.json` 取一个标量。"""
        node: Any = self.wr
        for part in dotted.split("."):
            node = node[part]
        return float(node)

    def rb(self, metric: str, method: str, col: str) -> float:
        """从 `metric_robustness.csv` 取 (metric, arm) 行的某一列。"""
        row = self.rb_metric[(self.rb_metric.metric == metric)
                             & (self.rb_metric.method == method)]
        if len(row) != 1:
            raise KeyError(f"metric_robustness 里 ({metric}, {method}) 命中 {len(row)} 行")
        return float(row.iloc[0][col])

    def rbs(self, dotted: str) -> float:
        node: Any = self.rb_summary
        for part in dotted.split("."):
            node = node[part]
        return float(node)

    def cost(self, method: str, col: str) -> float:
        row = self.rb_cost[self.rb_cost.method == method]
        return float(row.iloc[0][col])

    # ---- 需要真算的口径 -------------------------------------------------- #
    @cached_property
    def block_matrix(self) -> pd.DataFrame:
        """块 = backbone × dataset × pred_len，多 seed 取均值，列 = 臂。"""
        return (self.p1.groupby(["backbone", "dataset", "pred_len", "plugin"])
                .mse.mean().unstack("plugin"))

    def main_table(self, arm: str, complete_only: bool) -> dict[str, float]:
        """复算主表两种口径。

        故意不调用 `src/stats.py`：那正是被审计的实现，用它验证它等于自证。
        """
        blk = self.block_matrix
        cols = [CONTROL, *ARMS] if complete_only else [CONTROL, arm]
        sub = blk[[c for c in cols if c in blk.columns]].dropna(axis=0, how="any")
        rel = (sub[CONTROL] - sub[arm]) / sub[CONTROL] * 100.0
        return {
            "n": float(len(sub)),
            "mean": float(rel.mean()),
            "median": float(rel.median()),
            "p": float(wilcoxon(sub[arm], sub[CONTROL]).pvalue),
        }

    def per_backbone(self, arm: str, backbone: str) -> dict[str, float]:
        blk = self.block_matrix.reset_index()
        sub = blk[blk.backbone == backbone][[CONTROL, arm]].dropna(axis=0, how="any")
        rel = (sub[CONTROL] - sub[arm]) / sub[CONTROL] * 100.0
        return {
            "n": float(len(sub)),
            "mean": float(rel.mean()),
            "p": float(wilcoxon(sub[arm], sub[CONTROL]).pvalue),
        }

    # ---- §3 的 revin 剪枝验证子集 ---------------------------------------- #
    @cached_property
    def revin_paired(self) -> pd.DataFrame:
        """逐 (backbone, dataset, pred_len, seed) 配对的 revin vs none 相对增益。

        必须按 seed 配对：DLinear 的 cheap 层有 3 seed 而验证子集只有 seed 2021，
        先聚合再配对会把不同 seed 的 mse 混在一起，得到的「no-op」结论就是假的。
        """
        key = ["backbone", "dataset", "pred_len", "seed"]
        r = self.p1[self.p1.plugin == "revin"].set_index(key).mse
        n = self.p1[self.p1.plugin == CONTROL].set_index(key).mse
        j = pd.concat({"revin": r, "none": n}, axis=1).dropna()
        j["gain"] = (j["none"] - j["revin"]) / j["none"] * 100.0
        return j

    def revin_subset(self, col: str, internal_norm: bool = True) -> float:
        """`internal_norm=True` 取内置归一化的三个骨干（验证子集），False 取 DLinear。"""
        j = self.revin_paired
        bb = j.index.get_level_values("backbone")
        sub = j[bb != "DLinear"] if internal_norm else j[bb == "DLinear"]
        if col == "n":
            return float(len(sub))
        if col == "median":
            return float(sub.gain.median())
        if col == "mean":
            return float(sub.gain.mean())
        if col == "wilcoxon_p":
            return float(wilcoxon(sub.gain.to_numpy()).pvalue)
        raise KeyError(col)

    def revin_cell(self, backbone: str, dataset: str, pred_len: int) -> float:
        j = self.revin_paired.reset_index()
        row = j[(j.backbone == backbone) & (j.dataset == dataset)
                & (j.pred_len == pred_len)]
        if len(row) != 1:
            raise KeyError(f"revin 配对表里 {(backbone, dataset, pred_len)} 命中 "
                           f"{len(row)} 行")
        return float(row.iloc[0].gain)

    def p1_all_ok(self) -> float:
        return float((self.p1.status == "ok").sum())

    # ---- Friedman / Nemenyi -------------------------------------------- #
    def friedman(self, what: str) -> float:
        """完整块子集上的 Friedman 检验与平均秩。

        `scipy` 只给统计量和 p，平均秩要自己算；两者都不走 `src/stats.py`。
        """
        cols = [CONTROL, *ARMS]
        sub = self.block_matrix[cols].dropna(axis=0, how="any")
        if what == "n":
            return float(len(sub))
        if what in ("chi2", "p"):
            res = friedmanchisquare(*[sub[c].to_numpy() for c in cols])
            return float(res.statistic if what == "chi2" else res.pvalue)
        if what.startswith("rank_"):
            arm = what[len("rank_"):]
            ranks = np.apply_along_axis(rankdata, 1, sub[cols].to_numpy())
            return float(ranks.mean(axis=0)[cols.index(arm)])
        raise KeyError(what)

    @staticmethod
    def nemenyi_cd(k: int, n: int) -> float:
        """Demsar (2006) 式 (7)。q_0.05 表在 k<=5 上写死，避免依赖被审计的实现。"""
        q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}[k]
        return float(q05 * math.sqrt(k * (k + 1) / (6.0 * n)))

    # ---- §8.2 的 horizon 标度 ------------------------------------------- #
    @cached_property
    def horizon_medians(self) -> pd.DataFrame:
        return (self.oracle_audit
                .groupby("pred_len")
                .agg(hl=("persist_half_life_windows", "median"),
                     ratio=("half_life_over_H", "median")))

    def horizon_scaling(self, what: str) -> float:
        g = self.horizon_medians
        if what == "spearman_hl_medians":
            return float(spearmanr(g.index.to_numpy(), g.hl.to_numpy()).statistic)
        if what == "spearman_ratio_medians":
            return float(spearmanr(g.index.to_numpy(), g.ratio.to_numpy()).statistic)
        if what == "spearman_hl_groups":
            return float(spearmanr(self.oracle_audit.pred_len,
                                   self.oracle_audit.persist_half_life_windows).statistic)
        if what == "spearman_ratio_groups":
            return float(spearmanr(self.oracle_audit.pred_len,
                                   self.oracle_audit.half_life_over_H).statistic)
        if what == "hl_growth":
            return float(g.hl.loc[720] / g.hl.loc[24])
        if what.startswith("hl_at_"):
            return float(g.hl.loc[int(what[len("hl_at_"):])])
        if what.startswith("ratio_at_"):
            return float(g.ratio.loc[int(what[len("ratio_at_"):])])
        raise KeyError(what)

    def split_gate_realized(self, source: str, what: str) -> float:
        """oracle 实现比例的三种口径。重尾使得均值/中位数/比值之比差一个量级。"""
        d = self.splitfix(source)
        s = d.oracle_realized_pct.dropna()
        if what == "median":
            return float(s.median())
        if what == "mean":
            return float(s.mean())
        if what == "min":
            return float(s.min())
        if what == "q25":
            return float(s.quantile(0.25))
        if what == "q75":
            return float(s.quantile(0.75))
        if what == "ratio_of_means":
            return float(d.gain_gate_vs_none.mean() / d.gain_oracle_vs_none.mean()
                         * 100.0)
        raise KeyError(what)

    @cached_property
    def online_raw(self) -> pd.DataFrame:
        return pd.read_csv(self.results / "p2" / "selector_window_online.csv")

    @cached_property
    def lodo(self) -> pd.DataFrame:
        return pd.read_csv(self.results / "p2" / "selector_window_lodo.csv")

    def split_gate(self, source: str, col: str, how: str = "mean") -> float:
        d = self.splitfix(source)
        if col == "win_pct":
            return float((d.gain_gate_vs_best_fixed > 0).mean() * 100.0)
        if col == "gate_beats_majority_pct":
            return float((d.gate_acc > d.majority_acc).mean() * 100.0)
        if col == "wilcoxon_p":
            return float(wilcoxon(d.gain_gate_vs_best_fixed.dropna().to_numpy()).pvalue)
        s = d[col]
        return float(s.median() if how == "median" else s.mean())

    #: 覆盖缺口表里两行汇总的真实标签。论文表 15 只写 covered/missing，
    #: 产物里带 `ALL_gating_` 前缀，这里做一次映射，别让 claim 去记产物的内部命名。
    _GAP_ALIAS = {"covered": "ALL_gating_covered", "missing": "ALL_gating_missing"}

    def gap_row(self, backbone: str, col: str) -> float:
        key = self._GAP_ALIAS.get(backbone, backbone)
        row = self.coverage_gap[self.coverage_gap.backbone == key]
        if row.empty:
            raise KeyError(f"coverage_gap 里没有 backbone={key!r}；"
                           f"可选值 {self.coverage_gap.backbone.tolist()}")
        return float(row.iloc[0][col])


# --------------------------------------------------------------------------- #
# claim 定义
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Claim:
    """论文里印出来的一个数，加上它的复算方式。

    `printed` 是归一化后的字面串（见 `normalise_tex`）。留空表示不做字面检查——
    只对裸整数（`128`、`44`）这种到处都会偶然出现的值这么做，否则字面检查会退化成
    永远通过。
    """

    cid: str
    section: str
    expected: float
    tol: float
    source: str
    fn: Callable[[Artefacts], float]
    printed: tuple[str, ...] = ()
    note: str = ""


def _claims() -> list[Claim]:
    """全部 claim。分组顺序与论文章节一致，便于对着 PDF 逐条核。"""
    C = Claim
    out: list[Claim] = []

    # ---- §3 协议与覆盖 ---------------------------------------------------- #
    out += [
        C("protocol.p1_done", "§3", 844, 0, "weekend_report.json",
          lambda a: a.j("coverage.phase1.n_done"), ("844/844",)),
        C("protocol.p1_total", "§3", 844, 0, "weekend_report.json",
          lambda a: a.j("coverage.phase1.n_total")),
        C("protocol.p2_done", "§3", 423, 0, "weekend_report.json",
          lambda a: a.j("coverage.phase2.n_done"), ("423/428",)),
        C("protocol.p2_total", "§3", 428, 0, "weekend_report.json",
          lambda a: a.j("coverage.phase2.n_total")),
        C("protocol.p3_done", "§3", 768, 0, "weekend_report.json",
          lambda a: a.j("coverage.fredf.n_done"), ("768/768",)),
        C("protocol.total_runs", "§3", 2035, 0, "weekend_report.json（三期求和）",
          lambda a: (a.j("coverage.phase1.n_done") + a.j("coverage.phase2.n_done")
                     + a.j("coverage.fredf.n_done")),
          ("2,035",),
          note="2035 = 844 + 423 + 768，是运行数而非不同配置数"),
        C("protocol.p1_rows", "§3", 844, 0, "p1_results.csv 行数",
          lambda a: float(len(a.p1))),
        C("protocol.n_groups_test", "§3", 127, 0, "oracle_audit.csv 行数",
          lambda a: float(len(a.oracle_audit)), ("127",)),
        C("protocol.n_groups_gate", "§3", 96, 0, "weekend_report.json",
          lambda a: a.j("gating.coverage.n_groups_gating")),
        C("protocol.n_blocks", "§3", 128, 0, "p1_results.csv 块数",
          lambda a: float(len(a.block_matrix))),
        C("protocol.arms4", "§3", 43, 0, "oracle_audit.csv n_arms==4",
          lambda a: float((a.oracle_audit.n_arms == 4).sum()),
          note="43 组四臂、83 组三臂、1 组两臂"),
        C("protocol.arms3", "§3", 83, 0, "oracle_audit.csv n_arms==3",
          lambda a: float((a.oracle_audit.n_arms == 3).sum())),
        C("protocol.all_runs_ok", "§3", 844, 0, "p1_results.csv status=='ok'",
          lambda a: a.p1_all_ok(),
          note="没有一个格子是跑挂的；revin 的缺格全部来自 matrix.yaml 的剪枝规则"),
    ]

    # ---- §3 revin 剪枝的验证子集 ----------------------------------------- #
    # 这一组是审稿人最容易咬的地方：如果 revin 的缺格被写成「没跑完」，
    # 表 5 的 44 个完整块就变成了「预算决定了统计口径」，而事实是
    # 「一条先验剪枝规则决定了统计口径」。两者的可信度差别很大，必须钉住。
    out += [
        C("revin.verify_n", "§3", 12, 0, "p1_results.csv 配对（非 DLinear）",
          lambda a: a.revin_subset("n"),
          note="{ETTh1, Weather} x {96, 720} x 3 个内置归一化骨干"),
        C("revin.verify_median", "§3", 0.0041, 5e-3, "p1_results.csv 配对",
          lambda a: a.revin_subset("median"),
          note="中位数 +0.00%：外挂 RevIN 在内置归一化骨干上确实是 no-op"),
        C("revin.verify_wilcoxon_p", "§3", 0.2334, 5e-3, "p1_results.csv 配对",
          lambda a: a.revin_subset("wilcoxon_p"), ("p=0.23",)),
        C("revin.timesnet_etth1_720", "§3", 0.6186, 5e-3, "p1_results.csv 配对",
          lambda a: a.revin_cell("TimesNet", "ETTh1", 720), ("+0.62%",)),
        C("revin.timesnet_weather_720", "§3", 5.5647, 5e-3, "p1_results.csv 配对",
          lambda a: a.revin_cell("TimesNet", "Weather", 720), ("+5.56%",)),
        C("revin.dlinear_n", "§3", 64, 0, "p1_results.csv 配对（DLinear）",
          lambda a: a.revin_subset("n", internal_norm=False)),
        C("revin.dlinear_median", "§3", 0.9833, 5e-3, "p1_results.csv 配对",
          lambda a: a.revin_subset("median", internal_norm=False), ("+0.98%",),
          note="同一个臂在没有内置归一化的骨干上是全文最大的单臂效应"),
    ]

    # ---- §3 / §4.1 Friedman 与 Nemenyi ---------------------------------- #
    out += [
        C("nemenyi.cd_k4_n8", "§3", 1.6583, 5e-3, "Demsar (2006) 式 (7)",
          lambda a: a.nemenyi_cd(4, 8), ("1.66",),
          note="8 个数据集当块时 CD 比整个秩幅还宽"),
        C("nemenyi.cd_k4_n128", "§3", 0.4146, 5e-3, "Demsar (2006) 式 (7)",
          lambda a: a.nemenyi_cd(4, 128), ("0.41",)),
        C("nemenyi.cd_k4_n44", "§4.1", 0.7071, 5e-3, "Demsar (2006) 式 (7)",
          lambda a: a.nemenyi_cd(4, 44), ("0.71",)),
        C("friedman.n", "§4.1", 44, 0, "p1_results.csv 完整块",
          lambda a: a.friedman("n")),
        C("friedman.chi2", "§4.1", 30.90, 0.02, "p1_results.csv 完整块复算",
          lambda a: a.friedman("chi2"), ("30.9",)),
        C("friedman.p", "§4.1", 8.923e-7, 5e-9, "p1_results.csv 完整块复算",
          lambda a: a.friedman("p"), ("8.9x10^-7",)),
        C("friedman.rank_fredf", "§4.1", 1.9318, 5e-3, "p1_results.csv 完整块复算",
          lambda a: a.friedman("rank_fredf"), ("1.93",)),
        C("friedman.rank_revin", "§4.1", 2.1364, 5e-3, "p1_results.csv 完整块复算",
          lambda a: a.friedman("rank_revin"), ("2.14",)),
        C("friedman.rank_none", "§4.1", 2.5909, 5e-3, "p1_results.csv 完整块复算",
          lambda a: a.friedman("rank_none"), ("2.59",)),
        C("friedman.rank_sanlite", "§4.1", 3.3409, 5e-3, "p1_results.csv 完整块复算",
          lambda a: a.friedman("rank_san_lite"), ("3.34",)),
    ]

    # ---- §4.1 完整块 vs 逐对 --------------------------------------------- #
    out += [
        C("lever1.complete_blocks", "§4.1", 44, 0, "p1_results.csv 四臂完整块",
          lambda a: float(len(a.block_matrix[[CONTROL, *ARMS]].dropna(how="any")))),
        C("lever1.complete_dlinear_frac", "§4.1", 72.7, 0.6, "p1_results.csv",
          lambda a: float(
              a.block_matrix[[CONTROL, *ARMS]].dropna(how="any")
              .reset_index().backbone.eq("DLinear").mean() * 100.0),
          ("73%",), note="44 个完整块里 32 个来自 DLinear"),
        C("lever1.fredf_complete_mean", "§4.1", 3.604, 0.01, "p1_results.csv 复算",
          lambda a: a.main_table("fredf", True)["mean"], ("+3.60",)),
        C("lever1.fredf_complete_p", "§4.1", 0.005824, 5e-5, "p1_results.csv 复算",
          lambda a: a.main_table("fredf", True)["p"], ("0.0058",)),
        C("lever1.fredf_pairwise_mean", "§4.1", -2.373, 0.01, "p1_results.csv 复算",
          lambda a: a.main_table("fredf", False)["mean"], ("-2.37",)),
        C("lever1.fredf_pairwise_p", "§4.1", 0.4382, 0.002, "p1_results.csv 复算",
          lambda a: a.main_table("fredf", False)["p"], ("0.44",)),
        C("lever1.sanlite_complete_mean", "§4.1", -1.288, 0.01, "p1_results.csv 复算",
          lambda a: a.main_table("san_lite", True)["mean"], ("-1.29",)),
        C("lever1.sanlite_complete_p", "§4.1", 0.05583, 5e-4, "p1_results.csv 复算",
          lambda a: a.main_table("san_lite", True)["p"], ("0.056",)),
        C("lever1.sanlite_pairwise_mean", "§4.1", -3.939, 0.01, "p1_results.csv 复算",
          lambda a: a.main_table("san_lite", False)["mean"], ("-3.94",)),
        C("lever1.revin_mean", "§4.1", 3.548, 0.01, "p1_results.csv 复算",
          lambda a: a.main_table("revin", False)["mean"], ("+3.55",)),
        C("lever1.fredf_dlinear", "§4.2", 4.56, 0.01, "p1_results.csv 复算",
          lambda a: a.per_backbone("fredf", "DLinear")["mean"], ("+4.56",)),
        C("lever1.fredf_patchtst", "§4.2", -16.63, 0.01, "p1_results.csv 复算",
          lambda a: a.per_backbone("fredf", "PatchTST")["mean"], ("-16.63",)),
        C("lever1.fredf_timesnet", "§4.2", 1.41, 0.01, "p1_results.csv 复算",
          lambda a: a.per_backbone("fredf", "TimesNet")["mean"], ("+1.41",)),
        C("lever1.fredf_itrans", "§4.2", 1.17, 0.01, "p1_results.csv 复算",
          lambda a: a.per_backbone("fredf", "iTransformer")["mean"], ("+1.17",)),
    ]

    # ---- §4.2 指标杠杆 ---------------------------------------------------- #
    out += [
        C("lever2.revin_mae_median", "§4.2", 1.377, 0.005, "metric_robustness.csv",
          lambda a: a.rb("mae", "revin", "median_rel_gain_pct"), ("+1.38",)),
        C("lever2.revin_mae_wins", "§4.2", 37, 0, "metric_robustness.csv",
          lambda a: a.rb("mae", "revin", "win"), ("37 of 44",)),
        C("lever2.revin_mse_median", "§4.2", 0.0277, 5e-4, "metric_robustness.csv",
          lambda a: a.rb("mse", "revin", "median_rel_gain_pct"), ("+0.03",)),
        C("lever2.revin_spearman", "§4.2", 0.6357, 5e-3, "metric_agreement.csv",
          lambda a: float(a.rb_agree.set_index("method").loc["revin", "spearman_rho"]),
          ("0.64",)),
        C("lever2.revin_sign_agree", "§4.2", 63.64, 0.5, "metric_agreement.csv",
          lambda a: float(a.rb_agree.set_index("method").loc["revin", "sign_agreement"]) * 100,
          ("64%",)),
        C("lever2.fredf_spearman", "§4.2", 0.9664, 5e-3, "metric_agreement.csv",
          lambda a: float(a.rb_agree.set_index("method").loc["fredf", "spearman_rho"]),
          ("0.97",)),
        C("lever2.sanlite_spearman", "§4.2", 0.837, 5e-3, "metric_agreement.csv",
          lambda a: float(a.rb_agree.set_index("method").loc["san_lite", "spearman_rho"]),
          ("0.84",)),
        C("lever2.sanlite_mae_median", "§4.2", -1.012, 0.005, "metric_robustness.csv",
          lambda a: a.rb("mae", "san_lite", "median_rel_gain_pct"), ("-1.01",)),
    ]

    # ---- §4.3 均值/中位数与尾部 ------------------------------------------ #
    out += [
        C("lever3.fredf_mse_median", "§4.3", 0.448, 0.005, "metric_robustness.csv",
          lambda a: a.rb("mse", "fredf", "median_rel_gain_pct"), ("+0.45",)),
        C("lever3.fredf_mae_median", "§4.3", 0.896, 0.005, "metric_robustness.csv",
          lambda a: a.rb("mae", "fredf", "median_rel_gain_pct"), ("+0.90",)),
        C("lever3.fredf_mae_mean", "§4.3", -1.179, 0.005, "metric_robustness.csv",
          lambda a: a.rb("mae", "fredf", "mean_rel_gain_pct"), ("-1.18",)),
        C("lever3.fredf_skew", "§4.3", -4.471, 0.01, "metric_robustness.csv",
          lambda a: a.rb("mse", "fredf", "skew"), ("-4.5",)),
        C("lever3.fredf_worst", "§4.3", -123.36, 0.05, "metric_robustness.csv",
          lambda a: a.rb("mse", "fredf", "worst_rel_gain_pct"), ("-123",)),
        C("lever3.fredf_tail10", "§4.3", 10.94, 0.05, "metric_robustness.csv",
          lambda a: a.rb("mse", "fredf", "frac_worse_than_10pct") * 100, ("10.9%",)),
        C("lever3.fredf_tail25", "§4.3", 3.91, 0.05, "metric_robustness.csv",
          lambda a: a.rb("mse", "fredf", "frac_worse_than_25pct") * 100, ("3.9%",)),
        C("lever3.sanlite_tail10", "§4.3", 21.88, 0.05, "metric_robustness.csv",
          lambda a: a.rb("mse", "san_lite", "frac_worse_than_10pct") * 100, ("21.9%",)),
        C("lever3.sanlite_worst", "§4.3", -29.97, 0.05, "metric_robustness.csv",
          lambda a: a.rb("mse", "san_lite", "worst_rel_gain_pct"), ("-30%",)),
    ]

    # ---- §5 α 调参 -------------------------------------------------------- #
    out += [
        C("tuning.fixed_alpha05", "§5", -3.698, 0.005, "weekend_report.json",
          lambda a: a.j("fredf.summary.mean_gain_fixed_alpha05"), ("-3.70",)),
        C("tuning.val_tuned", "§5", 1.785, 0.005, "weekend_report.json",
          lambda a: a.j("fredf.summary.mean_gain_tuned_alpha"), ("+1.79",)),
        C("tuning.test_tuned", "§5", 2.294, 0.005, "weekend_report.json",
          lambda a: a.j("fredf.summary.mean_gain_oracle_alpha"), ("+2.29",)),
        C("tuning.gap", "§5", 5.484, 0.005, "weekend_report.json",
          lambda a: a.j("fredf.summary.tuning_gap"), ("+5.48",)),
        C("tuning.positive_after_tuning", "§5", 59.375, 0.05, "weekend_report.json",
          lambda a: a.j("fredf.summary.blocks_tuned_alpha_positive_pct"), ("59.4%",)),
        C("tuning.best_single_alpha", "§5", 0.1193, 5e-4, "weekend_report.json",
          lambda a: a.j("fredf.summary.plain_best_single_alpha_mean_gain"), ("+0.12",)),
        C("tuning.sqrth_single_alpha", "§5", 0.1768, 5e-4, "weekend_report.json",
          lambda a: a.j("fredf.summary.sqrth_best_single_alpha_mean_gain"), ("+0.18",)),
        C("tuning.distinct_alpha_plain", "§5", 1.7917, 5e-3, "weekend_report.json",
          lambda a: a.j("fredf.summary.plain_mean_distinct_best_alpha_across_horizons"),
          ("1.79",)),
        C("tuning.distinct_alpha_sqrth", "§5", 1.4167, 5e-3, "weekend_report.json",
          lambda a: a.j("fredf.summary.sqrth_mean_distinct_best_alpha_across_horizons"),
          ("1.42",)),
        C("tuning.alpha_blocks", "§5", 96, 0, "fredf_alpha.csv 行数",
          lambda a: float(len(a.p3))),
        C("tuning.alpha_cells", "§5", 768, 0, "fredf_alpha.csv 的 α 设置数求和",
          lambda a: float((a.p3.plain_n_alphas + a.p3.sqrth_n_alphas).sum()),
          note="96 块 ×（5 plain + 3 √H）= 768 个 α-cell，与 phase-3 的 768 次训练是两件事"),
    ]

    # ---- §6 效应量：seed 噪声与算力 --------------------------------------- #
    out += [
        C("effect.blocks_with_noise", "§6", 144, 0, "robustness_summary.json",
          lambda a: a.rbs("seed_stability.n_blocks_with_noise")),
        C("effect.median_seed_sd", "§6", 0.004947, 5e-6, "robustness_summary.json",
          lambda a: a.rbs("seed_stability.median_seed_sd"), ("0.0049",)),
        C("effect.frac_within_noise", "§6", 27.78, 0.05, "robustness_summary.json",
          lambda a: a.rbs("seed_stability.frac_within_noise") * 100, ("28%",)),
        C("effect.wins_within_noise", "§6", 27.42, 0.05, "robustness_summary.json",
          lambda a: a.rbs("seed_stability.frac_wins_within_noise") * 100, ("27%",)),
        C("effect.median_gain_over_noise", "§6", 2.1329, 5e-3, "robustness_summary.json",
          lambda a: a.rbs("seed_stability.median_gain_over_noise"), ("2.13",)),
        C("effect.n_seed_configs", "§6", 208, 0, "p1_results.csv 三 seed 配置数",
          lambda a: float((a.p1.groupby(["backbone", "dataset", "pred_len", "plugin"])
                           .seed.nunique() >= 3).sum()), ("208",)),
        C("effect.cost_fredf", "§6", 1.0335, 5e-3, "cost_benefit_by_method.csv",
          lambda a: a.cost("fredf", "median_train_time_ratio"), ("1.03",)),
        C("effect.cost_revin", "§6", 1.0490, 5e-3, "cost_benefit_by_method.csv",
          lambda a: a.cost("revin", "median_train_time_ratio"), ("1.05",)),
        C("effect.cost_sanlite", "§6", 1.7287, 5e-3, "cost_benefit_by_method.csv",
          lambda a: a.cost("san_lite", "median_train_time_ratio"), ("1.73",)),
        C("effect.gain_per_pct_time", "§6", 0.1335, 5e-4, "cost_benefit_by_method.csv",
          lambda a: a.cost("fredf", "gain_per_pct_extra_time"), ("0.13",)),
    ]

    # ---- §7 lead-time 合法轴 --------------------------------------------- #
    out += [
        C("leadtime.n_blocks", "§7", 128, 0, "robustness_summary.json",
          lambda a: a.rbs("leadtime.n_blocks")),
        C("leadtime.n_3arm", "§7", 84, 0, "leadtime_argmin_stability.csv",
          lambda a: float((a.rb_leadtime.n_arms_available == 3).sum()), ("84",),
          note="84/44 是块口径，和 §3 的 83/43/1 决策组口径不是一回事"),
        C("leadtime.n_4arm", "§7", 44, 0, "leadtime_argmin_stability.csv",
          lambda a: float((a.rb_leadtime.n_arms_available == 4).sum())),
        C("leadtime.frac_constant", "§7", 56.25, 0.05, "robustness_summary.json",
          lambda a: a.rbs("leadtime.frac_constant_argmin") * 100, ("56%",)),
        C("leadtime.frac_suboptimal", "§7", 43.75, 0.05, "robustness_summary.json",
          lambda a: a.rbs("leadtime.frac_blocks_whole_best_suboptimal_somewhere") * 100,
          ("44%",)),
        C("leadtime.oracle_vs_none", "§7", 3.8218, 5e-3, "robustness_summary.json",
          lambda a: a.rbs("leadtime.leadtime_oracle_vs_control_mean_pct"), ("+3.82",)),
        C("leadtime.oracle_vs_bestfixed", "§7", 0.2402, 5e-3, "robustness_summary.json",
          lambda a: a.rbs("leadtime.leadtime_oracle_vs_best_fixed_mean_pct"), ("+0.24",)),
        C("leadtime.oracle_vs_bestfixed_median", "§7", 0.0, 1e-9,
          "robustness_summary.json",
          lambda a: a.rbs("leadtime.leadtime_oracle_vs_best_fixed_median_pct"),
          ("0.00%",)),
        C("leadtime.oracle_vs_bestfixed_max", "§7", 3.2894, 5e-3,
          "robustness_summary.json",
          lambda a: a.rbs("leadtime.leadtime_oracle_vs_best_fixed_max_pct"), ("+3.29",)),
        C("leadtime.n_beats_bestfixed", "§7", 56, 0, "robustness_summary.json",
          lambda a: a.rbs("leadtime.n_blocks_leadtime_oracle_beats_best_fixed"),
          ("56/128",)),
        C("leadtime.frac_over_half_point", "§7", 12.5, 0.05, "robustness_summary.json",
          lambda a: a.rbs("leadtime.frac_blocks_leadtime_gain_over_0p5pp") * 100,
          ("12%",)),
    ]

    # ---- §8 oracle 与半衰期 ---------------------------------------------- #
    def _q(a: Artefacts, arm: str, quarter: int) -> float:
        rows = a.rb_summary["leadtime"]["gain_by_quarter"]
        for r in rows:
            if r["method"] == arm and int(r["quarter"]) == quarter:
                return float(r["median"])
        raise KeyError((arm, quarter))

    out += [
        C("leadtime.fredf_q1", "§7", -0.3609, 5e-3, "robustness_summary.json",
          lambda a: _q(a, "fredf", 1), ("-0.36",)),
        C("leadtime.fredf_q3", "§7", 0.9263, 5e-3, "robustness_summary.json",
          lambda a: _q(a, "fredf", 3), ("+0.93",)),
        C("leadtime.sanlite_q1", "§7", -5.0296, 5e-3, "robustness_summary.json",
          lambda a: _q(a, "san_lite", 1), ("-5.03",)),
        C("oracle.headroom_127", "§8", 10.1455, 5e-3, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.oracle_gain_real_pct.mean()), ("+10.15",)),
        C("oracle.headroom_96", "§8", 9.7406, 5e-3, "weekend_report.json",
          lambda a: a.j("gating.inpool.mean_gain_oracle_vs_none"), ("+9.74",)),
        C("oracle.bestfixed_val_96", "§8", 2.1213, 5e-3, "weekend_report.json",
          lambda a: a.j("gating.inpool.mean_gain_bestfixed_vs_none"), ("+2.12",)),
        C("oracle.relabel_invariant_groups", "§8", 127, 0, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.oracle_invariant_under_relabel.sum()),
          note="127/127 组在逐窗口重贴臂标签下 oracle 完全不变"),
        C("persist.chance", "§8", 0.3951, 5e-4, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.chance_persist.mean()), ("0.395",)),
        C("persist.lag1", "§8", 0.8639, 5e-4, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.persist_lag1.mean()), ("0.864",)),
        C("persist.excess_lag1", "§8", 0.4688, 5e-4, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.excess_persist_lag1.mean()), ("+0.469",)),
        C("persist.lagH_4", "§8", 0.4673, 5e-4, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.persist_lagH_4.mean()), ("0.467",)),
        C("persist.excess_lagH_4", "§8", 0.0723, 5e-4, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.excess_persist_lagH_4.mean()), ("+0.072",)),
        C("persist.lagH_2", "§8", 0.4149, 5e-4, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.persist_lagH_2.mean()), ("0.415",)),
        C("persist.lagH", "§8", 0.3886, 5e-4, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.persist_lagH.mean()), ("0.389",)),
        C("persist.excess_lagH", "§8", -0.0065, 5e-4, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.excess_persist_lagH.mean()), ("-0.007",)),
        C("persist.lag2H", "§8", 0.3965, 5e-4, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.persist_lag2H.mean()), ("0.397",)),
        C("persist.z_gt2_lag1", "§8", 127, 0, "oracle_audit.csv",
          lambda a: float((a.oracle_audit.z_lag1 > 2).sum()), ("127/127",)),
        C("persist.z_gt2_lagH_4", "§8", 93, 0, "oracle_audit.csv",
          lambda a: float((a.oracle_audit.z_lagH_4 > 2).sum()), ("93/127",)),
        C("persist.z_gt2_lagH_2", "§8", 72, 0, "oracle_audit.csv",
          lambda a: float((a.oracle_audit.z_lagH_2 > 2).sum()), ("72/127",)),
        C("persist.z_gt2_lagH", "§8", 37, 0, "oracle_audit.csv",
          lambda a: float((a.oracle_audit.z_lagH > 2).sum()), ("37/127",)),
        C("persist.z_gt2_lag2H", "§8", 41, 0, "oracle_audit.csv",
          lambda a: float((a.oracle_audit.z_lag2H > 2).sum()), ("41/127",)),
        C("persist.z_lt_m2_lagH", "§8", 37, 0, "oracle_audit.csv z_lagH<-2",
          lambda a: float((a.oracle_audit.z_lagH < -2).sum()), ("37/127",)),
        C("persist.pos_excess_lagH", "§8", 57, 0, "oracle_audit.csv excess>0",
          lambda a: float((a.oracle_audit.excess_persist_lagH > 0).sum()),
          ("57/127",)),
        C("persist.z_blockdeflated_lagH", "§8", 0, 0,
          "oracle_audit.csv |z_lagH|/sqrt(H)>2",
          lambda a: float(((a.oracle_audit.z_lagH.abs()
                            / np.sqrt(a.oracle_audit.pred_len)) > 2).sum()),
          ("0/127",)),
        C("halflife.mean_over_H", "§8", 0.06052, 5e-5, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.half_life_over_H.mean()), ("0.061", "6.1%")),
        C("halflife.median_over_H", "§8", 0.05101, 5e-5, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.half_life_over_H.median()),
          ("0.051", "5.1%")),
        C("halflife.median_windows", "§8", 9.043, 5e-3, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.persist_half_life_windows.median()), ("9.0",)),
        C("halflife.mean_windows", "§8", 11.319, 5e-3, "oracle_audit.csv",
          lambda a: float(a.oracle_audit.persist_half_life_windows.mean()), ("11.3",)),
        C("halflife.frac_below_H_over_10", "§8", 85.83, 0.05, "oracle_audit.csv",
          lambda a: float((a.oracle_audit.half_life_over_H < 0.1).mean() * 100), ("86%",)),
        C("strata.n_used", "§8", 20, 0, "audit_strata.csv",
          lambda a: float(len(a.strata))),
        C("strata.all_below_25pct", "§8", 20, 0, "audit_strata.csv",
          lambda a: float((a.strata.median_half_life_over_H < 0.25).sum()), ("20/20",)),
        C("strata.max_hl_over_H", "§8", 19.21, 0.05, "audit_strata.csv",
          lambda a: float(a.strata.median_half_life_over_H.max() * 100), ("19.2%",)),
        C("strata.max_excess_lagH", "§8", 0.0316, 5e-4, "audit_strata.csv",
          lambda a: float(a.strata.mean_excess_persist_lagH.max()), ("+0.032",)),
    ]

    # ---- §8.2 horizon 标度 ----------------------------------------------- #
    # 两个 Spearman 必须写清是「8 个 horizon 级中位数」还是「127 个组」：
    # 前者 +0.74 / -0.88，后者 +0.49 / -0.73。论文两个口径都印，这里两个都钉。
    out += [
        C("hscale.spearman_hl_medians", "§8.2", 0.7381, 5e-4,
          "oracle_audit.csv 按 pred_len 取中位数后 Spearman",
          lambda a: a.horizon_scaling("spearman_hl_medians"), ("+0.74",)),
        C("hscale.spearman_ratio_medians", "§8.2", -0.8810, 5e-4,
          "oracle_audit.csv 按 pred_len 取中位数后 Spearman",
          lambda a: a.horizon_scaling("spearman_ratio_medians"), ("-0.88",)),
        C("hscale.spearman_hl_groups", "§8.2", 0.4867, 5e-4,
          "oracle_audit.csv 逐组 Spearman",
          lambda a: a.horizon_scaling("spearman_hl_groups"), ("+0.49",)),
        C("hscale.spearman_ratio_groups", "§8.2", -0.7256, 5e-4,
          "oracle_audit.csv 逐组 Spearman",
          lambda a: a.horizon_scaling("spearman_ratio_groups"), ("-0.73",)),
        C("hscale.hl_at_24", "§8.2", 4.6101, 5e-3, "oracle_audit.csv",
          lambda a: a.horizon_scaling("hl_at_24"), ("4.6",)),
        C("hscale.hl_at_720", "§8.2", 15.3036, 5e-3, "oracle_audit.csv",
          lambda a: a.horizon_scaling("hl_at_720"), ("15.3",)),
        C("hscale.hl_growth", "§8.2", 3.3196, 5e-3, "oracle_audit.csv",
          lambda a: a.horizon_scaling("hl_growth"), ("3.3x",),
          note="H 涨 30 倍，半衰期只涨 3.3 倍，所以比值反而变小"),
        C("hscale.ratio_at_24", "§8.2", 0.1921, 5e-4, "oracle_audit.csv",
          lambda a: a.horizon_scaling("ratio_at_24"), ("0.192",)),
        C("hscale.ratio_at_720", "§8.2", 0.0213, 5e-4, "oracle_audit.csv",
          lambda a: a.horizon_scaling("ratio_at_720"), ("0.021",)),
    ]

    # ---- §8.3 成对 seed 复制 --------------------------------------------- #
    out += [
        C("seed.n_pairs", "§8.3", 48, 0, "weekend_report.json",
          lambda a: a.j("seed_audit.n_pairs"), ("48",)),
        C("seed.self_oracle", "§8.3", 7.260, 5e-3, "weekend_report.json",
          lambda a: a.j("seed_audit.mean_self_oracle_gain_pct"), ("+7.26",)),
        C("seed.xseed_oracle", "§8.3", 5.739, 5e-3, "weekend_report.json",
          lambda a: a.j("seed_audit.mean_xseed_oracle_gain_pct"), ("+5.74",)),
        C("seed.reproducible_frac", "§8.3", 79.05, 0.05, "weekend_report.json",
          lambda a: a.j("seed_audit.reproducible_frac_of_headroom") * 100, ("79%",)),
        C("seed.argmin_agree", "§8.3", 0.7878, 5e-4, "weekend_report.json",
          lambda a: a.j("seed_audit.mean_argmin_agree"), ("0.788",)),
        C("seed.chance_agree", "§8.3", 0.3743, 5e-4, "weekend_report.json",
          lambda a: a.j("seed_audit.mean_chance_agree"), ("0.374",)),
        C("seed.same_arm_corr", "§8.3", 0.9888, 5e-4, "weekend_report.json",
          lambda a: a.j("seed_audit.mean_same_arm_err_corr"), ("0.989",)),
        C("seed.mse_gap_pct", "§8.3", 1.326, 5e-3, "weekend_report.json",
          lambda a: a.j("seed_audit.mean_seed_mse_gap_pct"), ("1.33%",)),
        C("seed.best_fixed_stable", "§8.3", 87.5, 0.05, "weekend_report.json",
          lambda a: a.j("seed_audit.best_fixed_arm_stable_pct"), ("88%",)),
        C("seed.pairs_beating_bf", "§8.3", 46, 0, "weekend_report.json",
          lambda a: a.j("seed_audit.pairs_xseed_beats_best_fixed"), ("46/48",)),
    ]

    # ---- §9 三类选择器 ---------------------------------------------------- #
    out += [
        C("gate.inpool_vs_bf", "§9.1", -0.6679, 5e-4, "weekend_report.json",
          lambda a: a.j("gating.inpool.mean_gain_gate_vs_best_fixed"), ("-0.67",)),
        C("gate.inpool_beats_bf_pct", "§9.1", 21.875, 0.05, "weekend_report.json",
          lambda a: a.j("gating.inpool.groups_gate_beats_best_fixed_pct"), ("22%",)),
        C("gate.inpool_acc", "§9.1", 0.4161, 5e-4, "weekend_report.json",
          lambda a: a.j("gating.inpool.mean_gate_acc"), ("0.416",)),
        C("gate.inpool_majority_acc", "§9.1", 0.4862, 5e-4, "weekend_report.json",
          lambda a: a.j("gating.inpool.mean_majority_acc"), ("0.486",)),
        C("gate.lodo_n_groups", "§9.1", 76, 0, "weekend_report.json",
          lambda a: a.j("gating.lodo.n_groups"), ("76",),
          note="留一数据集需要同形状下至少 3 个数据集，ILI 整体不合格"),
        C("gate.lodo_rows", "§9.1", 76, 0, "selector_window_lodo.csv 行数",
          lambda a: float(len(a.lodo))),
        C("gate.lodo_no_ili", "§9.1", 0, 0, "selector_window_lodo.csv",
          lambda a: float((a.lodo.dataset == "ILI").sum()),
          note="ILI 每个 horizon 只存在于一个数据集，留一后训练集为空"),
        C("gate.lodo_vs_bf", "§9.1", -1.559, 5e-3, "weekend_report.json",
          lambda a: a.j("gating.lodo.mean_gain_gate_vs_best_fixed"), ("-1.56",)),
        C("gate.random_vs_bf", "§9.1", -1.0355, 5e-4, "weekend_report.json",
          lambda a: a.j("gating.inpool_controls.mean_gain_random_vs_best_fixed"),
          ("-1.04",)),
        C("gate.shuffled_vs_bf", "§9.1", -0.5360, 5e-4, "weekend_report.json",
          lambda a: a.j("gating.inpool_controls.mean_gain_shuffled_vs_best_fixed"),
          ("-0.54",)),
        C("gate.minus_shuffled", "§9.1", -0.2191, 5e-4, "weekend_report.json",
          lambda a: a.j("gating.inpool_controls.mean_gate_minus_shuffled"), ("-0.22",)),
        C("gate.minus_random", "§9.1", 0.3599, 5e-4, "weekend_report.json",
          lambda a: a.j("gating.inpool_controls.mean_gate_minus_random"), ("+0.36",)),
        # ---- 同 split 切分诊断 ---------------------------------------- #
        C("splitgate.val_n", "§9.2", 81, 0, "split_val_splitfix.csv",
          lambda a: float(len(a.splitfix("val")))),
        C("splitgate.test_n", "§9.2", 111, 0, "split_test_splitfix.csv",
          lambda a: float(len(a.splitfix("test")))),
        C("splitgate.val_vs_bf", "§9.2", -1.1833, 5e-3, "split_val_splitfix.csv",
          lambda a: a.split_gate("val", "gain_gate_vs_best_fixed"), ("-1.18",)),
        C("splitgate.test_vs_bf", "§9.2", -0.5276, 5e-3, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "gain_gate_vs_best_fixed"), ("-0.53",),
          note="曾经在 §12 被写成 -0.55；这条 claim 就是为了钉住它"),
        C("splitgate.val_vs_none", "§9.2", 2.0056, 5e-3, "split_val_splitfix.csv",
          lambda a: a.split_gate("val", "gain_gate_vs_none"), ("+2.01",)),
        C("splitgate.test_vs_none", "§9.2", 2.3254, 5e-3, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "gain_gate_vs_none"), ("+2.33",)),
        C("splitgate.val_oracle", "§9.2", 11.349, 5e-3, "split_val_splitfix.csv",
          lambda a: a.split_gate("val", "gain_oracle_vs_none"), ("+11.35",)),
        C("splitgate.test_oracle", "§9.2", 10.371, 5e-3, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "gain_oracle_vs_none"), ("+10.37",)),
        C("splitgate.test_win_pct", "§9.2", 37.84, 0.05, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "win_pct"), ("38%",)),
        C("splitgate.val_win_pct", "§9.2", 34.57, 0.05, "split_val_splitfix.csv",
          lambda a: a.split_gate("val", "win_pct"), ("35%",)),
        C("splitgate.test_p", "§9.2", 0.195, 5e-3, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "wilcoxon_p"), ("0.19",)),
        C("splitgate.test_acc", "§9.2", 0.4112, 5e-4, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "gate_acc"), ("0.411",)),
        C("splitgate.test_majority", "§9.2", 0.4859, 5e-4, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "majority_acc"), ("0.486",)),
        C("splitgate.test_beats_majority", "§9.2", 12.61, 0.05, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "gate_beats_majority_pct"), ("13%",)),
        # oracle 实现比例：重尾，三个口径差一个量级，论文三个都印，这里三个都钉。
        C("splitgate.oracle_realized_mean", "§9.2", 3.303, 5e-3,
          "split_test_splitfix.csv",
          lambda a: a.split_gate_realized("test", "mean"), ("3.3%",)),
        C("splitgate.oracle_realized_median", "§9.2", 4.5994, 5e-3,
          "split_test_splitfix.csv",
          lambda a: a.split_gate_realized("test", "median"), ("4.6%",)),
        C("splitgate.oracle_realized_ratio_of_means", "§9.2", 22.42, 0.02,
          "split_test_splitfix.csv",
          lambda a: a.split_gate_realized("test", "ratio_of_means"), ("22%",),
          note="+2.33 / +10.37；和逐组比值的均值 3.3% 不是同一个量"),
        C("splitgate.oracle_realized_q25", "§9.2", -9.4547, 0.02,
          "split_test_splitfix.csv",
          lambda a: a.split_gate_realized("test", "q25"), ("-9%",)),
        C("splitgate.oracle_realized_q75", "§9.2", 35.3835, 0.02,
          "split_test_splitfix.csv",
          lambda a: a.split_gate_realized("test", "q75"), ("+35%",)),
        C("splitgate.oracle_realized_min", "§9.2", -578.55, 0.05,
          "split_test_splitfix.csv",
          lambda a: a.split_gate_realized("test", "min"), ("-579%",),
          note="分母趋零的组把逐组比值的均值拖垮，这就是为什么不能只印均值"),
        C("splitgate.gate_vs_none", "§9.2", 2.3254, 5e-3, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "gain_gate_vs_none"), ("+2.33",)),
        C("splitgate.oracle_vs_none", "§9.2", 10.3706, 5e-3, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "gain_oracle_vs_none"), ("+10.37",)),
        C("splitgate.test_vs_valmean", "§9.2", 0.0819, 5e-3, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "gain_gate_vs_best_fixed_valmean"), ("+0.08",)),
        C("splitgate.test_bf_is_argmin", "§9.2", 70.27, 0.05, "split_test_splitfix.csv",
          lambda a: a.split_gate("test", "bf_split_is_eval_argmin") * 100, ("70%",)),
        C("splitgate.val_bf_is_argmin", "§9.2", 46.91, 0.05, "split_val_splitfix.csv",
          lambda a: a.split_gate("val", "bf_split_is_eval_argmin") * 100, ("47%",)),
        # ---- 延迟在线反馈 --------------------------------------------- #
        C("online.n_groups", "§9.3", 123, 0, "weekend_report.json",
          lambda a: a.j("gating.online.n_groups"), ("123",)),
        C("online.n_groups_all", "§9.3", 127, 0, "weekend_report.json",
          lambda a: a.j("gating.online.n_groups_all")),
        C("online.n_insufficient", "§9.3", 4, 0, "weekend_report.json",
          lambda a: a.j("gating.online.n_groups_insufficient"),
          note="被排除的 4 组全是 Exchange x H=720：合法延迟 720 > 评估窗口 399"),
        C("online.exchange720_eval_windows", "§9.3", 399, 0,
          "selector_window_online.csv（Exchange, H=720）",
          lambda a: float(a.online_raw[
              (a.online_raw.dataset == "Exchange")
              & (a.online_raw.pred_len == 720)].n_eval_windows.max()), ("399",),
          note="合法延迟 720 > 399 个评估窗口，这 4 组连一次合法决策都做不了"),
        C("online.insufficient_all_exchange720", "§9.3", 4, 0,
          "selector_window_online.csv",
          lambda a: float(((a.online_raw.insufficient)
                           & (a.online_raw.dataset == "Exchange")
                           & (a.online_raw.pred_len == 720)).sum()),
          note="确认 4 个不足组全部落在 Exchange x H=720，而不是散落各处"),
        C("online.vs_none", "§9.3", 1.1885, 5e-3, "weekend_report.json",
          lambda a: a.j("gating.online.mean_gain_online_vs_none"), ("+1.19",)),
        C("online.vs_bf", "§9.3", -0.8699, 5e-4, "weekend_report.json",
          lambda a: a.j("gating.online.mean_gain_online_vs_best_fixed"), ("-0.87",)),
        C("online.nodelay_vs_bf", "§9.3", 2.4713, 5e-3, "weekend_report.json",
          lambda a: a.j("gating.online.mean_gain_nodelay_vs_best_fixed"), ("+2.47",)),
        C("online.delay_cost_pp", "§9.3", 3.3412, 5e-3, "weekend_report.json",
          lambda a: a.j("gating.online.mean_delay_cost_pp"), ("3.34",)),
        C("online.beats_bf_pct", "§9.3", 20.33, 0.05, "weekend_report.json",
          lambda a: a.j("gating.online.groups_online_beats_best_fixed_pct"), ("20%",)),
        C("online.feedback_window", "§9.3", 200, 0, "weekend_report.json",
          lambda a: a.j("gating.online.feedback_window"), ("200",)),
        C("sweep.breakeven_ratio", "§9.3", 0.25, 1e-9, "weekend_report.json",
          lambda a: a.j("gating.online_sweep.breakeven_delay_ratio"), ("0.25",)),
        C("sweep.gain_at_delay_1H", "§9.3", -0.3837, 5e-4, "weekend_report.json",
          lambda a: a.j("gating.online_sweep.feasible_gain_vs_best_fixed"), ("-0.38",)),
        C("sweep.cheat_gain", "§9.3", 1.8913, 5e-3, "weekend_report.json",
          lambda a: a.j("gating.online_sweep.cheat_gain_vs_best_fixed"), ("+1.89",)),
    ]

    # ---- §10 可回收的部分 ------------------------------------------------ #
    out += [
        C("recover.bftest_minus_bfval", "§10", 1.3582, 5e-3, "weekend_report.json",
          lambda a: a.j("gating.online.mean_gain_bestfixed_test_vs_val"), ("+1.36",)),
        C("recover.arm_mismatch_pct", "§10", 34.15, 0.05, "weekend_report.json",
          lambda a: a.j("gating.online.groups_arm_id_mismatch_pct"), ("34%",)),
        C("recover.online_minus_bftest", "§10", -2.3204, 5e-3, "weekend_report.json",
          lambda a: a.j("gating.online.mean_gain_online_vs_best_fixed_test"), ("-2.32",)),
        C("recover.beats_bftest_pct", "§10", 4.878, 0.05, "weekend_report.json",
          lambda a: a.j("gating.online.groups_online_beats_best_fixed_test_pct"),
          ("5%",)),
    ]

    # ---- §12 覆盖缺口 ---------------------------------------------------- #
    out += [
        C("gap.covered_oracle", "§12", 9.74, 0.02, "coverage_gap_splitfix.csv",
          lambda a: a.gap_row("covered", "mean_oracle_gain_pct"), ("+9.7",)),
        C("gap.missing_oracle", "§12", 11.40, 0.02, "coverage_gap_splitfix.csv",
          lambda a: a.gap_row("missing", "mean_oracle_gain_pct"), ("+11.4",)),
        C("gap.covered_hl_over_H", "§12", 0.049, 5e-4, "coverage_gap_splitfix.csv",
          lambda a: a.gap_row("covered", "median_half_life_over_H"), ("0.049",)),
        C("gap.missing_hl_over_H", "§12", 0.053, 5e-4, "coverage_gap_splitfix.csv",
          lambda a: a.gap_row("missing", "median_half_life_over_H"), ("0.053",)),
        C("gap.covered_excess_lagH", "§12", -0.008, 5e-4, "coverage_gap_splitfix.csv",
          lambda a: a.gap_row("covered", "mean_excess_lagH"), ("-0.008",)),
        C("gap.missing_excess_lagH", "§12", -0.002, 5e-4, "coverage_gap_splitfix.csv",
          lambda a: a.gap_row("missing", "mean_excess_lagH"), ("-0.002",)),
        C("gap.n_missing", "§12", 31, 0, "coverage_gap_splitfix.csv",
          lambda a: a.gap_row("missing", "n_groups_test_side"), ("31",)),
    ]

    return out


CLAIMS: list[Claim] = _claims()


# --------------------------------------------------------------------------- #
# LaTeX 字面检查
# --------------------------------------------------------------------------- #
#: 归一化时要抹掉的 LaTeX 噪声。目的是让 `$2{,}035$` 与 `2,035`、
#: `$+10.15\%$` 与 `+10.15%` 能对上，同时不把数字本身改坏。
_TEX_STRIP = re.compile(r"\\[,;:!]|\\ |[{}$~]|\\mathbf|\\textbf|\\emph|\\arm|\\text")


def normalise_tex(text: str) -> str:
    """把 LaTeX 源码压成便于做字面数字搜索的纯文本。

    只做「删除」，不做「替换成别的数字」，所以不可能把一个错的数洗成对的。
    """
    text = _TEX_STRIP.sub("", text)
    text = text.replace("\\%", "%").replace("\\times", "x")
    text = re.sub(r"\s+", " ", text)
    return text


def load_tex(paper_dir: Path) -> str:
    """主文件 + 生成的表格，合成一份归一化文本。"""
    parts: list[str] = []
    main = paper_dir / "main.tex"
    if main.exists():
        parts.append(main.read_text())
    tables = paper_dir / "tables_robust"
    if tables.is_dir():
        parts.extend(p.read_text() for p in sorted(tables.glob("*.tex")))
    return normalise_tex("\n".join(parts))


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #
@dataclass
class Result:
    claim: Claim
    computed: float | None = None
    numeric_ok: bool = False
    literal_ok: bool = True
    missing_literals: tuple[str, ...] = ()
    error: str = ""

    @property
    def ok(self) -> bool:
        return self.numeric_ok and self.literal_ok and not self.error

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.claim.cid,
            "section": self.claim.section,
            "expected": self.claim.expected,
            "computed": self.computed,
            "tol": self.claim.tol,
            "source": self.claim.source,
            "numeric_ok": self.numeric_ok,
            "literal_ok": self.literal_ok,
            "missing_literals": list(self.missing_literals),
            "error": self.error,
            "ok": self.ok,
        }


def check(claims: Sequence[Claim], art: Artefacts, tex: str) -> list[Result]:
    """逐条校验。任何一条抛异常都记成失败而不是中断整轮。"""
    results: list[Result] = []
    for c in claims:
        r = Result(claim=c)
        try:
            r.computed = float(c.fn(art))
        except Exception as exc:                 # noqa: BLE001 - 报告而非中断
            r.error = f"{type(exc).__name__}: {exc}"
            results.append(r)
            continue
        if math.isnan(r.computed) or math.isnan(c.expected):
            r.numeric_ok = math.isnan(r.computed) and math.isnan(c.expected)
        else:
            r.numeric_ok = abs(r.computed - c.expected) <= c.tol
        missing = tuple(lit for lit in c.printed if lit not in tex)
        r.missing_literals = missing
        r.literal_ok = not missing
        results.append(r)
    return results


def render(results: Sequence[Result], verbose: bool) -> str:
    lines: list[str] = []
    width = max((len(r.claim.cid) for r in results), default=10)
    section = None
    for r in results:
        if r.claim.section != section:
            section = r.claim.section
            lines.append("")
            lines.append(f"--- {section} " + "-" * max(0, 66 - len(section)))
        mark = "ok  " if r.ok else "FAIL"
        got = "n/a" if r.computed is None else f"{r.computed:.6g}"
        line = (f"[{mark}] {r.claim.cid:<{width}} "
                f"expected {r.claim.expected:<12.6g} got {got:<12}")
        if r.error:
            line += f"  !! {r.error}"
        elif not r.numeric_ok:
            line += f"  !! 超出容差 {r.claim.tol:g}"
        elif not r.literal_ok:
            line += f"  !! 论文里找不到字面串 {r.missing_literals}"
        elif verbose:
            line += f"  <- {r.claim.source}"
        lines.append(line)
        if verbose and r.claim.note:
            lines.append(" " * (width + 8) + f"note: {r.claim.note}")
    n_fail = sum(1 for r in results if not r.ok)
    lines.append("")
    lines.append(f"{len(results) - n_fail}/{len(results)} 条通过"
                 + ("" if n_fail == 0 else f"，{n_fail} 条失败"))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="校验论文里的每个头条数字")
    ap.add_argument("--root", default=str(ROOT), help="仓库根目录")
    ap.add_argument("--results-dir", default=DEFAULT_RESULTS)
    ap.add_argument("--robust-dir", default=DEFAULT_ROBUST)
    ap.add_argument("--paper-dir", default="paper")
    ap.add_argument("--only", default=None, help="只跑 id 或 section 含该子串的 claim")
    ap.add_argument("--json", dest="json_out", default=None)
    ap.add_argument("--list", action="store_true", help="只列 claim 清单，不校验")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--skip-literals", action="store_true",
                    help="只做数值复算，跳过 LaTeX 字面检查（例如论文还没写到那一节）")
    args = ap.parse_args(argv)

    root = Path(args.root)
    claims = CLAIMS
    if args.only:
        needle = args.only
        claims = [c for c in claims if needle in c.cid or needle in c.section]
        if not claims:
            print(f"没有 claim 匹配 {needle!r}", file=sys.stderr)
            return 2

    if args.list:
        for c in claims:
            print(f"{c.section:<6} {c.cid:<38} {c.expected:<12.6g} {c.source}")
        print(f"\n共 {len(claims)} 条")
        return 0

    art = Artefacts(root, args.results_dir, args.robust_dir)
    tex = "" if args.skip_literals else load_tex(root / args.paper_dir)
    if args.skip_literals:
        claims = [Claim(**{**c.__dict__, "printed": ()}) for c in claims]

    results = check(claims, art, tex)
    print(render(results, args.verbose))

    if args.json_out:
        payload = {"n_total": len(results),
                   "n_failed": sum(1 for r in results if not r.ok),
                   "results": [r.as_dict() for r in results]}
        Path(args.json_out).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\n[json] {args.json_out}")

    return 1 if any(not r.ok for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
