"""把 ``src/robustness.py`` 的审计产物渲染成论文用的 LaTeX 表。

为什么要有这一步：本项目的纪律是**论文里不出现手打的数字**。
所有表都由脚本从 CSV/JSON 生成，改一次产物、重跑一次脚本，论文自动跟着变，
避免第二轮审计里那种「正文写 Holm、表里其实是 raw p」的错。

用法
----
    python tools/make_robustness_tables.py \\
        --robust-dir artifacts/robustness --out paper/tables_robust
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ARM_TEX = {"none": r"\texttt{none}", "revin": r"\texttt{revin}",
           "san_lite": r"\texttt{san\_lite}", "fredf": r"\texttt{fredf}"}
ARM_ORDER = ["revin", "san_lite", "fredf"]


def _arm(a: str) -> str:
    return ARM_TEX.get(str(a), rf"\texttt{{{str(a).replace('_', chr(92) + '_')}}}")


def _block(b: str) -> str:
    """块标签 ``dataset|pred_len|backbone`` 转成 LaTeX 安全的紧凑写法。

    下划线必须转义，否则任何带下划线的块名都能让编译整体失败；``|`` 换成 ``/``
    以免在 ``\\texttt`` 里被迫切换到数学模式（字体会和相邻文字不一致）。
    """
    return str(b).replace("_", chr(92) + "_").replace("|", "/")


def _p(v: float) -> str:
    """p 值排版：小于 1e-3 用科学计数，其余三位小数；避免 0.0000 这种误导写法。"""
    if not np.isfinite(v):
        return "--"
    if v < 1e-3:
        m, e = f"{v:.1e}".split("e")
        return rf"${m}\!\times\!10^{{{int(e)}}}$"
    return f"{v:.3f}"


def _ci(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "--"
    return f"[{lo:+.2f}, {hi:+.2f}]"


def _wrap(body: str, caption: str, label: str, colspec: str,
          note: str | None = None, star: bool = False) -> str:
    """把表体裹成 float。

    宽度用 ``\\fitwidth``（在 paper/main.tex 导言区定义）而不是裸的
    ``\\resizebox{\\linewidth}{!}``：后者在表格本来就窄的时候会把它*放大*到
    比正文还大，单栏 cas-sc 下尤其难看。``\\fitwidth`` 只在超宽时缩小。
    """
    env = "table*" if star else "table"
    # \centering is active inside the float, which would centre every line of a
    # multi-line note. \parbox restores normal justified paragraph setting.
    note_tex = (rf"\vspace{{3pt}}\parbox{{\linewidth}}{{\footnotesize {note}}}"
                + "\n") if note else ""
    return (
        f"\\begin{{{env}}}[t]\n\\centering\n"
        f"\\caption{{{caption}}}\\label{{{label}}}\n"
        f"\\fitwidth{{%\n"
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{body}\\bottomrule\n"
        f"\\end{{tabular}}}}\n{note_tex}"
        f"\\end{{{env}}}\n"
    )


# --------------------------------------------------------------------------- #
def table_metric_robustness(robust: Path) -> str:
    """表 A：同一批产物在 MSE 与 MAE 上各自复算的配对检验 + 跨指标一致性。"""
    pm = pd.read_csv(robust / "metric_robustness.csv")
    ag = pd.read_csv(robust / "metric_agreement.csv").set_index("method")

    lines = [
        (r"Arm & Metric & $N$ & W/T/L & Median gain (\%) & 95\% CI & $p$ & "
         r"$p_{\text{Holm}}$ & $r$ & Verdict \\"),
        r"\midrule",
    ]
    for arm in ARM_ORDER:
        sub = pm[pm.method == arm]
        if sub.empty:
            continue
        for i, metric in enumerate(["mse", "mae"]):
            r = sub[sub.metric == metric]
            if r.empty:
                continue
            r = r.iloc[0]
            name = _arm(arm) if i == 0 else ""
            verdict = r"\textbf{sig.}" if bool(r.reject_H0) else "n.s."
            sign = "worse" if float(r.median_rel_gain_pct) < 0 else "better"
            lines.append(
                f"{name} & {metric.upper()} & {int(r.n_blocks)} & "
                f"{int(r.win)}/{int(r.tie)}/{int(r.loss)} & "
                f"{float(r.median_rel_gain_pct):+.2f} & "
                f"{_ci(float(r.median_ci_lo), float(r.median_ci_hi))} & "
                f"{_p(float(r.p_value))} & {_p(float(r.p_holm))} & "
                f"{float(r.effect_r):.2f} & {verdict} ({sign}) \\\\"
            )
        if arm in ag.index:
            a = ag.loc[arm]
            flip = r"\textbf{flips}" if bool(a.verdict_flips) else "consistent"
            lines.append(
                rf"\multicolumn{{10}}{{l}}{{\quad\footnotesize cross-metric: "
                rf"Spearman $\rho={float(a.spearman_rho):.2f}$, "
                rf"sign agreement ${100 * float(a.sign_agreement):.0f}\%$, "
                rf"verdict {flip}}} \\"
            )
        lines.append(r"\addlinespace[2pt]")
    body = "\n".join(lines) + "\n"
    note = (r"Paired Wilcoxon signed-rank per arm against \texttt{none}; blocks are "
            r"(dataset $\times$ horizon $\times$ backbone), matched pairwise so that one "
            r"arm's pruning cannot shrink another arm's sample. Holm correction is applied "
            r"within each metric (family size 3). CIs are percentile bootstrap over blocks "
            r"($10^4$ resamples, blocks resampled jointly to preserve pairing). "
            r"Positive gain $=$ lower error than \texttt{none}.")
    return _wrap(body, "Metric robustness of the plug-in verdicts.",
                 "tab:metric-robust", "llrccccrcl", note, star=False)


def table_estimator_lever(robust: Path) -> str:
    """表 D：均值 vs 中位数这第三个「显著性杠杆」，连同左尾风险一起报。

    这张表服务的是 Result 1：完整块过滤是第一个杠杆，指标选择是第二个，
    **中心估计量的选择**是第三个。fredf 在 MSE 与 MAE 上均值与中位数符号相反，
    根因是极少数块出现 $-100\\%$ 量级的回退，所以必须把偏度和左尾占比一起摆出来，
    否则读者无法判断该信哪个数字。
    """
    pm = pd.read_csv(robust / "metric_robustness.csv")

    lines = [
        r"Arm & Metric & Mean (\%) & Median (\%) & Mean$-$Med & Skew & "
        r"$<\!-10\%$ & $<\!-25\%$ & Worst block & Worst (\%) \\",
        r"\midrule",
    ]
    for arm in ARM_ORDER:
        sub = pm[pm.method == arm]
        if sub.empty:
            continue
        for i, metric in enumerate(["mse", "mae"]):
            r = sub[sub.metric == metric]
            if r.empty:
                continue
            r = r.iloc[0]
            name = _arm(arm) if i == 0 else ""
            mean_s = f"{float(r.mean_rel_gain_pct):+.2f}"
            med_s = f"{float(r.median_rel_gain_pct):+.2f}"
            if bool(r.get("mean_median_sign_conflict", False)):
                mean_s, med_s = rf"\textbf{{{mean_s}}}", rf"\textbf{{{med_s}}}"
            lines.append(
                f"{name} & {metric.upper()} & {mean_s} & {med_s} & "
                f"{float(r.mean_minus_median):+.2f} & {float(r['skew']):+.2f} & "
                f"{100 * float(r.frac_worse_than_10pct):.1f}\\% & "
                f"{100 * float(r.frac_worse_than_25pct):.1f}\\% & "
                f"\\footnotesize\\texttt{{{_block(str(r.worst_block))}}} & "
                f"{float(r.worst_rel_gain_pct):+.1f} \\\\"
            )
        lines.append(r"\addlinespace[2pt]")
    body = "\n".join(lines) + "\n"
    note = (r"All columns are computed on the \emph{same} pairwise-matched blocks as "
            r"Table~\ref{tab:metric-robust}; only the summary functional changes. "
            r"Bold rows are sign conflicts, where the mean and the median disagree on "
            r"whether the plug-in helps at all. ``$<\!-10\%$'' is the fraction of blocks "
            r"whose error is more than $10\%$ worse than \texttt{none}. A paired Wilcoxon "
            r"test targets the median, so a paper that reports a mean and tests with "
            r"Wilcoxon is not testing the quantity it reports.")
    return _wrap(body,
                 "The choice of central estimator is a third significance lever.",
                 "tab:estimator-lever", "llrrrrrrlr", note, star=False)


def table_seed_and_cost(robust: Path) -> str:
    """表 B：种子噪声尺度 + 算力代价，两件「便宜但审稿人一定会问」的事放一张表。"""
    import json

    s = json.loads((robust / "robustness_summary.json").read_text(encoding="utf-8"))
    seed_by = {r["method"]: r for r in s["seed_stability"].get("by_method", [])}
    cost_by = {r["method"]: r for r in s.get("cost_benefit", [])}

    lines = [
        r"Arm & $N_{\text{noise}}$ & Within seed noise & Median $|\Delta|/\sigma_{\text{seed}}$"
        r" & Train-time ratio & Median gain (\%) \\",
        r"\midrule",
    ]
    for arm in ARM_ORDER:
        sd, cb = seed_by.get(arm), cost_by.get(arm)
        if sd is None and cb is None:
            continue
        n = int(sd["n"]) if sd else 0
        frac = f"{100 * float(sd['frac_within_noise']):.0f}\\%" if sd else "--"
        ratio = f"{float(sd['median_gain_over_noise']):.2f}" if sd else "--"
        tr = (f"{float(cb['median_train_time_ratio']):.2f}$\\times$"
              if cb and np.isfinite(cb.get("median_train_time_ratio", np.nan)) else "--")
        gain = f"{float(cb['median_rel_gain_pct']):+.2f}" if cb else "--"
        lines.append(f"{_arm(arm)} & {n} & {frac} & {ratio} & {tr} & {gain} \\\\")
    body = "\n".join(lines) + "\n"
    ss = s["seed_stability"]
    note = (rf"Left half: for the {int(ss['n_blocks_with_noise'])} block--arm pairs where both "
            rf"the arm and \texttt{{none}} have $\geq 2$ seeds, the noise scale is "
            rf"$\sigma_{{\text{{seed}}}}=\sqrt{{s^2_{{\text{{arm}}}}+s^2_{{\text{{none}}}}}}$ "
            rf"(median $\sigma_{{\text{{seed}}}}={float(ss['median_seed_sd']):.4f}$ MSE). "
            rf"``Within seed noise'' is the fraction of blocks whose entire plug-in effect is "
            rf"smaller than that scale, i.e.\ a different seed could flip its sign. "
            rf"Overall {100 * float(ss['frac_within_noise']):.0f}\% of block-level effects and "
            rf"{100 * float(ss['frac_wins_within_noise']):.0f}\% of the wins fall in this regime. "
            rf"Right half: training wall-clock ratio against \texttt{{none}} in the same block.")
    return _wrap(body, "Plug-in effects against seed noise and against their compute cost.",
                 "tab:seed-cost", "lrrrrr", note)


def table_leadtime(robust: Path) -> str:
    """表 C：lead-time 四分位上的 argmin 臂稳定性（分层）+ 增益的步长剖面。"""
    import json

    st = pd.read_csv(robust / "leadtime_argmin_stability.csv")
    st["backbone"] = st.block.str.split("|").str[2]
    s = json.loads((robust / "robustness_summary.json").read_text(encoding="utf-8"))
    gq = pd.DataFrame(s["leadtime"]["gain_by_quarter"])

    lines = [r"\multicolumn{5}{l}{\emph{(a) Does one arm win across all four lead-time"
             r" quarters?}} \\", r"\midrule",
             r"Stratum & Blocks & Constant argmin & Mean \# distinct & "
             r"Whole-horizon winner suboptimal somewhere \\", r"\midrule"]
    for k, sub in st.groupby("n_arms_available"):
        lines.append(
            f"{int(k)}-arm pool & {len(sub)} & {100 * sub.constant_argmin.mean():.0f}\\% & "
            f"{sub.n_distinct_argmin.mean():.2f} & "
            f"{100 * (sub.n_quarters_where_whole_best_suboptimal > 0).mean():.0f}\\% \\\\")
    lines.append(r"\addlinespace[2pt]")
    for bk, sub in st.groupby("backbone"):
        lines.append(
            rf"\quad\footnotesize {bk} & {len(sub)} & "
            rf"{100 * sub.constant_argmin.mean():.0f}\% & "
            rf"{sub.n_distinct_argmin.mean():.2f} & -- \\")
    lines.append(r"\addlinespace[4pt]")
    lines.append(r"\multicolumn{5}{l}{\emph{(b) Median relative gain (\%) by lead-time"
                 r" quarter}} \\")
    lines.append(r"\midrule")
    lines.append(r"Arm & Q1 (near) & Q2 & Q3 & Q4 (far) \\")
    lines.append(r"\midrule")
    for arm in ARM_ORDER:
        sub = gq[gq.method == arm].set_index("quarter")
        if sub.empty:
            continue
        cells = " & ".join(f"{float(sub.loc[q, 'median']):+.2f}" if q in sub.index else "--"
                           for q in (1, 2, 3, 4))
        lines.append(f"{_arm(arm)} & {cells} \\\\")

    # ---- (c) 这条轴是**合法**的，所以它的 oracle 上界才是真正的判决 ---------- #
    lt = s["leadtime"]
    lines.append(r"\addlinespace[4pt]")
    lines.append(r"\multicolumn{5}{l}{\emph{(c) Headroom of the one causally legal "
                 r"selection axis}} \\")
    lines.append(r"\midrule")
    lines.append(r"Reference & Mean (\%) & Median (\%) & Max (\%) & Blocks with gain \\")
    lines.append(r"\midrule")
    lines.append(
        rf"\texttt{{none}} & "
        rf"{float(lt['leadtime_oracle_vs_control_mean_pct']):+.2f} & "
        rf"{float(lt['leadtime_oracle_vs_control_median_pct']):+.2f} & -- & -- \\")
    lines.append(
        rf"\textbf{{best fixed arm}} & "
        rf"$\mathbf{{{float(lt['leadtime_oracle_vs_best_fixed_mean_pct']):+.2f}}}$ & "
        rf"$\mathbf{{{float(lt['leadtime_oracle_vs_best_fixed_median_pct']):+.2f}}}$ & "
        rf"{float(lt['leadtime_oracle_vs_best_fixed_max_pct']):+.2f} & "
        rf"{int(lt['n_blocks_leadtime_oracle_beats_best_fixed'])}/"
        rf"{int(lt['n_blocks_multi_arm'])} "
        rf"({100 * float(lt['frac_blocks_leadtime_gain_over_0p5pp']):.0f}\% "
        rf"$>0.5$\,pp) \\")
    body = "\n".join(lines) + "\n"
    note = (r"\texttt{seg}$_i$ errors partition the \emph{forecast horizon} into quarters "
            r"(per-timestep MSE averaged within each quarter), not the test period; the four "
            r"quarter errors average exactly to the block MSE, which is what makes panel (c) "
            r"a valid decomposition. "
            r"Stratifying by arm-pool size is necessary: a 4-arm block has more chances to "
            r"switch than a 3-arm block, so the two strata must not be pooled. "
            r"This audit covers all 128 blocks and all four backbones, including TimesNet, "
            r"which the per-window analysis cannot reach. "
            r"Panel (c) is the decisive one: unlike the per-window axis, the lead-time index "
            r"is known at decision time, so switching arms along it needs no feedback and is "
            r"causally legal --- but even its \emph{oracle} buys almost nothing over a single "
            r"frozen arm.")
    return _wrap(body, "Arm identity is unstable along the forecast horizon, yet the "
                       "legal axis it opens is empty.",
                 "tab:leadtime", "lrrrr", note)


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="渲染增量审计的 LaTeX 表")
    ap.add_argument("--robust-dir", default="artifacts/robustness")
    ap.add_argument("--out", default="paper/tables_robust")
    args = ap.parse_args()

    robust, out = Path(args.robust_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = {
        "tbl_metric_robustness.tex": table_metric_robustness,
        "tbl_estimator_lever.tex": table_estimator_lever,
        "tbl_seed_cost.tex": table_seed_and_cost,
        "tbl_leadtime.tex": table_leadtime,
    }
    for name, fn in jobs.items():
        (out / name).write_text(fn(robust), encoding="utf-8")
        print(f"[tables] {out / name}")


if __name__ == "__main__":
    main()
