"""从结果 CSV 自动生成投稿用的 LaTeX 表格与图。

产出（默认到 artifacts/report/）
--------------------------------
tables/
  main_<backbone>.tex        主对比表：行 = 数据集 × horizon，列 = 插件（MSE，行内最优加粗）
  gain_matrix.tex            增益矩阵表：骨干 × 数据集 的平均相对增益 %（带符号，负号=变差）
  decision_gain.tex          selector 决策收益表（selector / 固定策略 / 不挂 / oracle）
  ablation_features.tex      特征组消融表
  stats_wilcoxon.tex         Wilcoxon + Holm 检验表
  stats_ranks.tex            Friedman 平均秩 + CD
  efficiency.tex             效率表（新增参数量、显存、ms/iter 相对开销）
  horizon_criteria.tex       horizon-aware 选择准则对比表
figs/
  gain_heatmap.png           增益热图（每骨干一个面板）
  conditionality.png         「特征 → 增益」散点：本文核心论点的可视化
  decision_policies.png      策略收益柱状图
report.tex                   把上面所有表/图 \\input 进来的骨架文件

用法
----
    python -m src.report --results results/results.csv --features artifacts/features.csv \
        --selector-dir artifacts/selector --stats-dir artifacts/stats --out artifacts/report
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BLOCK_KEYS = ["dataset", "pred_len", "backbone"]


# --------------------------------------------------------------------------- #
# LaTeX 基础
# --------------------------------------------------------------------------- #
def _esc(s: Any) -> str:
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def latex_table(df: pd.DataFrame, caption: str, label: str, float_fmt: str = "{:.3f}",
                bold_min_rows: bool = False, index: bool = True) -> str:
    """把 DataFrame 渲染成 booktabs 风格的 LaTeX 表（可选行内最小值加粗）。

    2026-09-14 code review 修复项 #5：**含缺失格的行不加粗**。
    旧实现只要求 ``v.notna().any()``，于是被 ``skip_rules`` 剪掉 revin 的行（主表里每张 28 个 ``--``）
    也会在剩下 3 个插件之间评最优，而无缺失行是在 4 个插件之间评——caption 写「行内最优加粗」，
    读者会把加粗数量当胜出次数读，缺失的方法就白白「不参与竞争」（跑挂/被剪 = 免于失败）。
    因此改成 ``notna().all()``：只有全部方法都有值的行才加粗，缺格行留给读者自己看数值。
    """
    d = df.copy()
    num_cols = [c for c in d.columns if pd.api.types.is_numeric_dtype(d[c])]
    body: list[list[str]] = []
    for i in range(len(d)):
        row = []
        if index:
            idx = d.index[i]
            row.append(_esc(" ".join(map(str, idx)) if isinstance(idx, tuple) else idx))
        vals = d.iloc[i]
        mins = {}
        if bold_min_rows and num_cols:
            v = vals[num_cols].astype(float)
            if v.notna().all():          # 缺一格就整行不加粗：加粗必须是同一竞争集内的比较
                mins = {v.idxmin()}
        for c in d.columns:
            x = vals[c]
            if c in num_cols and pd.notna(x):
                txt = float_fmt.format(float(x))
                if c in mins:
                    txt = r"\textbf{" + txt + "}"
            else:
                txt = _esc(x) if pd.notna(x) else "--"
            row.append(txt)
        body.append(row)

    header = ([_esc(d.index.name or "")] if index else []) + [_esc(c) for c in d.columns]
    align = ("l" * (1 if index else 0)) + "r" * len(d.columns)
    lines = [
        r"\begin{table}[htbp]", r"\centering", r"\small",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{align}}}", r"\toprule",
        " & ".join(header) + r" \\", r"\midrule",
    ]
    lines += [" & ".join(r) + r" \\" for r in body]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# 表格
# --------------------------------------------------------------------------- #
def main_tables(results: pd.DataFrame, out_dir: Path, metric: str = "mse") -> list[Path]:
    """每个骨干一张主对比表：行 = 数据集 × horizon，列 = 插件。

    caption 里必须写清「缺格行不加粗」（2026-09-14 code review 修复项 #5）：
    revin 在 3 个骨干上被 skip_rules 剪成 `--`，若这些行仍加粗，读者数加粗次数就等于
    数了一个「缺席方法免于失败」的胜负表。
    """
    paths = []
    for bk, g in results.groupby("backbone"):
        piv = g.pivot_table(index=["dataset", "pred_len"], columns="plugin", values=metric)
        piv.index.name = "dataset / horizon"
        tex = latex_table(
            piv,
            caption=(f"{bk} 上各插件的测试 {metric.upper()}（行内最优加粗，"
                     f"**仅对所有插件均有结果的行**加粗；含 -- 的行不加粗，因为可用列不可比；"
                     f"所有 baseline 均由本文在统一协议 drop\\_last=False 下重跑）"),
            label=f"tab:main_{bk.lower()}", bold_min_rows=True)
        paths.append(_write(out_dir / f"main_{bk}.tex", tex))
    return paths


def gain_matrix(results: pd.DataFrame, control: str = "none", metric: str = "mse") -> pd.DataFrame:
    """(骨干 × 数据集 × 插件) 的平均相对增益 %（正 = 误差下降）。"""
    agg = results.groupby(BLOCK_KEYS + ["plugin"])[metric].mean().reset_index()
    piv = agg.pivot_table(index=BLOCK_KEYS, columns="plugin", values=metric)
    rel = piv.apply(lambda c: (piv[control] - c) / piv[control] * 100)
    rel = rel.drop(columns=[control]).reset_index()
    return rel.groupby(["backbone", "dataset"]).mean(numeric_only=True).drop(columns=["pred_len"])


def gain_matrix_table(results: pd.DataFrame, out_dir: Path, control: str = "none") -> Path:
    gm = gain_matrix(results, control)
    tex = latex_table(
        gm, caption="增益矩阵：各插件相对无插件基线的平均相对 MSE 降低 %（负值 = 挂上反而更差，"
                    "本文的核心观察正是这些负格子）",
        label="tab:gain_matrix", float_fmt="{:+.2f}")
    return _write(out_dir / "gain_matrix.tex", tex)


def decision_gain_table(selector_dir: Path, out_dir: Path) -> Path | None:
    f = selector_dir / "decision_gain.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f)
    cols = ["policy", "mean_mse", "mean_rel_gain_pct", "mean_regret_vs_oracle_pct", "harmful_rate"]
    tex = latex_table(df[cols].set_index("policy"),
                      caption="决策收益：按 selector 选择 vs 永远挂固定插件 vs 永远不挂 vs oracle 上界"
                              "（selector 使用 leave-one-dataset-out 预测，无数据泄漏）",
                      label="tab:decision_gain", float_fmt="{:.3f}")
    return _write(out_dir / "decision_gain.tex", tex)


def ablation_table(selector_dir: Path, out_dir: Path) -> Path | None:
    f = selector_dir / "feature_group_ablation.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f).set_index("ablation")
    tex = latex_table(df, caption="特征组消融：逐组移除特征后 selector 的 LODO 准确率与决策收益",
                      label="tab:ablation_features", float_fmt="{:.3f}")
    return _write(out_dir / "ablation_features.tex", tex)


def stats_tables(stats_dir: Path, out_dir: Path) -> list[Path]:
    paths = []
    f = stats_dir / "wilcoxon_vs_control.csv"
    if f.exists():
        df = pd.read_csv(f)
        # n_eff（去零差值后进入检验的对数）必须和 n_blocks 一起报：2026-09-14 code review
        # 修复项 #1/#2——n_blocks 是「该对可配对的块数」（逐对 dropna，不是矩阵总行数），
        # n_eff 才是效应量 r 的分母，两者混用会让审稿人算不平。
        cols = [c for c in ["method", "n_blocks", "n_eff", "win", "tie", "loss",
                            "mean_rel_gain_pct", "effect_r",
                            "p_value", "p_holm_adjusted", "reject_H0"] if c in df.columns]
        tex = latex_table(df[cols].set_index("method"),
                          caption="配对 Wilcoxon signed-rank（块 = 数据集 × horizon × 骨干）+ Holm 校正。"
                                  "k=2 的比较刻意不用 Friedman/Nemenyi（后者在小 N 下功效不足）。"
                                  "n\\_blocks = 该对方法都有结果的配对块数（逐对取，故各行可不同）；"
                                  "n\\_eff = 去掉零差值后进入检验的对数，效应量 r = Z/$\\sqrt{n\\_eff}$",
                          label="tab:stats_wilcoxon", float_fmt="{:.4g}")
        paths.append(_write(out_dir / "stats_wilcoxon.tex", tex))
    f = stats_dir / "avg_ranks.csv"
    if f.exists():
        df = pd.read_csv(f, index_col=0)
        tex = latex_table(df, caption="Friedman 平均秩（越小越好）；CD 图见图~\\ref{fig:cd}",
                          label="tab:stats_ranks", float_fmt="{:.3f}")
        paths.append(_write(out_dir / "stats_ranks.tex", tex))
    return paths


def efficiency_table(results: pd.DataFrame, out_dir: Path, control: str = "none") -> Path | None:
    need = {"extra_params", "ms_per_iter"}
    if not need.issubset(results.columns):
        return None
    base = results[results.plugin == control].groupby(["backbone"])["ms_per_iter"].mean()
    rows = []
    for (bk, pl), g in results.groupby(["backbone", "plugin"]):
        b = base.get(bk, np.nan)
        rows.append({
            "backbone": bk, "plugin": pl,
            "extra_params": float(g["extra_params"].mean()),
            "ms_per_iter": float(g["ms_per_iter"].mean()),
            "time_overhead_pct": float((g["ms_per_iter"].mean() / b - 1) * 100) if b else np.nan,
            "peak_mem_mib": float(g["peak_mem_mib"].mean()) if "peak_mem_mib" in g else np.nan,
        })
    df = pd.DataFrame(rows).set_index(["backbone", "plugin"])
    tex = latex_table(df, caption="效率表（单张 V100）：插件新增参数量、每迭代耗时相对基线的开销、峰值显存",
                      label="tab:efficiency", float_fmt="{:.2f}")
    return _write(out_dir / "efficiency.tex", tex)


def horizon_table(horizon_dir: Path, out_dir: Path) -> Path | None:
    f = horizon_dir / "criteria_comparison.csv"
    if not f.exists():
        return None
    df = pd.read_csv(f).set_index("criterion")
    tex = latex_table(df, caption="horizon-aware checkpoint 选择准则对比（零算力增量：同一批训练日志的不同读法）",
                      label="tab:horizon_criteria", float_fmt="{:.4g}")
    return _write(out_dir / "horizon_criteria.tex", tex)


# --------------------------------------------------------------------------- #
# 图（图内文字统一用英文：投稿用，且避免环境缺 CJK 字体导致乱码）
# --------------------------------------------------------------------------- #
def fig_gain_heatmap(results: pd.DataFrame, out_path: Path, control: str = "none") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gm = gain_matrix(results, control)
    backbones = sorted({i[0] for i in gm.index})
    fig, axes = plt.subplots(1, len(backbones), figsize=(4.2 * len(backbones), 3.4), squeeze=False)
    vmax = float(np.nanmax(np.abs(gm.to_numpy(float)))) or 1.0
    for ax, bk in zip(axes[0], backbones):
        sub = gm.loc[bk]
        im = ax.imshow(sub.to_numpy(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(sub.shape[1]), sub.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(sub.shape[0]), sub.index, fontsize=8)
        ax.set_title(bk, fontsize=10)
        for i in range(sub.shape[0]):
            for j in range(sub.shape[1]):
                v = sub.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.85, label="relative MSE reduction (%)")
    fig.suptitle("Conditionality of plug-in gains (blue = helps, red = hurts)", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_conditionality(results: pd.DataFrame, features: pd.DataFrame, out_path: Path,
                       feature: str = "hz_mean_shift", control: str = "none") -> Path:
    """核心论点图：某个复杂度特征 vs 各插件的相对增益（含线性拟合与相关系数）。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    agg = results.groupby(BLOCK_KEYS + ["plugin"])["mse"].mean().reset_index()
    piv = agg.pivot_table(index=BLOCK_KEYS, columns="plugin", values="mse")
    rel = piv.apply(lambda c: (piv[control] - c) / piv[control] * 100).drop(columns=[control])
    rel = rel.reset_index().merge(features[["dataset", "pred_len", feature]],
                                  on=["dataset", "pred_len"], how="left")
    plugins = [c for c in rel.columns if c not in BLOCK_KEYS + [feature]]
    fig, axes = plt.subplots(1, len(plugins), figsize=(4.0 * len(plugins), 3.2), squeeze=False)
    for ax, pl in zip(axes[0], plugins):
        d = rel[[feature, pl, "backbone"]].dropna()
        for bk, gb in d.groupby("backbone"):
            ax.scatter(gb[feature], gb[pl], s=16, label=bk, alpha=0.75)
        if len(d) > 2:
            k, b = np.polyfit(d[feature].to_numpy(float), d[pl].to_numpy(float), 1)
            xs = np.linspace(d[feature].min(), d[feature].max(), 20)
            r = float(np.corrcoef(d[feature], d[pl])[0, 1])
            ax.plot(xs, k * xs + b, "k--", lw=1, label=f"fit (r={r:.2f})")
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xlabel(feature, fontsize=9)
        ax.set_ylabel("relative MSE reduction (%)", fontsize=9)
        ax.set_title(pl, fontsize=10)
        ax.legend(fontsize=7)
    fig.suptitle("Can complexity features predict the sign/magnitude of plug-in gains?", fontsize=11)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def fig_decision_policies(selector_dir: Path, out_path: Path) -> Path | None:
    f = selector_dir / "decision_gain.csv"
    if not f.exists():
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(f).sort_values("mean_rel_gain_pct")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    colors = ["tab:green" if p == "selector_lodo" else
              ("tab:gray" if p == "oracle" else "tab:blue") for p in df.policy]
    ax.barh(df.policy, df.mean_rel_gain_pct, color=colors)
    ax.set_xlabel("mean MSE reduction vs. always-none (%)")
    ax.set_title("Decision policy gains")
    for y, (v, r) in enumerate(zip(df.mean_rel_gain_pct, df.mean_regret_vs_oracle_pct)):
        ax.text(v, y, f" {v:+.2f}% (regret {r:.2f}%)", va="center", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #
def build(results: pd.DataFrame, features: pd.DataFrame | None, out_dir: str | Path,
          selector_dir: str | Path | None = None, stats_dir: str | Path | None = None,
          horizon_dir: str | Path | None = None, control: str = "none") -> dict[str, Any]:
    out = Path(out_dir)
    tdir, fdir = out / "tables", out / "figs"
    made: dict[str, Any] = {"tables": [], "figs": []}

    made["tables"] += [str(p) for p in main_tables(results, tdir)]
    made["tables"].append(str(gain_matrix_table(results, tdir, control)))
    for fn, arg in ((decision_gain_table, selector_dir), (ablation_table, selector_dir),
                    (horizon_table, horizon_dir)):
        if arg:
            p = fn(Path(arg), tdir)
            if p:
                made["tables"].append(str(p))
    if stats_dir:
        made["tables"] += [str(p) for p in stats_tables(Path(stats_dir), tdir)]
    p = efficiency_table(results, tdir, control)
    if p:
        made["tables"].append(str(p))

    made["figs"].append(str(fig_gain_heatmap(results, fdir / "gain_heatmap.png", control)))
    if features is not None:
        made["figs"].append(str(fig_conditionality(results, features,
                                                   fdir / "conditionality.png", control=control)))
    if selector_dir:
        p2 = fig_decision_policies(Path(selector_dir), fdir / "decision_policies.png")
        if p2:
            made["figs"].append(str(p2))

    # report.tex 骨架
    lines = [r"% 由 src/report.py 自动生成，勿手改；重跑即可更新",
             r"\section{实验结果}"]
    for t in made["tables"]:
        lines.append(rf"\input{{tables/{Path(t).name}}}")
    for f in made["figs"]:
        name = Path(f).stem
        lines += [r"\begin{figure}[htbp]", r"\centering",
                  rf"\includegraphics[width=\linewidth]{{figs/{Path(f).name}}}",
                  rf"\caption{{{name}}}", rf"\label{{fig:{name}}}", r"\end{figure}"]
    _write(out / "report.tex", "\n".join(lines) + "\n")
    print(f"[report] {len(made['tables'])} 张表 + {len(made['figs'])} 张图 -> {out}")
    return made


def main() -> None:
    ap = argparse.ArgumentParser(description="生成投稿用 LaTeX 表格与图")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--features", default="artifacts/features.csv")
    ap.add_argument("--selector-dir", default="artifacts/selector")
    ap.add_argument("--stats-dir", default="artifacts/stats")
    ap.add_argument("--horizon-dir", default="artifacts/horizon")
    ap.add_argument("--out", default="artifacts/report")
    ap.add_argument("--control", default="none")
    args = ap.parse_args()

    results = pd.read_csv(args.results)
    features = pd.read_csv(args.features) if Path(args.features).exists() else None
    build(results, features, args.out, args.selector_dir, args.stats_dir,
          args.horizon_dir, args.control)


if __name__ == "__main__":
    main()
