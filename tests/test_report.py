"""`src/report.py` 自检：LaTeX 转义、行内加粗、增益符号约定、端到端产出。

论文里最容易出错的是**符号约定**：增益矩阵里的正数必须表示「误差下降」。
这里用一个人工构造的、增益方向已知的结果表把它钉死。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import report as R


def make_results(seed: int = 0) -> pd.DataFrame:
    """revin 一律更好 3%，san_lite 一律更差 2%，fredf 无差异。"""
    del seed  # 造数完全确定，保留形参只为调用点可读
    rows = []
    for ds in ("ETTh1", "Weather"):
        for h in (96, 336):
            for bk in ("DLinear", "PatchTST"):
                base = 0.4 + 0.05 * (h == 336)
                for pl, g in (("none", 0.0), ("revin", 0.03), ("san_lite", -0.02), ("fredf", 0.0)):
                    rows.append({"dataset": ds, "pred_len": h, "backbone": bk, "plugin": pl,
                                 "seed": 2021, "mse": base * (1 - g), "mae": base * 0.8,
                                 "extra_params": 0 if pl != "revin" else 14,
                                 "peak_mem_mib": 500 + 10 * (pl != "none"),
                                 "ms_per_iter": 10 * (1 + 0.05 * (pl != "none"))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# LaTeX 基础
# --------------------------------------------------------------------------- #
def test_latex_escaping_and_structure():
    df = pd.DataFrame({"a_b": [1.0], "c%": [2.0]}, index=pd.Index(["x_1"], name="idx_name"))
    tex = R.latex_table(df, caption="标题 with_underscore", label="tab:t")
    assert r"\begin{table}" in tex and r"\bottomrule" in tex
    assert r"a\_b" in tex and r"c\%" in tex and r"x\_1" in tex
    assert tex.count(r"\\") >= 2


def test_latex_bold_min_per_row():
    df = pd.DataFrame({"none": [0.5, 0.3], "revin": [0.4, 0.35]}, index=["r1", "r2"])
    tex = R.latex_table(df, "c", "l", bold_min_rows=True)
    body = [ln for ln in tex.splitlines() if ln.startswith("r")]
    assert r"\textbf{0.400}" in body[0]          # 第一行 revin 更好
    assert r"\textbf{0.300}" in body[1]          # 第二行 none 更好
    assert body[0].count(r"\textbf") == 1


def test_latex_handles_nan_as_dash():
    df = pd.DataFrame({"v": [np.nan]}, index=["r"])
    assert "--" in R.latex_table(df, "c", "l")


# --------------------------------------------------------------------------- #
# 2026-09-14 code review 修复项 #5：含缺失格的行不加粗
# --------------------------------------------------------------------------- #
def test_latex_bold_skips_rows_with_missing_cells():
    """缺格行若仍加粗，就是「只在剩下的列之间评最优」——缺席的方法白白免于竞争。

    第 2 行缺 revin，旧实现会把 0.300（none）加粗，让读者以为 none 在 4 个插件里最优。
    """
    df = pd.DataFrame({"none": [0.5, 0.3], "revin": [0.4, np.nan]}, index=["r1", "r2"])
    tex = R.latex_table(df, "c", "l", bold_min_rows=True)
    body = [ln for ln in tex.splitlines() if ln.startswith("r")]
    assert r"\textbf{0.400}" in body[0]              # 完整行照常加粗
    assert r"\textbf" not in body[1]                 # 缺格行整行不加粗
    assert "--" in body[1]
    assert tex.count(r"\textbf") == 1


def test_latex_bold_all_nan_row_is_untouched():
    df = pd.DataFrame({"a": [np.nan], "b": [np.nan]}, index=["r"])
    tex = R.latex_table(df, "c", "l", bold_min_rows=True)
    assert r"\textbf" not in tex


def test_main_tables_do_not_bold_rows_pruned_by_skip_rules(tmp_path):
    """复刻主表形态：revin 只在 {96} 上保留验证子集，其余行渲染成 `--`。

    4 行里只有 1 行是完整的 → 全表只能有 1 个加粗；旧实现会给出 4 个，
    读者数加粗次数就等于数了一张「缺席方法免于失败」的胜负表。
    """
    rows = []
    for ds in ("ETTh1", "Weather"):
        for h in (96, 720):
            base = 0.4 + 0.05 * (h == 720)
            for pl, g in (("none", 0.0), ("fredf", 0.01), ("san_lite", -0.01)):
                rows.append({"dataset": ds, "pred_len": h, "backbone": "PatchTST",
                             "plugin": pl, "seed": 2021, "mse": base * (1 - g)})
            if (ds, h) == ("ETTh1", 96):            # 只有这一格保留了 revin
                rows.append({"dataset": ds, "pred_len": h, "backbone": "PatchTST",
                             "plugin": "revin", "seed": 2021, "mse": base * 0.9})
    paths = R.main_tables(pd.DataFrame(rows), tmp_path)
    txt = paths[0].read_text(encoding="utf-8")
    body = [ln for ln in txt.splitlines() if ln.startswith(("ETTh1", "Weather"))]
    assert len(body) == 4
    assert txt.count(r"\textbf") == 1                # 只有 ETTh1 96 这一整行可比
    complete = [ln for ln in body if "--" not in ln]
    assert len(complete) == 1 and r"\textbf" in complete[0]
    for ln in body:
        if "--" in ln:
            assert r"\textbf" not in ln
    assert "仅对所有插件均有结果的行" in txt          # caption 必须把口径说清楚


# --------------------------------------------------------------------------- #
# 增益矩阵的符号约定
# --------------------------------------------------------------------------- #
def test_gain_matrix_sign_convention():
    gm = R.gain_matrix(make_results())
    assert set(gm.columns) == {"revin", "san_lite", "fredf"}     # 对照列被去掉
    assert gm["revin"].min() > 2.9                               # 正 = 误差下降 3%
    assert gm["san_lite"].max() < -1.9                           # 负 = 挂上更差
    assert abs(gm["fredf"]).max() < 1e-9
    assert gm.index.names == ["backbone", "dataset"]


def test_main_tables_one_per_backbone(tmp_path):
    paths = R.main_tables(make_results(), tmp_path)
    assert {p.name for p in paths} == {"main_DLinear.tex", "main_PatchTST.tex"}
    txt = paths[0].read_text(encoding="utf-8")
    assert "drop\\_last=False" in txt            # 提醒不能抄原论文数字
    assert txt.count(r"\textbf") == 4            # 4 行（2 数据集 × 2 horizon）各一个最优


def test_stats_wilcoxon_table_reports_both_sample_sizes(tmp_path):
    """论文表必须同时给出 n_blocks（逐对配对块）与 n_eff（进入检验的非零差值对）。

    2026-09-14 code review 修复项 #1/#2 的产出口径：只报一个数，读者就会把 44 或 128
    当成效应量 r 的分母。
    """
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    pd.DataFrame([
        {"method": "fredf", "control": "none", "n_blocks": 128, "n_eff": 120,
         "win": 90, "tie": 8, "loss": 30, "mean_rel_gain_pct": 3.6, "effect_r": 0.31,
         "p_value": 0.0058, "p_holm_adjusted": 0.0175, "reject_H0": True},
    ]).to_csv(stats_dir / "wilcoxon_vs_control.csv", index=False)
    paths = R.stats_tables(stats_dir, tmp_path / "tables")
    txt = [p for p in paths if p.name == "stats_wilcoxon.tex"][0].read_text(encoding="utf-8")
    assert r"n\_blocks" in txt and r"n\_eff" in txt
    assert "120" in txt                                  # n_eff 必须真的进表
    assert "逐对取" in txt and "去掉零差值" in txt        # caption 说明两个口径


def test_efficiency_table_reports_overhead(tmp_path):
    p = R.efficiency_table(make_results(), tmp_path)
    assert p is not None and p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "revin" in txt and "san\\_lite" in txt


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
def test_build_end_to_end_with_all_upstream_artifacts(tmp_path):
    """把 stats / selector / horizon 三个上游目录都喂进去，产出完整报告骨架。"""
    from src import horizon as HZ
    from src import selector as SEL
    from src import stats as ST

    results = ST.demo_results()
    features = pd.DataFrame([
        {"dataset": ds, "pred_len": h, "pc_perm_entropy": 0.9, "ns_mean_drift": 0.4,
         "fr_topk_energy": 0.5, "ch_eff_rank_ratio": 0.3, "sd_seasonal_strength": 0.6,
         "hz_mean_shift": 0.2 * h / 96}
        for ds in results["dataset"].unique() for h in (96, 192, 336, 720)
    ])

    stats_dir, sel_dir, hz_dir = tmp_path / "stats", tmp_path / "sel", tmp_path / "hz"
    ST.run_pipeline(results, stats_dir)
    SEL.run(features, results, sel_dir, tau=0.005, ablation=True)

    logs = pd.concat([
        pd.DataFrame([{"cell_id": f"c{c}", "epoch": e,
                       "val_mse": 0.5 + 0.01 * (e - 2) ** 2, "test_mse": 0.52 + 0.01 * (e - 3) ** 2,
                       "val_seg1_mse": 0.4 + 0.01 * (e - 1) ** 2,
                       "val_seg2_mse": 0.6 + 0.01 * (e - 4) ** 2} for e in range(6)])
        for c in range(5)
    ])
    HZ.run(logs, hz_dir)

    made = R.build(results, features, tmp_path / "report", sel_dir, stats_dir, hz_dir)
    names = {p.rsplit("/", 1)[-1] for p in made["tables"]}
    assert {"main_DLinear.tex", "gain_matrix.tex", "decision_gain.tex",
            "ablation_features.tex", "stats_wilcoxon.tex", "stats_ranks.tex",
            "horizon_criteria.tex"} <= names
    assert len(made["figs"]) == 3
    for f in made["figs"]:
        assert (tmp_path / "report" / "figs" / f.rsplit("/", 1)[-1]).stat().st_size > 5000

    skeleton = (tmp_path / "report" / "report.tex").read_text(encoding="utf-8")
    for t in names:
        assert rf"\input{{tables/{t}}}" in skeleton
    assert skeleton.count(r"\begin{figure}") == 3


def test_build_without_optional_dirs(tmp_path):
    """只有结果表时也必须能产出主表与增益矩阵（早期实验阶段的常见情形）。"""
    made = R.build(make_results(), None, tmp_path)
    assert len(made["tables"]) >= 3 and len(made["figs"]) == 1
    assert (tmp_path / "report.tex").exists()
