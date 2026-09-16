"""`src/stats.py` 自检。

除了常规正确性，这里**钉死两个论文里被反复引用的数字**：
- k=5, N=9（只按数据集分块）时 CD≈2.03 → 平均秩几乎不可能显著；
- k=4, N=128（按 数据集×horizon×骨干 分块）时 CD≈0.41 → 检验力足够。
这就是本项目把块定义为三元组的定量理由，测试挂了说明文档的论证也需要同步修改。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import stats as S


# --------------------------------------------------------------------------- #
# CD 阈值与块定义
# --------------------------------------------------------------------------- #
def test_nemenyi_cd_reference_values():
    assert S.nemenyi_cd(5, 9) == pytest.approx(2.03, abs=0.02)
    # 论文 §3 印的两个值：8 个数据集当块 vs 128 个三元组块
    assert S.nemenyi_cd(4, 8) == pytest.approx(1.66, abs=0.01)
    assert S.nemenyi_cd(4, 128) == pytest.approx(0.415, abs=0.01)
    # §4.1 完整块子集上的 CD
    assert S.nemenyi_cd(4, 44) == pytest.approx(0.71, abs=0.01)
    # 同样的 k，N 翻 4 倍 → CD 减半（1/sqrt(N)）
    assert S.nemenyi_cd(4, 36) / S.nemenyi_cd(4, 144) == pytest.approx(2.0, rel=1e-6)


def test_block_definition_gains_power():
    """三元组块定义把 CD 缩小 √16 = 4 倍，这是本项目块定义的定量依据。"""
    cd_dataset_only = S.nemenyi_cd(4, 8)
    cd_triple = S.nemenyi_cd(4, 8 * 4 * 4)
    assert cd_dataset_only / cd_triple == pytest.approx(4.0, rel=1e-9)
    assert cd_dataset_only > 1.6 and cd_triple < 0.5


def test_cd_table_shape_and_monotonicity():
    t = S.cd_table()
    assert t.shape == (6, 6)
    for k in t.index:  # N 越大 CD 越小
        assert list(t.loc[k]) == sorted(t.loc[k], reverse=True)


def test_build_block_matrix_keys_and_seed_averaging():
    rows = []
    for seed in (2021, 2022):
        for pl, v in [("none", 0.5), ("revin", 0.4)]:
            rows.append({"dataset": "ETTh1", "pred_len": 96, "backbone": "DLinear",
                         "plugin": pl, "seed": seed, "mse": v + 0.1 * (seed == 2022)})
    mat = S.build_block_matrix(pd.DataFrame(rows))
    assert mat.shape == (1, 2)
    assert mat.index[0] == "ETTh1|96|DLinear"       # 块 = 数据集 × horizon × 骨干
    assert mat.loc["ETTh1|96|DLinear", "none"] == pytest.approx(0.55)  # 多 seed 取均值


def test_build_block_matrix_keeps_incomplete_blocks_by_default():
    """2026-09-14 code review 修复项 #1：默认 **不再**整表 dropna。

    旧默认 require_complete=True 会把「任一方法缺格」的块从所有配对比较里删掉，
    于是 revin 的剪枝规则决定了 fredf/san_lite 的检验样本。配对检验只需逐对可用。
    """
    rows = [
        {"dataset": "A", "pred_len": 96, "backbone": "B", "plugin": "none", "seed": 1, "mse": 0.5},
        {"dataset": "A", "pred_len": 96, "backbone": "B", "plugin": "revin", "seed": 1, "mse": 0.4},
        {"dataset": "A", "pred_len": 192, "backbone": "B", "plugin": "none", "seed": 1, "mse": 0.6},
    ]
    mat = S.build_block_matrix(pd.DataFrame(rows))
    assert len(mat) == 2                      # 缺 revin 的块必须留着（none 那一列还有用）
    assert np.isnan(mat.loc["A|192|B", "revin"])
    # 只有显式要求完整块（Friedman/Nemenyi）时才丢
    mat_c = S.build_block_matrix(pd.DataFrame(rows), require_complete=True)
    assert len(mat_c) == 1 and not mat_c.isna().any().any()


def test_friedman_refuses_incomplete_matrix():
    """Friedman 是唯一真的要求同一块集合的检验；喂缺格矩阵必须报错而不是静默出 nan。"""
    m = pd.DataFrame({"none": [1.0, 1.1, 1.2, 1.3], "a": [0.9, 1.0, 1.1, 1.2],
                      "b": [1.0, np.nan, 1.3, 1.4]})
    with pytest.raises(ValueError, match="完整矩阵"):
        S.friedman_nemenyi(m)


# --------------------------------------------------------------------------- #
# k=2：Wilcoxon
# --------------------------------------------------------------------------- #
def _mat(n: int = 40, shift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.3, 0.8, size=n)
    return pd.DataFrame({"none": base, "revin": base - shift + rng.normal(0, 0.002, n)},
                        index=[f"b{i}" for i in range(n)])


def test_wilcoxon_detects_consistent_improvement():
    r = S.wilcoxon_pair(_mat(shift=0.02), "revin", "none")
    assert r["p_value"] < 1e-6
    assert r["win"] == 40 and r["loss"] == 0
    assert r["mean_rel_gain_pct"] > 0        # 正 = 相对对照降低误差
    assert 0.0 <= r["effect_r"] <= 1.0


def test_wilcoxon_no_effect_is_not_significant():
    r = S.wilcoxon_pair(_mat(shift=0.0, seed=7), "revin", "none")
    assert r["p_value"] > 0.05
    assert abs(r["mean_rel_gain_pct"]) < 1.0


def test_wilcoxon_sign_convention_for_harmful_plugin():
    r = S.wilcoxon_pair(_mat(shift=-0.03), "revin", "none")
    assert r["loss"] == 40 and r["win"] == 0
    assert r["mean_rel_gain_pct"] < 0


def test_wilcoxon_identical_columns_returns_p_one():
    m = _mat()
    m["revin"] = m["none"]
    r = S.wilcoxon_pair(m, "revin", "none")
    assert r["p_value"] == 1.0 and r["tie"] == len(m)


# --------------------------------------------------------------------------- #
# 2026-09-14 code review 修复项 #1：逐对可配对块（手算标定）
# --------------------------------------------------------------------------- #
def test_wilcoxon_pair_uses_only_pairwise_available_blocks():
    """手算标定：4 个可配对块 d = (-0.1, -0.2, -0.3, +0.4)，第 5 块 method 缺格。

    |d| 秩 = (1, 2, 3, 4)，W+ = 4、W- = 6 → scipy 的 statistic = min = 4；
    n=4 的精确双侧 p = 2·#{S ⊆ {1,2,3,4} : sum(S) ≤ 4}/2^4 = 2·7/16 = 0.875；
    r = |stat-μ|/σ/√n_eff，μ=5、σ=√7.5 → r = 0.18257。
    旧实现不做逐对 dropna，NaN 直接灌进 scipy → stat/p 变 nan，本例必挂。
    """
    m = pd.DataFrame({"none":  [1.0, 1.0, 1.0, 1.0, 1.0],
                      "revin": [0.9, 0.8, 0.7, 1.4, np.nan]},
                     index=[f"D{i}|96|DLinear" for i in range(5)])
    r = S.wilcoxon_pair(m, "revin", "none")
    assert r["stat"] == pytest.approx(4.0)                 # 旧实现在这里就是 nan
    assert r["p_value"] == pytest.approx(0.875, abs=1e-12)
    assert r["effect_r"] == pytest.approx(0.18257418583505536, abs=1e-9)
    assert r["n_blocks"] == 4 and r["n_eff"] == 4 and r["tie"] == 0
    assert r["win"] == 3 and r["loss"] == 1


def _pruned_results() -> pd.DataFrame:
    """复刻本仓库的 skip_rules 形态：revin 只在 DLinear 上有，fredf/none 全都有。

    fredf−none 的 8 个差值 = (+0.01, −0.02, −0.03, −0.08 | −0.04, −0.05, −0.06, −0.07)，
    |d| 互不相同 → 秩 = 1..8，唯一的正差值占秩 1。
    - 全部 8 个可配对块：stat=1，精确双侧 p = 2·#{S:sum≤1}/2^8 = 4/256 = 0.015625（显著）；
    - 只用 4 个「完整块」（= DLinear，旧口径）：stat=1，p = 2·2/2^4 = 0.25（不显著）。
    这就是修复项 #1 的核心：revin 的剪枝规则不该决定 fredf 的结论。
    """
    d = {("DLinear", "A"): +0.01, ("DLinear", "B"): -0.02,
         ("DLinear", "C"): -0.03, ("DLinear", "D"): -0.08,
         ("PatchTST", "A"): -0.04, ("PatchTST", "B"): -0.05,
         ("PatchTST", "C"): -0.06, ("PatchTST", "D"): -0.07}
    rows = []
    for (bk, ds), diff in d.items():
        common = {"dataset": ds, "pred_len": 96, "backbone": bk, "seed": 2021}
        rows.append({**common, "plugin": "none", "mse": 1.0})
        rows.append({**common, "plugin": "fredf", "mse": 1.0 + diff})
        if bk == "DLinear":                      # revin 在其余骨干上被 skip_rules 剪掉
            rows.append({**common, "plugin": "revin", "mse": 1.0 - 0.05})
    return pd.DataFrame(rows)


def test_pair_sample_is_not_decided_by_a_third_methods_pruning():
    mat = S.build_block_matrix(_pruned_results())
    assert mat.shape == (8, 3) and mat["revin"].isna().sum() == 4
    r = S.wilcoxon_pair(mat, "fredf", "none")
    assert r["n_blocks"] == 8 and r["n_eff"] == 8          # 旧口径只会有 4
    assert r["stat"] == pytest.approx(1.0)
    assert r["p_value"] == pytest.approx(0.015625, abs=1e-12)
    assert r["blocks_by_group"] == "DLinear:4;PatchTST:4"  # 骨干构成必须落盘
    # 旧口径（整表 dropna 后只剩 DLinear 4 块）会得到 0.25，结论从显著变不显著
    old = S.wilcoxon_pair(mat.dropna(axis=0, how="any"), "fredf", "none")
    assert old["n_blocks"] == 4 and old["p_value"] == pytest.approx(0.25, abs=1e-12)
    assert r["p_value"] < 0.05 <= old["p_value"]


def test_run_pipeline_separates_pairwise_and_complete_block_families(tmp_path):
    """流水线：k=2 用逐对块，Friedman/Nemenyi 才用完整块。"""
    s = S.run_pipeline(_pruned_results(), tmp_path)
    assert s["n_blocks"] == 8 and s["n_blocks_complete"] == 4
    wil = pd.read_csv(tmp_path / "wilcoxon_vs_control.csv").set_index("method")
    assert wil.loc["fredf", "n_blocks"] == 8               # 不被 revin 的缺格拖累
    assert wil.loc["revin", "n_blocks"] == 4               # revin 自己只有 4 块
    assert wil.loc["fredf", "p_value"] == pytest.approx(0.015625, abs=1e-12)
    assert s["friedman"]["n_blocks"] == 4                  # Friedman 只能用完整块
    wtl = pd.read_csv(tmp_path / "win_tie_loss.csv")
    all_rows = wtl[wtl.group == "ALL"].set_index("method")
    assert all_rows.loc["fredf", "n"] == 8 and all_rows.loc["revin", "n"] == 4
    assert (all_rows["win"] + all_rows["tie"] + all_rows["loss"] == all_rows["n"]).all()


# --------------------------------------------------------------------------- #
# 2026-09-14 code review 修复项 #2：效应量必须用去零后的 n_eff（手算标定）
# --------------------------------------------------------------------------- #
def test_effect_size_uses_nonzero_pair_count():
    """10 个配对块里 6 个完全打平，只有 4 个非零差值 (-0.1,-0.2,-0.3,+0.4)。

    检验只吃这 4 对：stat=4、精确 p=0.875、n_eff=4；
    μ=n_eff(n_eff+1)/4=5、σ=√(4·5·9/24)=√7.5 → r=|4-5|/√7.5/√4 = 0.18257。
    旧实现用含零差值的 n=10（μ=27.5、σ=√96.25）→ r=0.7575，被系统性放大 4 倍多。
    """
    diffs = [0.0] * 6 + [-0.1, -0.2, -0.3, 0.4]
    m = pd.DataFrame({"val_mse": [1.0] * 10,
                      "cand": [1.0 + x for x in diffs]})
    r = S.wilcoxon_pair(m, "cand", "val_mse")
    assert r["stat"] == pytest.approx(4.0)
    assert r["p_value"] == pytest.approx(0.875, abs=1e-12)
    assert r["effect_r"] == pytest.approx(0.18257418583505536, abs=1e-9)
    assert r["effect_r"] < 0.4                             # 旧实现给 0.7575
    assert r["n_blocks"] == 10 and r["tie"] == 6 and r["n_eff"] == 4
    assert r["n_eff"] + r["tie"] == r["n_blocks"]          # 三个口径必须自洽


def test_effect_size_is_invariant_to_adding_pure_ties():
    """加进任意多个「完全打平」的块，不改变检验，也不该改变效应量。

    这是修复项 #2 的判别性性质：旧实现里 r 会随打平块数单调上升（p 却一动不动）。
    """
    diffs = [-0.10, -0.21, -0.32, 0.43, -0.54, 0.65]
    base = pd.DataFrame({"none": [1.0] * 6, "a": [1.0 + d for d in diffs]})
    padded = pd.DataFrame({"none": [1.0] * 26, "a": [1.0 + d for d in diffs] + [1.0] * 20})
    r0, r1 = S.wilcoxon_pair(base, "a", "none"), S.wilcoxon_pair(padded, "a", "none")
    assert r1["stat"] == pytest.approx(r0["stat"])
    assert r1["p_value"] == pytest.approx(r0["p_value"])
    assert r1["effect_r"] == pytest.approx(r0["effect_r"], abs=1e-12)
    assert r1["n_blocks"] == 26 and r1["tie"] == 20 and r1["n_eff"] == r0["n_eff"] == 6


def test_effect_size_zero_when_everything_ties():
    m = pd.DataFrame({"none": [0.3, 0.4, 0.5], "a": [0.3, 0.4, 0.5]})
    r = S.wilcoxon_pair(m, "a", "none")
    assert r["n_eff"] == 0 and r["tie"] == 3 and r["p_value"] == 1.0
    assert r["effect_r"] == 0.0


# --------------------------------------------------------------------------- #
# 2026-09-14 code review 修复项 #3：分层检验的 Holm 族必须覆盖所有层
# --------------------------------------------------------------------------- #
def _strata_results() -> pd.DataFrame:
    """2 骨干 × 4 数据集 × 2 horizon = 每层 8 块、共 16 块，3 个插件 + 对照。"""
    rng = np.random.default_rng(11)
    rows = []
    for bk, gain in (("DLinear", {"revin": -0.05, "san_lite": -0.004, "fredf": -0.02}),
                     ("PatchTST", {"revin": -0.0002, "san_lite": 0.004, "fredf": -0.02})):
        for ds in ("A", "B", "C", "D"):
            for h in (96, 720):
                base = 0.4 + 0.05 * (h == 720) + rng.uniform(0, 0.02)
                rows.append({"dataset": ds, "pred_len": h, "backbone": bk,
                             "plugin": "none", "seed": 2021, "mse": base})
                for pl, g in gain.items():
                    rows.append({"dataset": ds, "pred_len": h, "backbone": bk, "plugin": pl,
                                 "seed": 2021, "mse": base + g + rng.normal(0, 0.0005)})
    return pd.DataFrame(rows)


def test_stratified_holm_family_spans_all_backbones(tmp_path):
    """12 个 backbone × method 比较必须有一列覆盖整个族的校正（旧实现只有 m=3 的层内校正）。"""
    S.run_pipeline(_strata_results(), tmp_path)
    df = pd.read_csv(tmp_path / "wilcoxon_by_backbone.csv")
    m_total = len(df)
    assert m_total == 2 * 3                                # 2 骨干 × 3 插件
    for col in ("p_holm_within_backbone", "reject_H0_within_backbone",
                "p_holm_across_strata", "reject_H0_across_strata",
                "holm_threshold_across_strata"):
        assert col in df.columns, col
    assert (df["holm_family_size_across_strata"] == m_total).all()
    # 层内族只有 3 个比较 → 最严阈值 α/3；跨层族有 m_total 个 → α/m_total
    assert df["holm_threshold_within_backbone"].min() == pytest.approx(0.05 / 3)
    assert df["holm_threshold_across_strata"].min() == pytest.approx(0.05 / m_total)
    # 手算 Holm：全族最小的原始 p 的校正值 = min(1, m_total · p_raw)
    top = df.sort_values("p_value").iloc[0]
    assert top["p_holm_across_strata"] == pytest.approx(min(1.0, m_total * top["p_value"]))
    # 跨层校正必然不松于层内校正
    assert (df["p_holm_across_strata"] >= df["p_holm_within_backbone"] - 1e-12).all()


def test_holm_correction_is_monotone_and_stepwise():
    df = S.holm({"a": 0.001, "b": 0.02, "c": 0.4})
    assert list(df["comparison"]) == ["a", "b", "c"]                 # 按 p 升序
    assert list(df["p_holm_adjusted"]) == sorted(df["p_holm_adjusted"])  # 校正后单调不减
    assert df.iloc[0]["reject_H0"] and not df.iloc[2]["reject_H0"]
    # Holm 的「一旦不拒绝就全部停止」性质
    stop = df["reject_H0"].tolist()
    assert stop == sorted(stop, reverse=True)


def test_holm_adjusted_p_never_below_raw():
    pv = {"a": 0.01, "b": 0.03, "c": 0.049}
    df = S.holm(pv).set_index("comparison")
    for name, p in pv.items():
        assert df.loc[name, "p_holm_adjusted"] >= p - 1e-12


def test_wilcoxon_vs_control_adds_holm_columns():
    df = S.wilcoxon_vs_control(_mat(shift=0.02).assign(fredf=lambda d: d["none"] - 0.01), "none")
    assert set(df["method"]) == {"revin", "fredf"}
    assert {"p_holm_adjusted", "reject_H0", "holm_threshold"} <= set(df.columns)


def test_win_tie_loss_table_with_grouping():
    m = _mat(n=20, shift=0.02)
    grp = pd.Series(["DLinear"] * 10 + ["PatchTST"] * 10, index=m.index)
    t = S.win_tie_loss_table(m, "none", group=grp)
    assert set(t["group"]) == {"ALL", "DLinear", "PatchTST"}
    assert t[t.group == "ALL"].iloc[0]["win"] == 20
    assert t["win_rate"].between(0, 1).all()


# --------------------------------------------------------------------------- #
# k>=3：Friedman + Nemenyi
# --------------------------------------------------------------------------- #
def test_friedman_rejects_when_one_method_dominates():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.3, 0.8, size=60)
    mat = pd.DataFrame({"none": base, "a": base - 0.05, "b": base - 0.02, "c": base + 0.01})
    fn = S.friedman_nemenyi(mat)
    assert fn["reject_H0"] and fn["p_value"] < 1e-6
    assert list(fn["avg_rank"].index)[0] == "a"      # 平均秩越小越好
    assert fn["nemenyi"].loc["a", "none"] < 0.05
    assert fn["cd"] == pytest.approx(S.nemenyi_cd(4, 60))


def test_friedman_requires_three_methods():
    with pytest.raises(ValueError, match="k>=3"):
        S.friedman_nemenyi(_mat())


def test_average_rank_gap_vs_cd_is_interpretable():
    """秩差 > CD 才可宣称显著；测试同时确认 CD 与秩差的量纲一致。"""
    rng = np.random.default_rng(2)
    base = rng.uniform(0.3, 0.8, size=128)
    mat = pd.DataFrame({"none": base, "a": base - 0.05, "b": base - 0.03, "c": base - 0.01})
    fn = S.friedman_nemenyi(mat)
    gap = fn["avg_rank"]["none"] - fn["avg_rank"]["a"]
    assert gap > fn["cd"]


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
def test_run_pipeline_end_to_end(tmp_path):
    df = S.demo_results()
    summary = S.run_pipeline(df, tmp_path)
    assert summary["n_blocks"] == 8 * 4 * 4          # 数据集 × horizon × 骨干
    assert summary["k_methods"] == 4
    assert summary["block_definition"] == "dataset × pred_len × backbone"
    for f in ["wilcoxon_vs_control.csv", "win_tie_loss.csv", "avg_ranks.csv",
              "nemenyi_pvalues.csv", "cd_diagram.png", "cd_threshold_table.csv",
              "wilcoxon_by_backbone.csv"]:
        assert (tmp_path / f).exists(), f
    assert (tmp_path / "cd_diagram.png").stat().st_size > 5000


def test_run_pipeline_recovers_injected_conditionality(tmp_path):
    """demo 数据里 revin 只在 DLinear 上有效 → 分层 Wilcoxon 必须还原这一条件性。"""
    S.run_pipeline(S.demo_results(), tmp_path)
    strat = pd.read_csv(tmp_path / "wilcoxon_by_backbone.csv")
    rev = strat[strat.method == "revin"].set_index("backbone")
    assert rev.loc["DLinear", "mean_rel_gain_pct"] > 3 * rev.loc["PatchTST", "mean_rel_gain_pct"]
    assert rev.loc["DLinear", "p_value"] < 0.05


# --------------------------------------------------------------------------- #
# 2026-09-14 code review 修复项 #4：`src/horizon.py` 同时报出的多个 Wilcoxon
# 必须走 stats.holm 校正，且 oracle_test（上界）不进假设族。
#
# 这几条本该放 tests/test_horizon.py，但本次 code review 的可改文件清单里没有它
# （有别的 agent 在并行改），因此放在这里 —— 被测的口径本身就是 stats.py 的规范。
# --------------------------------------------------------------------------- #
def _hz_cell(cell_id: str, val_segs: list[list[float]], test: list[float]) -> pd.DataFrame:
    """按 (epoch × 段) 造一个 cell 的训练日志（同 tests/test_horizon.make_cell）。"""
    rows = []
    for e, segs in enumerate(val_segs):
        rec = {"cell_id": cell_id, "epoch": e, "val_mse": float(np.mean(segs)),
               "test_mse": float(test[e])}
        rec.update({f"val_seg{i+1}_mse": float(v) for i, v in enumerate(segs)})
        rec.update({f"test_seg{i+1}_mse": float(v * 1.02) for i, v in enumerate(segs)})
        rows.append(rec)
    return pd.DataFrame(rows)


def test_horizon_hypothesis_family_excludes_baseline_and_oracle():
    from src import horizon as HZ

    tested = HZ.hypothesis_criteria("val_mse")
    assert "val_mse" not in tested and "oracle_test" not in tested
    assert set(tested) == {"last", "rank_agg", "long_seg", "worst_seg", "slope_penalized"}
    assert len(tested) == 5                       # 族规模 5，不是 6（oracle 是上界不是假设）


def test_horizon_criteria_wilcoxon_is_holm_corrected(tmp_path):
    """手算标定：8 个 cell，只有 `last` 与基线不同，其余 4 个准则选到同一个 epoch。

    验证曲线 1.0 → 0.5 → 0.6（两段等值 → 无退化斜率），所以 val_mse / rank_agg /
    long_seg / worst_seg / slope_penalized 全选 epoch1；`last` 被迫选 epoch2，测试 MSE
    比 epoch1 高 0.01·(i+1)（8 个差值互不相同且全为正）。
    → `last`：8 对同号差值，W=0，精确双侧 p = 2/2^8 = 0.0078125；
      Holm 第一步阈值 α/5 = 0.01，校正 p = 5 × 0.0078125 = 0.0390625 → 拒绝；
    → 其余 4 个准则与基线完全打平（n_eff=0）→ p=1，校正后仍是 1。
    旧实现只写原始 p_value（族错误率 ~0.26），本例必挂。
    """
    from src import horizon as HZ

    logs = pd.concat([
        _hz_cell(f"c{i}", [[1.0, 1.0], [0.5, 0.5], [0.6, 0.6]],
                 test=[1.5, 0.5, 0.5 + 0.01 * (i + 1)])
        for i in range(8)
    ])
    out = HZ.run(logs, tmp_path)
    df = pd.read_csv(tmp_path / "criteria_wilcoxon.csv")
    assert "oracle_test" not in set(df["method"])          # 上界不参与检验
    assert len(df) == 5 and (df["holm_family_size"] == 5).all()
    for col in ("p_holm_adjusted", "reject_H0", "holm_threshold"):
        assert col in df.columns, col
    d = df.set_index("method")
    assert d.loc["last", "n_blocks"] == 8 and d.loc["last", "n_eff"] == 8
    assert d.loc["last", "p_value"] == pytest.approx(0.0078125, abs=1e-12)
    assert d.loc["last", "holm_threshold"] == pytest.approx(0.05 / 5)
    assert d.loc["last", "p_holm_adjusted"] == pytest.approx(0.0390625, abs=1e-12)
    assert bool(d.loc["last", "reject_H0"]) is True
    for name in ("rank_agg", "long_seg", "worst_seg", "slope_penalized"):
        assert d.loc[name, "n_eff"] == 0 and d.loc[name, "tie"] == 8
        assert d.loc[name, "p_value"] == pytest.approx(1.0)
        assert d.loc[name, "p_holm_adjusted"] == pytest.approx(1.0)
        assert not bool(d.loc[name, "reject_H0"])
    assert {r["method"] for r in out["wilcoxon"]} == set(HZ.hypothesis_criteria())
    # oracle 仍留在描述性对比表里（报「距上界的差距」），只是不做假设检验
    assert "oracle_test" in set(pd.read_csv(tmp_path / "criteria_comparison.csv")["criterion"])


def test_horizon_holm_never_loosens_raw_p(tmp_path):
    """判别性：族规模 5 时校正 p 必须严格大于原始 p（除非已封顶到 1），且单调不减。"""
    from src import horizon as HZ

    rng = np.random.default_rng(3)
    cells = []
    for c in range(14):
        segs, test = [], []
        for e in range(5):
            short = 0.3 + 0.02 * (e - 1) ** 2 + rng.normal(0, 0.002)
            long = 0.6 + 0.02 * (e - 3) ** 2 + rng.normal(0, 0.002)
            segs.append([short, long])
            test.append(0.5 * (short + long) * (1 + rng.normal(0, 0.01)))
        cells.append(_hz_cell(f"c{c}", segs, test=test))
    HZ.run(pd.concat(cells), tmp_path)
    df = pd.read_csv(tmp_path / "criteria_wilcoxon.csv")
    assert (df["p_holm_adjusted"] >= df["p_value"] - 1e-12).all()
    strict = df[df["p_holm_adjusted"] < 1.0]
    assert len(strict) >= 1 and (strict["p_holm_adjusted"] > strict["p_value"]).all()
    s = df.sort_values("p_value")["p_holm_adjusted"].tolist()
    assert s == sorted(s)
