"""``src/robustness.py`` 的单元测试。

测试原则：每个断言都对应一个「如果实现写错、论文里会出现什么错数字」的具体风险，
而不是只测「函数能跑通」。合成数据全部构造成解析可算的，期望值写死在断言里。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.robustness import (
    SEG_COLS,
    TAIL_THRESHOLDS,
    cost_benefit,
    demo_results_rich,
    leadtime_profile,
    metric_robustness,
    paired_bootstrap_ci,
    run_all,
    seed_stability,
    tail_stats,
)

BLOCKS = [("ETTh1", 96, "DLinear"), ("ETTh1", 192, "DLinear"),
          ("ETTh2", 96, "PatchTST"), ("ETTh2", 192, "PatchTST"),
          ("Weather", 96, "TimesNet"), ("Weather", 192, "TimesNet")]


def _frame(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def _simple(arm_mse: dict[str, float], arm_mae: dict[str, float] | None = None,
            seeds: tuple[int, ...] = (2021,), jitter: float = 0.0,
            segs: dict[str, list[float]] | None = None,
            train_seconds: dict[str, float] | None = None) -> pd.DataFrame:
    """所有块上给同一组臂同一个误差值（可加确定性抖动），便于反推期望值。"""
    rows = []
    for bi, (ds, h, bk) in enumerate(BLOCKS):
        for arm, v in arm_mse.items():
            for si, s in enumerate(seeds):
                r = {"dataset": ds, "pred_len": h, "backbone": bk, "plugin": arm,
                     "seed": s, "mse": v + jitter * si,
                     "mae": (arm_mae or arm_mse)[arm] + jitter * si}
                if segs is not None:
                    for i, sv in enumerate(segs[arm]):
                        r[f"seg{i + 1}_mse"] = sv
                if train_seconds is not None:
                    r["train_seconds"] = train_seconds[arm]
                    r["extra_params"] = 0.0
                    r["peak_mem_mib"] = 100.0
                rows.append(r)
    return _frame(rows)


# --------------------------------------------------------------------------- #
# paired_bootstrap_ci
# --------------------------------------------------------------------------- #
class TestPairedBootstrapCI:
    def test_constant_gain_gives_degenerate_ci(self):
        """所有块增益完全相同时，重抽样不可能产生变异，CI 必须退化到点估计。

        风险：若实现对增益值独立重抽样而不是按块，这里会出现虚假宽度。
        """
        control = np.full(20, 1.0)
        treat = np.full(20, 0.9)          # 相对增益恒为 10%
        r = paired_bootstrap_ci(treat, control, n_boot=500, seed=1)
        assert r["mean_rel_gain_pct"] == pytest.approx(10.0)
        assert r["mean_ci_lo"] == pytest.approx(10.0)
        assert r["mean_ci_hi"] == pytest.approx(10.0)
        assert r["median_ci_lo"] == pytest.approx(10.0)

    def test_ci_brackets_point_estimate_and_is_ordered(self):
        rng = np.random.default_rng(0)
        control = rng.uniform(0.5, 1.5, 60)
        treat = control * rng.normal(0.95, 0.05, 60)
        r = paired_bootstrap_ci(treat, control, n_boot=2000, seed=7)
        assert r["mean_ci_lo"] <= r["mean_rel_gain_pct"] <= r["mean_ci_hi"]
        assert r["median_ci_lo"] <= r["median_rel_gain_pct"] <= r["median_ci_hi"]

    def test_relative_gain_definition_matches_wilcoxon_pair(self):
        """相对增益必须是 (control-treat)/|control|，与 stats.wilcoxon_pair 同口径。

        风险：如果这里改成 /|treat| 或不取绝对值，论文同一个数字会有两个版本。
        """
        control = np.array([2.0, 4.0])
        treat = np.array([1.0, 2.0])
        r = paired_bootstrap_ci(treat, control, n_boot=100, seed=3)
        assert r["mean_rel_gain_pct"] == pytest.approx(50.0)

    def test_negative_control_uses_absolute_denominator(self):
        control = np.array([-2.0, -2.0])
        treat = np.array([-3.0, -3.0])       # 误差更负 → 相对「增益」= (-2 - -3)/2 = +50%
        r = paired_bootstrap_ci(treat, control, n_boot=100, seed=3)
        assert r["mean_rel_gain_pct"] == pytest.approx(50.0)

    def test_empty_and_single_block(self):
        r0 = paired_bootstrap_ci(np.array([]), np.array([]))
        assert r0["n_blocks"] == 0 and np.isnan(r0["mean_rel_gain_pct"])
        r1 = paired_bootstrap_ci(np.array([0.9]), np.array([1.0]), n_boot=10)
        assert r1["mean_rel_gain_pct"] == pytest.approx(10.0)
        assert np.isnan(r1["mean_ci_lo"])       # n=1 不做自举

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="形状不一致"):
            paired_bootstrap_ci(np.zeros(3), np.zeros(4))

    def test_deterministic_given_seed(self):
        rng = np.random.default_rng(5)
        c = rng.uniform(1, 2, 30)
        t = c * 0.97
        a = paired_bootstrap_ci(t, c, n_boot=300, seed=42)
        b = paired_bootstrap_ci(t, c, n_boot=300, seed=42)
        assert a == b


# --------------------------------------------------------------------------- #
# metric_robustness
# --------------------------------------------------------------------------- #
class TestMetricRobustness:
    def test_detects_verdict_flip_between_metrics(self):
        """构造「MSE 上无差异、MAE 上一致占优」的臂，必须报出 verdict_flips=True。

        这正是真实数据里 revin 的形态，是论文的一个核心新结论，不能漏检。
        """
        rows = []
        for bi, (ds, h, bk) in enumerate(BLOCKS * 4):        # 24 个块，功效足够
            rows.append({"dataset": ds, "pred_len": h + bi, "backbone": bk,
                         "plugin": "none", "seed": 1, "mse": 1.0, "mae": 1.0})
            rows.append({"dataset": ds, "pred_len": h + bi, "backbone": bk,
                         "plugin": "arm", "seed": 1,
                         "mse": 1.0 + (0.01 if bi % 2 else -0.01),   # 正负交替 → 不显著
                         "mae": 0.9})                                # 一致更优 → 显著
        per_metric, agree = metric_robustness(_frame(rows), ("mse", "mae"), "none",
                                              n_boot=500)
        mse_row = per_metric[(per_metric.metric == "mse") & (per_metric.method == "arm")].iloc[0]
        mae_row = per_metric[(per_metric.metric == "mae") & (per_metric.method == "arm")].iloc[0]
        assert not bool(mse_row.reject_H0)
        assert bool(mae_row.reject_H0)
        assert bool(agree.iloc[0].verdict_flips)

    def test_pairwise_dropna_not_complete_case(self):
        """某臂只在部分块存在时，其 n_blocks 必须是「该对可配对的块数」，
        而不能被别的臂的缺格拖累（这是 code review 修复项 #1 的口径）。"""
        rows = []
        for bi, (ds, h, bk) in enumerate(BLOCKS):
            rows.append({"dataset": ds, "pred_len": h, "backbone": bk,
                         "plugin": "none", "seed": 1, "mse": 1.0, "mae": 1.0})
            rows.append({"dataset": ds, "pred_len": h, "backbone": bk,
                         "plugin": "full", "seed": 1, "mse": 0.9, "mae": 0.9})
            if bi < 2:                                  # sparse 只在 2 个块出现
                rows.append({"dataset": ds, "pred_len": h, "backbone": bk,
                             "plugin": "sparse", "seed": 1, "mse": 0.8, "mae": 0.8})
        per_metric, _ = metric_robustness(_frame(rows), ("mse",), "none", n_boot=200)
        n = per_metric.set_index("method").n_blocks
        assert int(n["full"]) == len(BLOCKS)
        assert int(n["sparse"]) == 2

    def test_holm_family_is_per_metric(self):
        """Holm 族规模 = 该指标内的方法数；不跨指标合并（否则功效被无谓稀释）。"""
        df = _simple({"none": 1.0, "a": 0.9, "b": 0.95})
        per_metric, _ = metric_robustness(df, ("mse", "mae"), "none", n_boot=200)
        assert set(per_metric.holm_family_size) == {2}

    def test_missing_metric_column_raises(self):
        df = _simple({"none": 1.0, "a": 0.9})
        with pytest.raises(KeyError, match="缺少指标列"):
            metric_robustness(df, ("mse", "rmse"), "none", n_boot=100)

    def test_missing_control_raises(self):
        df = _simple({"a": 1.0, "b": 0.9})
        with pytest.raises(KeyError, match="没有对照臂"):
            metric_robustness(df, ("mse",), "none", n_boot=100)

    def test_sign_agreement_is_fraction(self):
        df = _simple({"none": 1.0, "a": 0.9}, {"none": 1.0, "a": 0.9})
        _, agree = metric_robustness(df, ("mse", "mae"), "none", n_boot=100)
        assert agree.iloc[0].sign_agreement == pytest.approx(1.0)

    def test_mean_median_sign_conflict_is_flagged(self):
        """单个块的巨幅回退把均值拉到负、中位数仍为正时，必须打出冲突标记。

        这是真实数据里 fredf 的形态（均值 -2.37%、中位数 +0.45%），论文
        \\S Result 1 用它论证「估计量的选择也是显著性杠杆」，漏检就没有这个结论。
        """
        rows = []
        for bi in range(12):
            rows.append({"dataset": "D", "pred_len": 96 + bi, "backbone": "B",
                         "plugin": "none", "seed": 1, "mse": 1.0, "mae": 1.0})
            # 11 个块小幅变好，1 个块灾难性变差
            mse = 0.98 if bi < 11 else 3.0
            rows.append({"dataset": "D", "pred_len": 96 + bi, "backbone": "B",
                         "plugin": "arm", "seed": 1, "mse": mse, "mae": mse})
        per_metric, _ = metric_robustness(_frame(rows), ("mse",), "none", n_boot=300)
        row = per_metric.iloc[0]
        assert row.median_rel_gain_pct > 0          # 中位数说「有效」
        assert row.mean_rel_gain_pct < 0            # 均值说「有害」
        assert bool(row.mean_median_sign_conflict)
        assert row.frac_worse_than_25pct == pytest.approx(1 / 12)
        assert row.worst_block is not None

    def test_tail_columns_present_for_every_row(self):
        """尾部列必须对每个 (metric, method) 都有，论文表格直接按列渲染。

        ``skew`` 在增益完全恒定时按设计返回 NaN（方差为零，偏度无定义），
        所以这里用带块间差异的数据，确保正常路径下每一列都算得出来。
        """
        df = _simple({"none": 1.0, "a": 0.9, "b": 1.1}, jitter=0.0)
        # 让每个块的增益不同，否则 skew 无定义
        df.loc[df.plugin == "a", "mse"] += np.linspace(0, 0.05, (df.plugin == "a").sum())
        df.loc[df.plugin == "a", "mae"] += np.linspace(0, 0.05, (df.plugin == "a").sum())
        per_metric, _ = metric_robustness(df, ("mse", "mae"), "none", n_boot=100)
        for col in ("skew", "mean_minus_median", "frac_worse_than_10pct",
                    "frac_worse_than_25pct", "worst_block", "worst_rel_gain_pct",
                    "mean_median_sign_conflict"):
            assert col in per_metric.columns, col
        arm_a = per_metric[per_metric.method == "a"]
        assert arm_a["skew"].notna().all()
        assert per_metric[["mean_minus_median", "frac_worse_than_10pct",
                           "worst_rel_gain_pct"]].notna().all().all()
        assert per_metric["worst_block"].notna().all()


# --------------------------------------------------------------------------- #
# tail_stats
# --------------------------------------------------------------------------- #
class TestTailStats:
    def test_thresholds_are_strict_and_counted_as_fractions(self):
        """尾部占比必须是「严格跌破阈值」的**比例**，边界值不计入。

        取 -10 恰好等于阈值：如果实现写成 ``<=``，论文里的尾部风险会被系统性夸大。
        """
        rel = pd.Series([+5.0, 0.0, -10.0, -10.1, -30.0], index=list("abcde"))
        t = tail_stats(rel)
        assert t["frac_worse_than_10pct"] == pytest.approx(2 / 5)   # -10.1, -30
        assert t["frac_worse_than_25pct"] == pytest.approx(1 / 5)   # -30

    def test_reports_worst_and_best_block_labels(self):
        """必须点名最坏/最好的块，论文正文要直接引用这个块名。"""
        rel = pd.Series([+3.0, -50.0, +9.0], index=["Weather|96|DLinear",
                                                    "Electricity|720|PatchTST",
                                                    "ETTh2|336|DLinear"])
        t = tail_stats(rel)
        assert t["worst_block"] == "Electricity|720|PatchTST"
        assert t["worst_rel_gain_pct"] == pytest.approx(-50.0)
        assert t["best_block"] == "ETTh2|336|DLinear"
        assert t["best_rel_gain_pct"] == pytest.approx(+9.0)

    def test_mean_minus_median_captures_left_tail(self):
        """左偏分布下 mean < median，差值必须为负——这是「估计量杠杆」的方向。"""
        rel = pd.Series([1.0, 1.0, 1.0, 1.0, -100.0])
        t = tail_stats(rel)
        assert t["mean_minus_median"] < 0
        assert t["skew"] < 0

    def test_empty_input_is_safe(self):
        """空输入不能抛异常（某个臂在某指标上无可配对块时会走到这里）。"""
        t = tail_stats(pd.Series(dtype=float))
        assert t["worst_block"] is None
        assert np.isnan(t["mean_minus_median"])

    def test_custom_thresholds_are_honoured(self):
        t = tail_stats(pd.Series([-1.0, -3.0, -7.0]), thresholds=(-2.0, -5.0))
        assert t["frac_worse_than_2pct"] == pytest.approx(2 / 3)
        assert t["frac_worse_than_5pct"] == pytest.approx(1 / 3)

    def test_default_thresholds_constant(self):
        assert TAIL_THRESHOLDS == (-10.0, -25.0)


# --------------------------------------------------------------------------- #
# seed_stability
# --------------------------------------------------------------------------- #
class TestSeedStability:
    def test_single_seed_yields_no_noise_estimate(self):
        """单种子时噪声不可估，必须如实报 0 而不是拿 0 当噪声（那会让所有增益都'超过噪声'）。"""
        df = _simple({"none": 1.0, "a": 0.9}, seeds=(2021,))
        detail, summary = seed_stability(df)
        assert summary["n_blocks_with_noise"] == 0
        assert detail.noise_scale.isna().all()

    def test_noise_scale_is_pooled_sd(self):
        """noise_scale 必须等于 sqrt(sd_arm^2 + sd_ctrl^2)。"""
        rows = []
        for ds, h, bk in BLOCKS[:1]:
            for arm, vals in (("none", [1.0, 1.2]), ("a", [0.9, 1.1])):
                for s, v in zip((1, 2), vals):
                    rows.append({"dataset": ds, "pred_len": h, "backbone": bk,
                                 "plugin": arm, "seed": s, "mse": v, "mae": v})
        detail, _ = seed_stability(_frame(rows))
        sd = np.std([1.0, 1.2], ddof=1)          # 两臂 sd 相同
        assert detail.iloc[0].noise_scale == pytest.approx(np.sqrt(2 * sd**2))

    def test_within_noise_flag_thresholds(self):
        """|gain| 小于噪声 → within_noise=True；远大于噪声 → False。"""
        def build(gain: float) -> pd.DataFrame:
            rows = []
            for ds, h, bk in BLOCKS[:1]:
                for s, d in zip((1, 2), (-0.05, +0.05)):     # sd 可估
                    rows.append({"dataset": ds, "pred_len": h, "backbone": bk,
                                 "plugin": "none", "seed": s, "mse": 1.0 + d, "mae": 1.0})
                    rows.append({"dataset": ds, "pred_len": h, "backbone": bk,
                                 "plugin": "a", "seed": s, "mse": 1.0 - gain + d, "mae": 1.0})
            return _frame(rows)

        small, _ = seed_stability(build(0.001))
        big, _ = seed_stability(build(5.0))
        assert bool(small.iloc[0].within_noise) is True
        assert bool(big.iloc[0].within_noise) is False

    def test_summary_reports_win_subset(self):
        df = _simple({"none": 1.0, "a": 0.9}, seeds=(1, 2, 3), jitter=0.01)
        _, summary = seed_stability(df)
        assert summary["n_wins"] >= 1
        assert 0.0 <= summary["frac_within_noise"] <= 1.0

    def test_missing_column_raises(self):
        df = _simple({"none": 1.0, "a": 0.9}).drop(columns=["seed"])
        with pytest.raises(KeyError, match="缺列"):
            seed_stability(df)


# --------------------------------------------------------------------------- #
# leadtime_profile
# --------------------------------------------------------------------------- #
class TestLeadtimeProfile:
    def test_constant_argmin_detected(self):
        """同一个臂在四个 lead-time 四分位上都最优 → constant_argmin 全为 True。"""
        df = _simple({"none": 1.0, "a": 0.5},
                     segs={"none": [1.0, 1.0, 1.0, 1.0], "a": [0.5, 0.5, 0.5, 0.5]})
        _, stability, summary = leadtime_profile(df)
        assert bool(stability.constant_argmin.all())
        assert summary["frac_constant_argmin"] == pytest.approx(1.0)
        assert summary["mean_n_distinct_argmin"] == pytest.approx(1.0)

    def test_switching_argmin_detected(self):
        """近端 a 更优、远端 b 更优 → 必须报出 2 个不同 argmin 臂。

        这是审计 3 的核心信号；若 pivot 或 idxmin 用错轴，这里会退化成 1。
        """
        df = _simple({"none": 1.0, "a": 0.9, "b": 0.9},
                     segs={"none": [1.0, 1.0, 1.0, 1.0],
                           "a": [0.5, 0.6, 1.4, 1.5],       # 近端好
                           "b": [1.5, 1.4, 0.6, 0.5]})      # 远端好
        _, stability, summary = leadtime_profile(df)
        assert not bool(stability.constant_argmin.any())
        assert int(stability.n_distinct_argmin.iloc[0]) == 2
        assert summary["frac_constant_argmin"] == pytest.approx(0.0)
        assert stability.iloc[0].argmin_by_quarter.startswith("q1:a")
        assert stability.iloc[0].argmin_by_quarter.endswith("q4:b")

    def test_whole_horizon_best_can_be_suboptimal_per_quarter(self):
        """整格最优臂在某些四分位上不是最优 → 计数必须 > 0。"""
        df = _simple({"none": 1.0, "a": 0.9, "b": 0.95},
                     segs={"none": [1.0, 1.0, 1.0, 1.0],
                           "a": [1.3, 1.3, 0.5, 0.5],       # 整格均值最低
                           "b": [0.6, 0.6, 1.3, 1.3]})
        _, stability, summary = leadtime_profile(df)
        row = stability.iloc[0]
        assert row.whole_horizon_best == "a"
        assert int(row.n_quarters_where_whole_best_suboptimal) == 2
        assert summary["frac_blocks_whole_best_suboptimal_somewhere"] == pytest.approx(1.0)

    def test_leadtime_oracle_is_zero_when_one_arm_dominates_everywhere(self):
        """一个臂在四个四分位全面最优时，按 lead-time 换臂的 headroom 必须恰好为 0。

        这是论文「合法轴上也没东西可拿」这条结论的下界检查：若实现把
        per-lead-time oracle 算成了「逐四分位取最小再和 control 比」而忘了
        和**最优固定臂**比，这里会错误地报出正的 headroom。
        """
        df = _simple({"none": 1.0, "a": 0.5},
                     segs={"none": [1.0, 1.0, 1.0, 1.0], "a": [0.5, 0.5, 0.5, 0.5]})
        _, stability, summary = leadtime_profile(df)
        assert stability.leadtime_oracle_gain_vs_best_fixed_pct.abs().max() < 1e-9
        assert summary["leadtime_oracle_vs_best_fixed_mean_pct"] == pytest.approx(0.0)
        # 相对 control 仍应有 50% 的增益（那是选臂本身的收益，不是换臂的收益）
        assert summary["leadtime_oracle_vs_control_mean_pct"] == pytest.approx(50.0)

    def test_leadtime_oracle_positive_only_when_arms_cross(self):
        """两臂在 lead-time 上交叉时，headroom 必须为正且等于解析值。

        构造：a=[0.4,0.4,1.0,1.0]，b=[1.0,1.0,0.4,0.4]。
        两臂整格 MSE 都是 0.7，最优固定臂 = 0.7；
        逐四分位取最小 = mean(0.4,0.4,0.4,0.4) = 0.4。
        headroom = (0.7-0.4)/0.7 = 42.857%。
        """
        df = _simple({"none": 2.0, "a": 0.7, "b": 0.7},
                     segs={"none": [2.0, 2.0, 2.0, 2.0],
                           "a": [0.4, 0.4, 1.0, 1.0],
                           "b": [1.0, 1.0, 0.4, 0.4]})
        _, stability, summary = leadtime_profile(df)
        assert stability.iloc[0].per_leadtime_oracle_mse == pytest.approx(0.4)
        assert stability.iloc[0].best_fixed_arm_mse == pytest.approx(0.7)
        assert (stability.iloc[0].leadtime_oracle_gain_vs_best_fixed_pct
                == pytest.approx(100 * 0.3 / 0.7))
        assert summary["n_blocks_leadtime_oracle_beats_best_fixed"] == len(BLOCKS)
        assert summary["frac_blocks_leadtime_gain_over_0p5pp"] == pytest.approx(1.0)

    def test_block_mse_equals_mean_of_quarters(self):
        """四分位均值必须等于整格 MSE，否则 per-lead-time oracle 的口径就不成立。

        真实数据上这一恒等式成立到 0 误差（``array_split`` 等分 pred_len），
        这里把它固化成断言，防止将来有人改了 seg 的定义而无人察觉。
        """
        df = _simple({"none": 2.5, "a": 1.0},
                     segs={"none": [1.0, 2.0, 3.0, 4.0],      # 均值 2.5 = mse
                           "a": [0.4, 0.8, 1.2, 1.6]})        # 均值 1.0 = mse
        _, stability, _ = leadtime_profile(df)
        row = stability.iloc[0]
        assert row.control_mse == pytest.approx(2.5)
        assert row.best_fixed_arm_mse == pytest.approx(1.0)

    def test_quarter_index_parsed_from_column_name(self):
        df = _simple({"none": 1.0, "a": 0.9},
                     segs={"none": [1.0, 2.0, 3.0, 4.0], "a": [0.9, 1.9, 2.9, 3.9]})
        profile, _, _ = leadtime_profile(df)
        assert sorted(profile.quarter.unique().tolist()) == [1, 2, 3, 4]
        q4 = profile[(profile.method == "none") & (profile.quarter == 4)]
        assert q4.seg_mse.iloc[0] == pytest.approx(4.0)

    def test_gain_sign_convention(self):
        """rel_gain_pct 正 = 比 control 误差更低，与其他审计一致。"""
        df = _simple({"none": 1.0, "a": 0.9},
                     segs={"none": [1.0, 1.0, 1.0, 1.0], "a": [0.5, 0.5, 0.5, 0.5]})
        profile, _, _ = leadtime_profile(df)
        a = profile[(profile.method == "a") & (profile.quarter == 1)].iloc[0]
        assert a.rel_gain_pct == pytest.approx(50.0)

    def test_missing_seg_columns_raises(self):
        df = _simple({"none": 1.0, "a": 0.9})
        with pytest.raises(KeyError, match="lead-time 分段列"):
            leadtime_profile(df)

    def test_seg_cols_constant_matches_runner_output(self):
        assert SEG_COLS == ("seg1_mse", "seg2_mse", "seg3_mse", "seg4_mse")


# --------------------------------------------------------------------------- #
# cost_benefit
# --------------------------------------------------------------------------- #
class TestCostBenefit:
    def test_train_time_ratio_and_gain(self):
        df = _simple({"none": 1.0, "slow": 0.99},
                     train_seconds={"none": 100.0, "slow": 200.0})
        detail, per_method = cost_benefit(df)
        assert detail.iloc[0].train_time_ratio == pytest.approx(2.0)
        row = per_method.set_index("method").loc["slow"]
        assert row.median_train_time_ratio == pytest.approx(2.0)
        assert row.median_rel_gain_pct == pytest.approx(1.0)
        # 多花 100% 时间换 1% 增益 → 0.01 个百分点/1% 额外时间
        assert row.gain_per_pct_extra_time == pytest.approx(0.01)

    def test_no_extra_time_gives_nan_efficiency(self):
        """训练时间没变（比值 ≤ 1）时「每 1% 额外时间的收益」无定义，必须是 NaN 而不是 inf。"""
        df = _simple({"none": 1.0, "same": 0.9},
                     train_seconds={"none": 100.0, "same": 100.0})
        _, per_method = cost_benefit(df)
        assert np.isnan(per_method.set_index("method").loc["same"].gain_per_pct_extra_time)

    def test_works_without_cost_columns(self):
        df = _simple({"none": 1.0, "a": 0.9})
        detail, per_method = cost_benefit(df)
        assert "train_time_ratio" not in detail.columns
        assert len(per_method) == 1


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
class TestRunAll:
    def test_writes_all_artefacts_and_json_parses(self, tmp_path):
        df = demo_results_rich(seed=3, n_seeds=2)
        summary = run_all(df, tmp_path, n_boot=200)
        expected = ["metric_robustness.csv", "metric_agreement.csv", "seed_stability.csv",
                    "leadtime_profile.csv", "leadtime_argmin_stability.csv",
                    "cost_benefit.csv", "cost_benefit_by_method.csv",
                    "robustness_summary.json"]
        for name in expected:
            assert (tmp_path / name).exists(), f"缺产物 {name}"
        loaded = json.loads((tmp_path / "robustness_summary.json").read_text())
        assert loaded["control"] == "none"
        for key in ("metric_robustness", "seed_stability", "leadtime", "cost_benefit"):
            assert key in summary and summary[key], f"{key} 为空，审计没真正跑到"

    def test_demo_rich_exercises_all_four_audits(self):
        """--demo 必须能触发全部四项审计；否则自检形同虚设。"""
        df = demo_results_rich(seed=4, n_seeds=3)
        assert {"seed", "train_seconds", *SEG_COLS} <= set(df.columns)
        _, seed_sum = seed_stability(df)
        assert seed_sum["n_blocks_with_noise"] > 0
        _, _, lt = leadtime_profile(df)
        assert lt["n_blocks_multi_arm"] > 0
        assert lt["frac_constant_argmin"] < 1.0        # 构造上就应该有切换

    def test_leadtime_skipped_gracefully_without_seg_cols(self, tmp_path):
        """真实项目里可能拿到没有 seg 列的旧结果表，整条链不能因此崩。"""
        df = demo_results_rich(seed=5, n_seeds=2).drop(columns=list(SEG_COLS))
        summary = run_all(df, tmp_path, n_boot=100)
        assert "skipped" in summary["leadtime"]
