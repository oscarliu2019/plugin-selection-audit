"""周末报告脚本的测试：它是周一唯一会被人读的产物，绝不能因为数据残缺而崩。

重点覆盖三类风险：
1. α 名字解析错 -> 整个 FreDF 结论错；
2. 缺 none / 缺 p3 / 缺门控 -> 报告应降级而不是抛异常；
3. 多 seed 参考混进来 -> 「同格重复标准差」自检误报。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.weekend_report import (  # noqa: E402
    _alpha_of,
    _next_steps,
    _pct_or_dash,
    _render_markdown,
    fredf_analysis,
    fredf_verdict,
)


# --------------------------------------------------------------------------- #
def test_alpha_parsing():
    assert _alpha_of("fredf_a01") == ("plain", 0.1)
    assert _alpha_of("fredf_a09") == ("plain", 0.9)
    assert _alpha_of("fredf_sqrth_a05") == ("sqrth", 0.5)
    # phase-1 里那个没后缀的臂就是 plain α=0.5
    assert _alpha_of("fredf") == ("plain", 0.5)
    for other in ("none", "revin", "san_lite"):
        assert _alpha_of(other) is None


def _synth(alpha_gain: float = 0.06, sqrth_flat: bool = True, seeds=(2021,)):
    """构造已知答案的合成结果：增益随 α 线性上升、被一个 horizon 相关的惩罚项压回来。

    ``sqrth_flat=True`` 时惩罚项与 H 无关，即模拟「√H 归一化奏效」。
    """
    p3arms = ["none", "fredf_a01", "fredf_a03", "fredf_a07", "fredf_a09",
              "fredf_sqrth_a01", "fredf_sqrth_a05", "fredf_sqrth_a09"]
    rng = np.random.default_rng(0)
    p3, ref = [], []
    for bb in ("DLinear", "PatchTST", "iTransformer"):
        for ds in ("ETTh1", "Weather", "Exchange", "ETTm1"):
            for H in (96, 192, 336, 720):
                base = 0.4 + 0.1 * rng.random()

                def mse(kind: str, a: float, H=H, base=base) -> float:
                    pen = 0.02 if (kind == "sqrth" and sqrth_flat) else 0.02 * np.sqrt(H / 96)
                    return base * (1 - alpha_gain * a + pen * a * a)

                for arm in p3arms:
                    parsed = _alpha_of(arm)
                    m = base if parsed is None else mse(*parsed)
                    p3.append(dict(backbone=bb, dataset=ds, pred_len=H, plugin=arm,
                                   seed=2021, mse=m, mae=m * 0.7, val_mse=m * 1.01))
                for sd in seeds:
                    jitter = 0.0 if sd == 2021 else 0.01
                    for arm, m in (("fredf", mse("plain", 0.5)), ("none", base)):
                        ref.append(dict(backbone=bb, dataset=ds, pred_len=H, plugin=arm,
                                        seed=sd, mse=m + jitter, mae=1.0,
                                        val_mse=(m + jitter) * 1.01))
    return pd.DataFrame(p3), pd.DataFrame(ref)


def test_fredf_recovers_known_tuning_gap():
    """调 α 一定不比固定 α=0.5 差，且 oracle-α ≥ val-选-α。"""
    p3, ref = _synth()
    s = fredf_analysis(p3, ref)["summary"]
    assert s["n_blocks"] == 48
    assert s["mean_gain_tuned_alpha"] >= s["mean_gain_fixed_alpha05"] - 1e-9
    assert s["mean_gain_oracle_alpha"] >= s["mean_gain_tuned_alpha"] - 1e-9
    assert s["tuning_gap"] > 0
    assert s["blocks_tuned_alpha_positive_pct"] == pytest.approx(100.0)


def test_fredf_detects_sqrth_stabilises_alpha():
    """构造上就是「√H 有效」：归一化后最优 α 跨 horizon 应该收敛到 1 个取值。"""
    p3, ref = _synth(sqrth_flat=True)
    s = fredf_analysis(p3, ref)["summary"]
    assert s["sqrth_mean_distinct_best_alpha_across_horizons"] == pytest.approx(1.0)
    assert (s["plain_mean_distinct_best_alpha_across_horizons"]
            > s["sqrth_mean_distinct_best_alpha_across_horizons"])
    assert s["sqrth_best_single_alpha_mean_gain"] > s["plain_best_single_alpha_mean_gain"]
    assert any("归一化让" in x for x in fredf_verdict(s))


def test_fredf_does_not_claim_sqrth_when_it_is_useless():
    """反例：让 sqrth 也带 H 依赖惩罚，结论必须转成『降级为消融』。"""
    p3, ref = _synth(sqrth_flat=False)
    s = fredf_analysis(p3, ref)["summary"]
    assert any("降级为消融" in x for x in fredf_verdict(s))


def test_multi_seed_reference_does_not_trigger_false_repeat_alarm():
    """参考表里塞 3 个 seed（且非 2021 的加了抖动），自检不应误报随机性。"""
    p3, ref = _synth(seeds=(2021, 2022, 2023))
    s = fredf_analysis(p3, ref)["summary"]
    assert s["repeat_max_std_mse"] == pytest.approx(0.0, abs=1e-9)
    assert any("复现性正常" in x for x in fredf_verdict(s))


def test_fredf_handles_missing_none_and_empty():
    p3, _ = _synth()
    no_none = p3[p3["plugin"] != "none"]
    out = fredf_analysis(no_none, None)
    assert out.get("n_cells", 0) > 0 and "note" in out          # 报错要说人话
    empty = pd.DataFrame(columns=["backbone", "dataset", "pred_len", "plugin", "mse", "val_mse"])
    assert fredf_analysis(empty, None)["n_cells"] == 0
    assert fredf_verdict({}) == ["FreDF 公平性组还没有数据。"]


def test_timesnet_is_excluded_from_fredf_group():
    """p3 刻意不含 TimesNet；万一混进来必须被剔掉，否则和 3 骨干协议不一致。"""
    p3, ref = _synth()
    bad = p3.copy()
    bad.loc[:, "backbone"] = "TimesNet"
    out = fredf_analysis(bad, None)
    assert out["n_cells"] == 0


def test_render_markdown_survives_empty_and_partial_data():
    md = _render_markdown({}, {}, "now")
    assert "PlugGate 周末无人值守报告" in md and "无数据" in md
    partial = {"coverage": {"phase2": {"config": "c", "error": "文件不存在"}}}
    assert "读取失败" in _render_markdown(partial, {}, "now")


def test_next_steps_branches_on_gating_outcome():
    assert "先补 phase-2" in _next_steps({})
    win = {"gating": {"inpool": {"n_groups": 20, "mean_gain_gate_vs_best_fixed": 1.5,
                                 "mean_gain_oracle_vs_none": 5.0}}}
    assert "门控有信号" in _next_steps(win)
    dead = {"gating": {"inpool": {"n_groups": 20, "mean_gain_gate_vs_best_fixed": 0.0,
                                  "mean_gain_oracle_vs_none": 0.3}}}
    assert "oracle 上界太低" in _next_steps(dead)
    hard = {"gating": {"inpool": {"n_groups": 20, "mean_gain_gate_vs_best_fixed": 0.0,
                                  "mean_gain_oracle_vs_none": 6.0}}}
    assert "有空间但学不到" in _next_steps(hard)


def test_report_markdown_contains_the_three_headline_numbers():
    p3, ref = _synth()
    s = fredf_analysis(p3, ref)["summary"]
    data = {
        "coverage": {"phase1": {"config": "c", "n_total": 844, "n_done": 844, "n_todo": 0,
                                "n_pruned": 0, "n_failed": 0, "pct_done": 100.0,
                                "todo_gpu_hours": 0.0, "failed_examples": []}},
        "gating": {"inpool": {"n_groups": 24, "verdict": "有信号",
                              "mean_gain_gate_vs_none": 1.2,
                              "mean_gain_bestfixed_vs_none": 0.5,
                              "mean_gain_gate_vs_best_fixed": 0.7,
                              "groups_gate_beats_best_fixed_pct": 70,
                              "groups_significant_p05": 9,
                              "mean_gain_oracle_vs_none": 4.0,
                              "mean_oracle_realized_pct": 30.0,
                              "mean_gate_acc": 0.5, "mean_majority_acc": 0.4,
                              "gate_beats_majority_pct": 75.0,
                              "mean_switch_rate": 0.6,
                              "wilcoxon_p_across_groups": 0.004}},
        "fredf": {"summary": s},
    }
    md = _render_markdown(data, {"p1_stats": "ok"}, "now")
    assert "844/844" in md
    assert "门控 vs 最佳固定插件" in md
    assert "调参差距" in md
    assert "建议的下一步" in md


# --------------------------------------------------------------------------- #
# 在线选择器章节：报告必须能在「有 / 无」在线结果两种情况下都不崩，
# 并且当静态门控失败而在线成功时，next_steps 必须给出"换主线"的建议
# --------------------------------------------------------------------------- #
def _online_summary(gain: float, delay_cost: float = 0.5) -> dict:
    return {
        "protocol": "online_delayed_feedback", "n_groups": 12, "feedback_window": 200,
        "mean_gain_online_vs_none": gain + 1.0,
        "mean_gain_online_vs_best_fixed": gain,
        "median_gain_online_vs_best_fixed": gain,
        "mean_gain_nodelay_vs_best_fixed": gain + delay_cost,
        "mean_delay_cost_pp": delay_cost,
        "mean_gain_oracle_vs_none": 6.0,
        "mean_oracle_realized_pct": 30.0,
        "groups_online_beats_best_fixed_pct": 75.0 if gain > 0 else 25.0,
        "groups_online_beats_none_pct": 80.0,
        "groups_significant_p05": 9 if gain > 0 else 1,
        "mean_switch_rate": 0.4,
        "mode": "error-feedback", "verdict": "✅ 测试用结论",
    }


def _data_with_online(gate_gain: float, online: dict | None) -> dict:
    inpool = {
        "protocol": "inpool", "n_groups": 12,
        "mean_gain_gate_vs_none": 0.3, "mean_gain_gate_vs_best_fixed": gate_gain,
        "mean_gain_bestfixed_vs_none": 1.2, "mean_gain_oracle_vs_none": 6.0,
        "mean_oracle_realized_pct": 5.0, "mean_gate_acc": 0.4,
        "mean_majority_acc": 0.42, "gate_beats_majority_pct": 40.0,
        "mean_switch_rate": 0.5, "groups_gate_beats_best_fixed_pct": 30.0,
        "groups_significant_p05": 1, "wilcoxon_p_across_groups": 0.4,
        "verdict": "🟡 测试",
    }
    gating = {"inpool": inpool}
    if online is not None:
        gating["online"] = online
    return {"coverage": {}, "gating": gating, "fredf": {}, "logs": {}}


def test_report_renders_online_section():
    from tools.weekend_report import _render_markdown
    md = _render_markdown(_data_with_online(-0.4, _online_summary(2.5)), {}, "now")
    assert "延迟误差反馈" in md
    assert "延迟**恰好 H 个窗口**" in md
    assert "+2.50%" in md
    # 必须把「静态 vs 在线」并排放出来，否则读者无法判断该走哪条主线
    assert "和静态特征门控直接对比" in md


def test_report_survives_missing_online_section():
    from tools.weekend_report import _render_markdown
    md = _render_markdown(_data_with_online(-0.4, None), {}, "now")
    assert "逐窗口门控" in md
    assert "延迟误差反馈的在线选择器" not in md


def test_next_steps_pivots_when_static_fails_but_online_wins():
    from tools.weekend_report import _next_steps
    s = _next_steps(_data_with_online(-0.4, _online_summary(2.5)))
    assert "换主线" in s and "运行时反馈" in s


def test_next_steps_reports_delay_as_the_binding_constraint():
    from tools.weekend_report import _next_steps
    s = _next_steps(_data_with_online(-0.4, _online_summary(-0.3, delay_cost=2.4)))
    assert "反馈延迟约束" in s and "2.40 pp" in s


def test_next_steps_keeps_low_oracle_priority_over_online():
    """oracle 上界太低是更根本的问题：这时不该急着换主线去追反馈信号。"""
    from tools.weekend_report import _next_steps
    d = _data_with_online(-0.4, _online_summary(2.5))
    d["gating"]["inpool"]["mean_gain_oracle_vs_none"] = 0.2
    s = _next_steps(d)
    assert "oracle 上界太低" in s and "换主线" not in s


def _sweep_summary(breakeven: float = 0.5) -> dict:
    return {
        "ref_window": 200, "n_groups": 51, "windows": [50, 200, 1000],
        "delay_ratios": [0.0, 0.5, 1.0],
        "gain_by_delay_ratio": {"0.0": 2.5, "0.5": 0.8, "1.0": -1.6},
        "gain_by_delay_and_window": {
            "0.0": {"50": 3.1, "200": 2.5, "1000": 1.4},
            "0.5": {"50": 0.9, "200": 0.8, "1000": 0.3},
            "1.0": {"50": -1.9, "200": -1.6, "1000": -0.9},
        },
        "breakeven_delay_ratio": breakeven, "best_feedback_window": 50,
        "feasible_gain_vs_best_fixed": -1.6, "cheat_gain_vs_best_fixed": 2.5,
        "mode": "error-feedback", "verdict": "⚠️ 测试用扫描结论",
    }


def test_report_renders_delay_sweep_table():
    from tools.weekend_report import _render_markdown
    d = _data_with_online(-0.4, _online_summary(-0.3, delay_cost=4.1))
    d["gating"]["online_sweep"] = _sweep_summary()
    md = _render_markdown(d, {}, "now")
    assert "盈亏平衡延迟" in md
    assert "反馈窗口=1000" in md
    # 必须显式标注哪一行是作弊、哪一行可部署，否则读者会误引用零延迟数字
    assert "作弊" in md and "可部署" in md
    assert "+2.50%" in md and "-1.60%" in md


def test_report_survives_missing_sweep():
    from tools.weekend_report import _render_markdown
    md = _render_markdown(_data_with_online(-0.4, _online_summary(-0.3)), {}, "now")
    assert "延迟误差反馈的在线选择器" in md
    assert "盈亏平衡延迟" not in md


def test_next_steps_quotes_breakeven_ratio():
    from tools.weekend_report import _next_steps
    d = _data_with_online(-0.4, _online_summary(-0.3, delay_cost=4.1))
    d["gating"]["online_sweep"] = _sweep_summary(0.5)
    s = _next_steps(d)
    assert "0.5×H" in s and "反馈延迟约束" in s


def test_report_renders_gain_attribution_decomposition():
    from tools.weekend_report import _render_markdown
    o = _online_summary(3.0)
    o.update({"mean_gain_bestfixed_test_vs_val": 3.4,
              "mean_gain_online_vs_best_fixed_test": -0.3,
              "groups_arm_id_mismatch_pct": 68.0,
              "groups_online_beats_best_fixed_test_pct": 35.0})
    md = _render_markdown(_data_with_online(-0.4, o), {}, "now")
    assert "收益来源分解" in md
    assert "best_fixed_test" in md and "选臂失配" in md
    # 时变成分为负时必须给出明确的"不能这么写"的警告
    assert "不能" in md and "窗口级自适应" in md


def test_report_credits_adaptivity_when_component_positive():
    from tools.weekend_report import _render_markdown
    o = _online_summary(3.0)
    o.update({"mean_gain_bestfixed_test_vs_val": 1.0,
              "mean_gain_online_vs_best_fixed_test": 2.0,
              "groups_arm_id_mismatch_pct": 40.0,
              "groups_online_beats_best_fixed_test_pct": 75.0})
    md = _render_markdown(_data_with_online(-0.4, o), {}, "now")
    assert "+2.00%" in md and "窗口级自适应" not in md


# --------------------------------------------------------------------------- #
# oracle 审计章节：这是报告里最重要的一节，必须出现在 headline 且能驱动 next_steps
# --------------------------------------------------------------------------- #
def _audit_summary(excess_h: float = -0.016, excess_1: float = 0.478) -> dict:
    return {
        "protocol": "argmin_identity_persistence", "n_groups": 57,
        "mean_oracle_gain_real_pct": 11.94,
        "oracle_invariant_under_relabel_all": True,
        "mean_win_rate_top": 0.489, "mean_win_rate_entropy": 0.897,
        "mean_chance_persist": 0.386, "mean_persist_lag1_relabelled": 0.320,
        "mean_persist_half_life_windows": 13.9,
        "median_persist_half_life_windows": 7.1, "mean_half_life_over_H": 0.077,
        "excess_decay_curve": {"1": 0.482, "8": 0.235, "64": 0.048, "256": -0.024},
        "mean_persist_lag1": 0.868, "mean_excess_persist_lag1": excess_1,
        "groups_z_gt2_lag1": 57,
        "mean_persist_lagH_4": 0.469, "mean_excess_persist_lagH_4": 0.078,
        "groups_z_gt2_lagH_4": 38,
        "mean_persist_lagH_2": 0.406, "mean_excess_persist_lagH_2": 0.014,
        "groups_z_gt2_lagH_2": 28,
        "mean_persist_lagH": 0.371, "mean_excess_persist_lagH": excess_h,
        "groups_z_gt2_lagH": 13,
        "mean_persist_lag2H": 0.389, "mean_excess_persist_lag2H": 0.001,
        "groups_z_gt2_lag2H": 17,
        "mode": "permutation-null", "verdict": "⚠️ 测试用审计结论：半衰期 7.1 个窗口",
    }


def test_report_puts_audit_first_in_headline():
    from tools.weekend_report import _render_markdown
    d = _data_with_online(-0.4, _online_summary(-0.3, delay_cost=4.2))
    d["gating"]["oracle_audit"] = _audit_summary()
    md = _render_markdown(d, {}, "now")
    head = md[: md.index("## 1.")]
    assert "oracle 上界审计" in head, "审计结论必须出现在 headline，它决定其他结论怎么读"
    assert "2.0 先审计 oracle 上界这个统计量本身" in md
    assert "重贴臂标签" in md and "多重集" in md
    assert "相关半衰期" in md and "7.1 个窗口" in md
    assert "衰减曲线" in md and "+0.482" in md


def test_report_survives_missing_audit():
    from tools.weekend_report import _render_markdown
    md = _render_markdown(_data_with_online(-0.4, None), {}, "now")
    assert "2.0 先审计" not in md and "逐窗口门控" in md


def test_next_steps_stops_tuning_when_audit_explains_failure():
    from tools.weekend_report import _next_steps
    d = _data_with_online(-0.4, _online_summary(-0.3, delay_cost=4.2))
    d["gating"]["oracle_audit"] = _audit_summary()
    s = _next_steps(d)
    assert "停止调门控" in s and "7.1 个窗口" in s
    assert "零 GPU" in s
    # 不该再给出"继续查特征表达力"这类已被机制解释否掉的建议
    assert "有空间但学不到" not in s


def test_next_steps_keeps_tuning_when_identity_is_predictable():
    """若合法延迟下身份仍可预测，就不能收摊，必须继续做选择器。"""
    from tools.weekend_report import _next_steps
    d = _data_with_online(-0.4, _online_summary(-0.3, delay_cost=4.2))
    d["gating"]["oracle_audit"] = _audit_summary(excess_h=0.22)
    s = _next_steps(d)
    assert "停止调门控" not in s


def test_clf_and_ext_runs_skip_expensive_lodo():
    """lodo 是报告里最慢的一步（>25min）。只有主 run 该跑它，否则会挤掉最终报告。"""
    import inspect
    from tools import weekend_report as wr
    src = inspect.getsource(wr.build_report)
    assert 'proto = "both" if key == "gating" else "inpool"' in src
    assert '"--protocol", proto' in src


def test_report_discloses_lodo_scope_limitation():
    from tools.weekend_report import _render_markdown
    d = _data_with_online(-0.4, None)
    d["gating_clf"] = {"inpool": {"n_groups": 12, "verdict": "x",
                                  "mean_gain_gate_vs_none": 0.1,
                                  "mean_gain_gate_vs_best_fixed": -0.2,
                                  "mean_gain_oracle_vs_none": 6.0,
                                  "mean_oracle_realized_pct": 2.0,
                                  "mean_switch_rate": 0.3}}
    md = _render_markdown(d, {}, "now")
    assert "只跑 inpool" in md and "lodo" in md


# --------------------------------------------------------------------------- #
# 成对 seed 审计章节
# --------------------------------------------------------------------------- #
def _seed_audit_summary(noise: float, same_corr: float = 0.5) -> dict:
    self_ = 10.0
    return {
        "n_pairs": 12, "seeds": [2021, 2022],
        "mean_self_oracle_gain_pct": self_,
        "mean_xseed_oracle_gain_pct": self_ * (1 - noise),
        "reproducible_frac_of_headroom": 1 - noise,
        "noise_frac_of_headroom": noise,
        "mean_argmin_agree": 0.9, "mean_chance_agree": 0.35,
        "mean_excess_agree": 0.55, "pairs_z_gt2": 12,
        "mean_adv_corr_across_seeds": 0.8,
        "mean_same_arm_err_corr": same_corr,
        "mean_seed_mse_gap_pct": 3.2,
        "best_fixed_arm_stable_pct": 75.0,
        "pairs_xseed_beats_best_fixed": 12,
        "verdict": "✅ 测试用结论",
    }


def test_report_renders_seed_audit_section_and_headline():
    md = _render_markdown({"seed_audit": _seed_audit_summary(0.05)}, {}, "now")
    assert "2.0b" in md
    assert "跨 seed oracle" in md
    assert "+9.50%" in md          # 跨 seed oracle = 10 * 0.95
    assert "训练随机性" in md or "训练噪声" in md
    assert "✅ 测试用结论" in md
    # headline 里也必须出现，周一第一眼就能看到
    assert md.index("✅ 测试用结论") < md.index("2.0b")


def test_report_seed_audit_surfaces_low_power_warning_row():
    """同臂误差跨 seed 相关≈1 时，报告必须把这个前提量摆出来，别让人误读成强证据。"""
    md = _render_markdown({"seed_audit": _seed_audit_summary(0.01, same_corr=0.9995)}, {}, "now")
    assert "0.9995" in md
    assert "同一个臂逐窗口误差的跨 seed 相关" in md
    assert "区分力" in md


def test_report_survives_missing_seed_audit():
    """周末 seed 复制组没跑出来时，报告必须照常生成——这是周一唯一的交付物。"""
    md = _render_markdown({}, {}, "now")
    assert "2.0b" not in md
    assert "PlugGate 周末无人值守报告" in md
    md2 = _render_markdown({"seed_audit": {}}, {}, "now")
    assert "2.0b" not in md2


def test_report_does_not_apply_gating_template_to_non_gating_protocols():
    """回归：2026-09-14 的真实报告里，controls/ablation/oracle_audit/online/online_sweep
    这 5 个 protocol 被套用了门控模板，打出整片 `+nan%`（每个 6 行，共 30 行），
    把 §2.0 的真结论淹掉了。它们各自有专门的一节，这里必须只留一行指针。"""
    gat = {
        "inpool": {"n_groups": 127, "verdict": "真门控结论",
                   "mean_gain_gate_vs_none": 0.7, "mean_gain_gate_vs_best_fixed": -1.76,
                   "mean_gain_bestfixed_vs_none": 2.02, "mean_gain_oracle_vs_none": 10.13,
                   "mean_oracle_realized_pct": 3.5, "mean_gate_acc": 0.386,
                   "mean_majority_acc": 0.486, "gate_beats_majority_pct": 2.0,
                   "mean_switch_rate": 0.26, "groups_gate_beats_best_fixed_pct": 18.0,
                   "groups_significant_p05": 62, "wilcoxon_p_across_groups": 4.3e-05},
        # 下面这些 summary 天然没有 gate_vs_* 字段
        "inpool_controls": {"n_groups": 127, "verdict": "对照结论"},
        "inpool_ablation": {"n_groups": 127},
        "oracle_audit": {"n_groups": 127, "verdict": "审计结论"},
        "online": {"n_groups": 127, "verdict": "在线结论", "mean_switch_rate": 0.61},
        "online_sweep": {"n_groups": 127, "verdict": "扫描结论"},
    }
    md = _render_markdown({"gating": gat}, {}, "now")

    # 真门控那一节照常渲染
    assert "### 协议 `inpool`（127 个决策组）" in md
    assert "-1.76%" in md
    # 5 个非门控 protocol 不再有自己的 "### 协议" 小节
    for proto in ("inpool_controls", "inpool_ablation", "oracle_audit", "online", "online_sweep"):
        assert f"### 协议 `{proto}`" not in md, proto
    # 但要留下可追溯的一行指针，不能悄悄丢掉
    assert "inpool_controls" in md and "证伪对照" in md
    assert "oracle_audit" in md and "§2.0" in md
    # 最关键：门控小节里不再出现 nan
    # 只截「协议清单」这一段（到第一个非协议小节为止），§5 里合法地提到了 NaN 窗口
    sec = md[md.index("### 协议 `inpool`"):]
    for stop in ("### 两种门控头", "> 另有", "\n## "):
        if stop in sec:
            sec = sec[:sec.index(stop)]
    assert "nan" not in sec.lower(), f"协议小节里不该再有 nan:\n{sec}"


def test_report_never_hardcodes_an_oracle_number():
    """回归：argmin 审计的 verdict 里曾把 oracle 上界写死成 11.9%，
    而同一份报告的实测值是 10.13%，两个数字自相矛盾。"""
    import re
    from src.selector_window import argmin_structure_verdict
    s = {"mean_excess_persist_lag1": 0.469, "mean_excess_persist_lagH": -0.005,
         "mean_chance_persist": 0.394, "n_groups": 127, "groups_z_gt2_lagH": 38,
         "mean_oracle_gain_real_pct": 10.13, "mean_persist_half_life_windows": 9.2,
         "median_persist_half_life_windows": 9.2, "mean_half_life_over_H": 0.06,
         "mean_win_rate_top": 0.486, "mean_win_rate_entropy": 0.88,
         "oracle_invariant_under_relabel_all": True,
         "mean_persist_lag1_relabelled": 0.307}
    v = argmin_structure_verdict(s)
    assert "10.1" in v
    assert "11.9" not in v
    # verdict 里出现的每个百分数都必须能在输入里找到来源（防止再写死别的常量）
    for num in re.findall(r"(\d+\.\d+)%", v):
        assert any(abs(float(num) - x) < 0.06
                   for x in (10.13, 6.0, 0.469, 0.005, 0.394)), num


def test_missing_metrics_render_as_dash_not_bare_nan():
    """回归：lodo 协议不产出 best_fixed_vs_none / gate_acc 等指标，
    早期版本直接打成 `+nan%`，看起来像一个真实数值。必须是 `—`。"""
    gat = {"lodo": {"n_groups": 99, "verdict": "lodo 结论",
                    "mean_gain_gate_vs_none": 0.22,
                    "mean_gain_gate_vs_best_fixed": -3.04,
                    "mean_gain_oracle_vs_none": 10.75,
                    "mean_switch_rate": 0.10,
                    "mean_gain_bestfixed_vs_none": float("nan"),
                    "mean_oracle_realized_pct": float("nan"),
                    "mean_gate_acc": None,
                    "groups_gate_beats_best_fixed_pct": 20.0,
                    "groups_significant_p05": 70,
                    "wilcoxon_p_across_groups": 1.3e-05}}
    md = _render_markdown({"gating": gat}, {}, "now")
    sec = md[md.index("### 协议 `lodo`"):]
    sec = sec[:sec.index("\n## ")] if "\n## " in sec else sec
    assert "nan" not in sec.lower(), sec
    assert "—" in sec
    # 真实存在的数值不能被误伤
    assert "+0.22%" in sec and "-3.04%" in sec and "+10.75%" in sec


# --------------------------------------------------------------------------- #
# 2026-09-14 code review 的 P1 回归：口径错误 / 协议违规 / 缺失值污染
# 这一组测试的期望值全部是**手算**出来的（增益 = 100*(mse_none-mse)/mse_none），
# 不是拿实现的输出当答案；退回旧实现每一条都会失败。
# --------------------------------------------------------------------------- #
def _cell(bb: str, ds: str, H: int, plugin: str, gain_test: float, gain_val: float) -> dict:
    """按「想要的增益」反推 mse：mse_none 固定为 1.0，于是 mse = 1 - gain/100。"""
    return dict(backbone=bb, dataset=ds, pred_len=H, plugin=plugin, seed=2021,
                mse=1.0 - gain_test / 100.0, mae=1.0, val_mse=1.0 - gain_val / 100.0)


def _alpha_scan_where_val_and_test_disagree() -> pd.DataFrame:
    """构造「验证集最优 α ≠ 测试集最优 α」的 2 个块。

    每块两个 α：α=0.1 在**验证集**上明显更好，α=0.9 在**测试集**上明显更好。
    手算：α=0.1 的测试增益均值 = (5.0+3.0)/2 = 4.0；α=0.9 = (10.0+20.0)/2 = 15.0。
    """
    rows = [_cell("DLinear", "ETTh1", 96, "none", 0.0, 0.0),
            _cell("DLinear", "ETTh1", 96, "fredf_a01", 5.0, 10.0),
            _cell("DLinear", "ETTh1", 96, "fredf_a09", 10.0, 5.0),
            _cell("DLinear", "ETTh1", 192, "none", 0.0, 0.0),
            _cell("DLinear", "ETTh1", 192, "fredf_a01", 3.0, 20.0),
            _cell("DLinear", "ETTh1", 192, "fredf_a09", 20.0, 1.0)]
    return pd.DataFrame(rows)


def test_best_single_alpha_is_picked_on_validation_never_on_test():
    """修复项 #1：`{var}_best_single_alpha_mean_gain` 必须是「**按验证集**挑出的那个 α」
    在测试集上的平均增益。旧实现用 `groupby("alpha")["gain"].mean().max()`，即按测试
    表现选 α（本模块 docstring 明令禁止），会返回 15.0 而不是 4.0。"""
    s = fredf_analysis(_alpha_scan_where_val_and_test_disagree(), None)["summary"]
    assert s["plain_best_single_alpha_by_val"] == pytest.approx(0.1)
    assert s["plain_best_single_alpha_mean_gain"] == pytest.approx(4.0)
    # 作弊上界必须保留，但只能挂在另一个带 test_selected 字样的字段名下
    assert s["plain_best_single_alpha_test_selected"] == pytest.approx(0.9)
    assert s["plain_best_single_alpha_mean_gain_test_selected"] == pytest.approx(15.0)
    # 诚实口径严格小于作弊口径：这正是旧实现被掩盖掉的那部分
    assert s["plain_best_single_alpha_mean_gain"] < s["plain_best_single_alpha_mean_gain_test_selected"]
    assert s["plain_best_single_alpha_n_blocks"] == 2


def test_best_single_alpha_only_compares_blocks_covered_by_every_alpha():
    """修复项 #1 的第二半：各 α 覆盖块数不同时，不能拿落在不同块集合上的均值比大小。

    这里给 α=0.7 只在第一个块上跑了一格，且它在验证集上很差、在测试集上极好（+50）。
    正确行为：比较只在「所有 α 都有」的块（只剩 96 那一块）上做，验证集仍然选 α=0.1，
    报出的测试增益是该块的 5.0。旧实现会把 α=0.7 那一格的 +50 当成「最优单一 α」。
    """
    df = pd.concat([_alpha_scan_where_val_and_test_disagree(),
                    pd.DataFrame([_cell("DLinear", "ETTh1", 96, "fredf_a07", 50.0, -10.0)])],
                   ignore_index=True)
    s = fredf_analysis(df, None)["summary"]
    assert s["plain_best_single_alpha_n_alphas"] == 3
    assert s["plain_best_single_alpha_n_blocks"] == 1        # 只有 96 这一块被 3 个 α 全覆盖
    assert s["plain_best_single_alpha_by_val"] == pytest.approx(0.1)
    assert s["plain_best_single_alpha_mean_gain"] == pytest.approx(5.0)
    assert s["plain_best_single_alpha_mean_gain"] < 50.0


def _unbalanced_alpha05_coverage() -> tuple[pd.DataFrame, pd.DataFrame]:
    """4 个块，只有前 2 个有 α=0.5 参考（α=0.5 只来自 p2/phase-1 的 `fredf` 臂）。

    没有参考的那 2 个块（模拟「最贵的角落」）调 α 后增益系统性更高（4.0 vs 1.5）。
    手算：
      * 配对块（96/192）：调 α 均值 1.5、固定 α=0.5 均值 1.0 -> 调参差距 **+0.5**；
      * 全部块：调 α 均值 (1.5+1.5+4.0+4.0)/4 = 2.75 -> 旧实现的差值 = 2.75-1.0 = **+1.75**。
    两者一个在 1.0 阈值以下、一个以上，正好卡在 verdict / next_steps 的分支上。
    """
    paired, lone = (96, 192), (336, 720)
    p3, ref = [], []
    for H in paired + lone:
        p3.append(_cell("DLinear", "ETTh1", H, "none", 0.0, 0.0))
        p3.append(_cell("DLinear", "ETTh1", H, "fredf_a01",
                        1.5 if H in paired else 4.0, 5.0))
        p3.append(_cell("DLinear", "ETTh1", H, "fredf_a09", 0.2, 1.0))
    for H in paired:
        ref.append(_cell("DLinear", "ETTh1", H, "none", 0.0, 0.0))
        ref.append(_cell("DLinear", "ETTh1", H, "fredf", 1.0, 3.0))
    return pd.DataFrame(p3), pd.DataFrame(ref)


def test_tuning_gap_is_a_paired_difference_over_the_same_blocks():
    """修复项 #2：`tuning_gap` 必须是同一批块上的差，不能拿两个不同块集合的均值相减。"""
    p3, ref = _unbalanced_alpha05_coverage()
    s = fredf_analysis(p3, ref)["summary"]
    assert s["n_blocks"] == 4
    assert s["n_blocks_paired_alpha05"] == 2                 # 配对块数必须进 summary
    assert s["mean_gain_fixed_alpha05"] == pytest.approx(1.0)
    assert s["mean_gain_tuned_alpha"] == pytest.approx(1.5)  # 配对块上的调 α 均值
    assert s["tuning_gap"] == pytest.approx(0.5)
    # 全块口径可以另存，但绝不能参与减法
    assert s["mean_gain_tuned_alpha_all_blocks"] == pytest.approx(2.75)
    naive_gap = s["mean_gain_tuned_alpha_all_blocks"] - s["mean_gain_fixed_alpha05"]
    assert naive_gap == pytest.approx(1.75)
    assert abs(naive_gap - s["tuning_gap"]) > 1.0            # 本用例确实有区分力


def test_unpaired_inflation_no_longer_triggers_the_plug_and_play_claim():
    """修复项 #2 的下游：旧实现的 +1.75 会越过 1.0 阈值，让报告写下
    「FreDF 的收益主要来自调参，这是论文最有力的一句话」并在下一步里立项。
    配对后差距只有 +0.5，两处结论都必须改口。"""
    p3, ref = _unbalanced_alpha05_coverage()
    s = fredf_analysis(p3, ref)["summary"]
    v = " ".join(fredf_verdict(s))
    assert "调参挽回有限" in v
    assert "最有力的一句话" not in v
    assert "在 2 个同时有两种口径的配对块上" in v             # 配对块数要写在结论里
    steps = _next_steps({"fredf": {"summary": s},
                         "gating": {"inpool": {"n_groups": 5,
                                               "mean_gain_gate_vs_best_fixed": 0.0,
                                               "mean_gain_oracle_vs_none": 6.0}}})
    assert "即插即用性被高估" not in steps


def test_fredf_table_labels_the_test_selected_alpha_as_cheating():
    """修复项 #1 的交付端：诚实口径与作弊口径必须**同时出现且各自标明**，
    否则读者（= 直接抄数字进论文的作者）无法分辨。"""
    s = fredf_analysis(_alpha_scan_where_val_and_test_disagree(), None)["summary"]
    md = _render_markdown({"fredf": {"summary": s}}, {}, "now")
    sec = md[md.index("## 3."):]
    assert "**按验证集**选出的单一 α" in sec
    assert "+4.00" in sec                                    # 诚实口径
    assert "按**测试集**选单一 α" in sec and "作弊上界" in sec
    assert "+15.00" in sec                                   # 作弊上界，且被标成作弊
    assert "不可引用为方法结果" in sec


# --------------------------------------------------------------------------- #
# 修复项 #3：子步骤失败时绝不能读上一轮的 summary json
# --------------------------------------------------------------------------- #
def test_step_summary_reads_json_only_when_the_step_succeeded(tmp_path):
    from tools.weekend_report import _step_summary
    p = tmp_path / "selector_window_summary.json"
    p.write_text('{"inpool": {"n_groups": 127, "verdict": "上一轮的结论"}}', encoding="utf-8")
    # 成功：照读
    got = _step_summary(True, p, log_key="selector_window_gating")
    assert got["inpool"]["n_groups"] == 127
    # 失败（含 7200s 超时）：即使旧文件还在，也**不能**把它当本轮结论
    bad = _step_summary(False, p, log_key="selector_window_gating")
    assert "inpool" not in bad
    assert "error" in bad and "本轮没有产出" in bad["error"]
    # 旧文件的存在必须被披露，并带上 mtime，供人显式决定要不要引用上一轮
    assert bad["stale_json"] == str(p)
    assert bad["stale_json_mtime"] and "未采用" in bad["error"]
    # 子进程返回 0 但没写文件，也算没有产出
    missing = _step_summary(True, tmp_path / "nope.json", log_key="selector_window_gating")
    assert "error" in missing and "stale_json" not in missing


def test_report_surfaces_failed_steps_in_the_headline_instead_of_stale_numbers():
    """修复项 #3 的渲染端：失败信息必须出现在 §0 headline 与 §2 正文，
    而不是只躺在报告末尾折叠的执行日志里。"""
    err = {"error": "本轮没有产出：子进程失败或超时（见执行日志 selector_window_gating）；"
                    "磁盘上仍有上一轮的 selector_window_summary.json"
                    "（写于 2026-09-07 03:00:00），本报告**未采用**它",
           "failed_step": "selector_window_gating",
           "stale_json": "artifacts/p2/selector_window_summary.json",
           "stale_json_mtime": "2026-09-07 03:00:00"}
    md = _render_markdown({"gating": err}, {}, "now")
    head = md[: md.index("## 1.")]
    assert "⛔" in head and "本轮没有产出" in head
    assert "不得引用" in head
    sec = md[md.index("## 2."):md.index("## 3.")]
    assert "本节本轮没有产出" in sec
    assert "2026-09-07 03:00:00" in sec and "数据来自上一轮" in sec
    # 失败时不能渲染出任何门控小节（那会让读者以为有本轮数字）
    assert "### 协议 `inpool`" not in md
    assert "nan" not in sec.lower()


# --------------------------------------------------------------------------- #
def test_pct_or_dash_never_prints_a_bare_nan():
    """缺失的百分数必须渲染成 `—`，而不是 `nan%` 或 `nan`。

    这条约束以前藏在一行 187 字符的内联三元表达式里，没有任何测试守着；
    只要有人改格式串就可能让裸 `nan` 混进正文冒充一个真实数值。
    """
    assert _pct_or_dash(12.34) == "12.3%"
    assert _pct_or_dash(0.0) == "0.0%"
    assert _pct_or_dash(-3.0) == "-3.0%"
    for bad in (None, float("nan"), "—", "", object()):
        assert _pct_or_dash(bad) == "—", bad


def test_pct_or_dash_honours_the_format_spec():
    assert _pct_or_dash(12.345, ".2f") == "12.35%"
    assert _pct_or_dash(12.345, "+.0f") == "+12%"
