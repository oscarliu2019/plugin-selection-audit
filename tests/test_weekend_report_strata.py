"""§2.0a 分层普遍性章节的渲染回归。

审稿人对「127 组平均下来没信号」最自然的反驳是「你平均掉了」，所以分层结论
必须出现在报告里、且必须出现在 headline（它和 §2.0 是同一条论证的两半）。
这一组测试锁死：有分层数据时该节和两张图必须出现；没有时报告不能崩。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _strata_summary(exc: float = 0.033, below: int = 20, above: int = 0) -> dict:
    return {
        "protocol": "audit_strata",
        "n_strata": 20, "n_strata_underpowered": 0,
        "strata_keys": ["backbone", "dataset", "pred_len"],
        "strata_total_used": 20,
        "strata_with_half_life_over_H_below_25pct": below,
        "strata_with_excess_lagH_above_005": above,
        "max_median_half_life_over_H": 0.187,
        "max_median_half_life_over_H_stratum": "pred_len=24",
        "max_excess_persist_lagH": exc,
        "max_excess_persist_lagH_stratum": "dataset=Exchange",
        "median_half_life_windows_range": [2.9, 14.1],
        "horizon_hl_spearman": 0.74, "horizon_ratio_spearman": -0.88,
        "hl_at_min_H": 4.5, "hl_at_max_H": 14.1, "H_min": 24.0, "H_max": 720.0,
        "verdict": "✅ 测试用分层结论：20/20 层半衰期都远短于 H",
    }


def _data(strata: dict | None, figs: dict | None = None) -> dict:
    d: dict = {"gating": {"inpool": {"n_groups": 127, "verdict": "门控输给最佳固定臂",
                                     "mean_gain_gate_vs_best_fixed": -1.76}}}
    if strata is not None:
        d["gating"]["audit_strata"] = strata
    if figs is not None:
        d["audit_figs"] = figs
    return d


def test_report_renders_strata_section_and_headline():
    from tools.weekend_report import _render_markdown
    md = _render_markdown(_data(_strata_summary()), {}, "now")
    head = md[: md.index("## 1.")]
    assert "分层普遍性" in head, "分层结论必须进 headline：它是 §2.0 抗反驳的那一半"
    assert "2.0a" in md
    # 三种切法、两个硬判据、以及「最像有信号的那层」都必须写出来
    assert "backbone" in md and "dataset" in md and "pred_len" in md
    assert "20/20" in md and "0/20" in md
    assert "dataset=Exchange" in md and "+0.033" in md
    assert "2.9 ~ 14.1 个窗口" in md
    # horizon 依赖的两个 Spearman 是本节最有信息量的一句话
    assert "+0.74" in md and "-0.88" in md
    assert "追不上 H 本身的增长" in md
    assert "✅ 测试用分层结论" in md


def test_report_embeds_both_audit_figures_when_present():
    from tools.weekend_report import _render_markdown
    figs = {"fig_halflife_vs_horizon": "artifacts/paper/fig_halflife_vs_horizon.png",
            "fig_decay_curves": "artifacts/paper/fig_decay_curves.png"}
    md = _render_markdown(_data(_strata_summary(), figs), {}, "now")
    assert "![半衰期 vs 预测视界](artifacts/paper/fig_halflife_vs_horizon.png)" in md
    assert "![分层衰减曲线](artifacts/paper/fig_decay_curves.png)" in md


def test_report_omits_figures_that_were_not_generated():
    """画图这一步是包在 try 里的，失败时 data['audit_figs'] 只有一半甚至没有。
    报告绝不能因此写出指向不存在文件的 ![](...)。"""
    from tools.weekend_report import _render_markdown
    md = _render_markdown(_data(_strata_summary(), {"fig_decay_curves": "x/fig_decay_curves.png"}),
                          {}, "now")
    assert "fig_halflife_vs_horizon" not in md
    assert "![分层衰减曲线](x/fig_decay_curves.png)" in md
    md2 = _render_markdown(_data(_strata_summary(), {}), {}, "now")
    assert "![" not in md2.split("2.0a")[1].split("\n## ")[0]


def test_figure_paths_in_report_are_relative_to_results_dir():
    """报告写在 results/ 下、图在 artifacts/paper/ 下。写绝对路径的话，
    这份 md 一 rsync 到别的机器（或本地）看，两张图就全是断链。

    2026-09-14 review：这条以前是对源码正则匹配（`re.search(r"figs\\[n\\] = ..."`），
    实现一改就假失败/假通过；改成直接调用那段逻辑做**行为断言**。
    """
    from tools import weekend_report as wr
    ref = wr._fig_ref_path(wr.ROOT / "artifacts" / "paper" / "fig_decay_curves.png")
    assert not Path(ref).is_absolute(), ref
    # 从 results/ 出发能走到那个文件，才算真的可用
    assert (wr.ROOT / "results" / ref).resolve() == (
        wr.ROOT / "artifacts" / "paper" / "fig_decay_curves.png").resolve()
    # 报告里就是这么被引用的
    md = _render_markdown_with_figs({"fig_decay_curves": ref})
    assert f"![分层衰减曲线]({ref})" in md


def _render_markdown_with_figs(figs: dict) -> str:
    from tools.weekend_report import _render_markdown
    return _render_markdown(_data(_strata_summary(), figs), {}, "now")


def test_report_survives_missing_or_empty_strata():
    from tools.weekend_report import _render_markdown
    assert "2.0a" not in _render_markdown(_data(None), {}, "now")
    assert "2.0a" not in _render_markdown(_data({"protocol": "audit_strata", "n_strata": 0}),
                                          {}, "now")


def test_strata_is_not_rendered_with_the_gating_template():
    """audit_strata 的 summary 里没有 gate_vs_best_fixed 这一族字段（也没有 n_groups）。
    2026-09-14 的 nan 事故就是这么来的：非门控 protocol 被套了门控模板。
    这里锁死它只出现在 §2.0a，绝不生成一个 `### 协议 audit_strata` 的 nan 表。"""
    from tools.weekend_report import _render_markdown
    md = _render_markdown(_data(_strata_summary()), {}, "now")
    assert "### 协议 `audit_strata`" not in md
    # 「NaN 窗口」是已知问题一节的正当散文，这里只禁裸数值 nan
    assert "nan%" not in md.lower() and "=nan" not in md.lower()
    assert "nan" not in md[md.index("2.0a"):md.index("\n## ", md.index("2.0a"))].lower()
    # 就算将来给它补上 n_groups，也必须走「一行指针」而不是门控模板。
    # 2026-09-14 review：这条以前直接断言源码里有 `"audit_strata": "见 §2.0a"`；
    # 改成行为断言——真的塞一个 n_groups 进去，看渲染结果。
    d = _data(dict(_strata_summary(), n_groups=41))
    md2 = _render_markdown(d, {}, "now")
    assert "### 协议 `audit_strata`" not in md2
    assert "audit_strata" in md2 and "见 §2.0a" in md2


# --------------------------------------------------------------------------- #
# 2026-09-14 code review 修复项 #4：正文不许出现写死的实验规模数字
# --------------------------------------------------------------------------- #
def _audit(n_groups: int) -> dict:
    """§2.0 的 oracle 审计 summary（只放渲染要用到的字段）。"""
    return {"protocol": "argmin_identity_persistence", "n_groups": n_groups,
            "mean_oracle_gain_real_pct": 10.13,
            "oracle_invariant_under_relabel_all": True,
            "mean_win_rate_top": 0.486, "mean_win_rate_entropy": 0.88,
            "mean_chance_persist": 0.394, "mean_persist_lag1_relabelled": 0.307,
            "mean_persist_half_life_windows": 9.2,
            "median_persist_half_life_windows": 7.1, "mean_half_life_over_H": 0.06,
            "excess_decay_curve": {"1": 0.469, "64": 0.03},
            "mean_persist_lag1": 0.86, "mean_excess_persist_lag1": 0.469,
            "groups_z_gt2_lag1": n_groups,
            "mean_persist_lagH_4": 0.47, "mean_excess_persist_lagH_4": 0.07,
            "groups_z_gt2_lagH_4": 3,
            "mean_persist_lagH_2": 0.41, "mean_excess_persist_lagH_2": 0.01,
            "groups_z_gt2_lagH_2": 2,
            "mean_persist_lagH": 0.39, "mean_excess_persist_lagH": -0.005,
            "groups_z_gt2_lagH": 1,
            "mean_persist_lag2H": 0.39, "mean_excess_persist_lag2H": 0.001,
            "groups_z_gt2_lag2H": 1, "verdict": "⚠️ 审计结论"}


def test_strata_section_quotes_the_measured_group_count():
    """§2.0a 原来写死「127 个决策组」，而同一份 md 的 §2.0 用的是实测 n_groups。
    覆盖度一变，同一份报告里就出现两个自相矛盾的组数（作者会照抄进论文）。"""
    from tools.weekend_report import _render_markdown
    d = _data(_strata_summary())
    d["gating"]["inpool"]["n_groups"] = 41        # 数据里没有 127，凡出现即写死
    d["gating"]["oracle_audit"] = _audit(41)
    md = _render_markdown(d, {}, "now")
    sec = md[md.index("2.0a"):]
    assert "41 个决策组" in sec
    assert "127" not in md, "报告正文不得写死实验规模"
    # 换一个组数，正文必须跟着变（证明它真的来自数据而不是另一个常量）
    d["gating"]["oracle_audit"] = _audit(88)
    md2 = _render_markdown(d, {}, "now")
    sec2 = md2[md2.index("2.0a"):md2.index("### 协议")]
    assert "88 个决策组" in sec2 and "41 个决策组" not in sec2


def test_report_and_next_steps_contain_no_hardcoded_experiment_scale():
    """全文体检：几个曾经写死的规模数字都不许再出现（127 组 / ILI 74 个验证窗口 /
    ETTm1 11425 窗口 / 20 个 NaN 窗口 / horizon 写成 96~720——ILI 实际是 24~60）。"""
    from tools.weekend_report import _next_steps, _render_markdown
    d = _data(_strata_summary())
    d["gating"]["inpool"]["n_groups"] = 41        # 数据里没有 127，凡出现即写死
    d["gating"]["oracle_audit"] = _audit(41)
    d["gating"]["inpool_learnable"] = {"n_groups": 12, "filter": "训练窗口 ≥1000",
                                       "verdict": "x", "mean_gain_gate_vs_best_fixed": -1.0,
                                       "groups_gate_beats_best_fixed_pct": 10,
                                       "groups_significant_p05": 1,
                                       "mean_gain_oracle_vs_none": 6.0,
                                       "mean_oracle_realized_pct": 2.0}
    text = _render_markdown(d, {}, "now") + "\n" + _next_steps(d)
    for forbidden in ("127 个决策组", "74 个验证窗口", "11425", "20 个 NaN 窗口", "96~720"):
        assert forbidden not in text, forbidden


def test_next_steps_quotes_the_measured_horizon_range():
    """「合法延迟是 H（96~720 个窗口）」原来是写死的，ILI 的 H 实际是 24~60。
    必须从分层审计的实测 H_min/H_max 现读；读不到就不许编一个范围出来。"""
    from tools.weekend_report import _next_steps
    d = _data(dict(_strata_summary(), H_min=24.0, H_max=720.0))
    d["gating"]["inpool"]["n_groups"] = 41
    d["gating"]["oracle_audit"] = _audit(41)
    s = _next_steps(d)
    assert "停止调门控" in s                      # 走的是那条会引用 H 范围的分支
    assert "24~720 个窗口" in s and "96~720" not in s
    # 没有分层数据时，不许凭空写一个 horizon 范围
    d2 = _data(None)
    d2["gating"]["inpool"]["n_groups"] = 41
    d2["gating"]["oracle_audit"] = _audit(41)
    s2 = _next_steps(d2)
    assert "停止调门控" in s2
    assert "个窗口）" not in s2.split("合法延迟是 H")[1][:20]
