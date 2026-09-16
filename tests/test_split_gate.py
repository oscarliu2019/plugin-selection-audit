"""同 split 内部时序切分（gate-train / gate-eval）与覆盖缺口表的测试。

这两个协议是为堵两条具体审稿意见而加的，所以测试的重点不是"能不能跑通"，
而是**它们声称的严格性是否真的成立**：

* 切分必须是时序的、训练段与评估段不重叠；
* 门控只能看到训练段（用拦截 ``_fit_gate`` 的方式断言，而不是相信注释）；
* ``best_fixed`` 必须用**训练段**选臂——否则基线偷看了评估段，整个诊断失去意义；
* ``source="test"`` 必须在没有 val 逐窗口的组上也能跑（这是它能覆盖全部组的前提）。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import src.selector_window as sw
from src.selector_window import (
    Group,
    SPLIT_MIN_EVAL,
    SPLIT_MIN_TRAIN,
    coverage_gap_table,
    coverage_gap_verdict,
    evaluate_split_gate,
    split_gate_verdict,
    summarize,
)


def _mk_group(n_val=1000, n_test=1000, n_feat=3, with_val=True, seed=0,
              backbone="BB", dataset="DS"):
    rng = np.random.default_rng(seed)
    e_test = np.abs(rng.normal(1.0, 0.2, size=(2, n_test)))
    x_test = rng.normal(size=(n_test, n_feat))
    if with_val:
        e_val = np.abs(rng.normal(1.0, 0.2, size=(2, n_val)))
        x_val = rng.normal(size=(n_val, n_feat))
        val_mean = None
    else:
        e_val = x_val = None
        val_mean = np.array([1.0, 1.1])
    # seed 进 key，保证同一批测试组的 Group.key 互不相同（真实数据里 key 也含 seed）
    return Group(backbone, dataset, 96, 2021 + seed, ["none", "arm1"],
                 e_val, e_test, x_val, x_test,
                 [f"w_f{i}" for i in range(n_feat)], val_mean=val_mean)


# --------------------------------------------------------------------------- #
# 切分本身的严格性
# --------------------------------------------------------------------------- #
def test_split_sizes_and_no_overlap():
    g = _mk_group(n_test=1000)
    r = evaluate_split_gate(g, source="test", train_frac=0.7)
    assert r is not None
    assert r["n_val_windows"] == 700 and r["n_test_windows"] == 300
    assert r["n_val_windows"] + r["n_test_windows"] == r["n_windows_total"] == 1000


def test_gate_only_sees_the_train_prefix(monkeypatch):
    """拦截 _fit_gate：它拿到的必须是**前段**，且长度恰好等于 train_frac 那一段。"""
    g = _mk_group(n_test=1000)
    seen: dict[str, np.ndarray] = {}
    real = sw._fit_gate

    def spy(x_tr, e_tr, arms, i_none, mode, seed=26):
        seen["x"] = np.array(x_tr, copy=True)
        seen["e"] = np.array(e_tr, copy=True)
        return real(x_tr, e_tr, arms, i_none, mode, seed)

    monkeypatch.setattr(sw, "_fit_gate", spy)
    evaluate_split_gate(g, source="test", train_frac=0.7)

    assert seen["x"].shape[0] == 700
    assert seen["e"].shape[1] == 700
    # 必须是前 700 个窗口（时序前缀），不是随机 700 个
    assert np.array_equal(seen["x"], g.x_test[:700])
    assert np.array_equal(seen["e"], g.e_test[:, :700])


def test_insufficient_windows_returns_none():
    # 评估段不足
    g = _mk_group(n_test=SPLIT_MIN_TRAIN + SPLIT_MIN_EVAL - 10)
    assert evaluate_split_gate(g, source="test", train_frac=0.9) is None
    # 训练段不足
    g2 = _mk_group(n_test=SPLIT_MIN_TRAIN + SPLIT_MIN_EVAL + 50)
    assert evaluate_split_gate(g2, source="test", train_frac=0.1) is None
    # 两段都够就必须有结果
    g3 = _mk_group(n_test=2 * (SPLIT_MIN_TRAIN + SPLIT_MIN_EVAL))
    assert evaluate_split_gate(g3, source="test", train_frac=0.7) is not None


# --------------------------------------------------------------------------- #
# 基线的口径：best_fixed 必须只用训练段选臂
# --------------------------------------------------------------------------- #
def test_best_fixed_uses_train_prefix_not_eval_segment():
    """构造「前段 arm1 更好、后段 none 更好」的臂漂移，基线必须选前段赢家 arm1。"""
    n = 1000
    n_tr = 700
    e = np.ones((2, n))
    e[0, :n_tr] = 2.0      # none 在前段差
    e[1, :n_tr] = 1.0      # arm1 在前段好  -> 训练段选 arm1
    e[0, n_tr:] = 1.0      # none 在后段好
    e[1, n_tr:] = 3.0      # arm1 在后段差
    rng = np.random.default_rng(0)
    g = Group("BB", "DS", 96, 2021, ["none", "arm1"], None, e, None,
              rng.normal(size=(n, 2)), ["w_a", "w_b"], val_mean=np.array([1.0, 9.0]))

    r = evaluate_split_gate(g, source="test", train_frac=0.7)
    assert r is not None
    # 前段赢家是 arm1，它在评估段的 MSE 是 3.0
    assert r["best_fixed_split"] == "arm1"
    assert r["mse_best_fixed"] == pytest.approx(3.0)
    # 而评估段事后最优固定臂是 none -> 必须被标成「不一致」
    assert r["bf_split_is_eval_argmin"] is False
    # 参照口径（整段 val 均值）选的是 none，在评估段是 1.0
    assert r["best_fixed_val"] == "none"
    assert r["mse_best_fixed_valmean"] == pytest.approx(1.0)
    # 两个口径给出的相对增益必须不同，否则说明代码把它们算成了同一个东西
    assert r["gain_gate_vs_best_fixed"] != pytest.approx(r["gain_gate_vs_best_fixed_valmean"])


def test_all_quantities_are_settled_on_eval_segment():
    """none / oracle 必须只用评估段：用常数臂构造可解析验证的数值。"""
    n, n_tr = 1000, 700
    e = np.empty((2, n))
    e[0, :n_tr], e[1, :n_tr] = 5.0, 4.0
    e[0, n_tr:], e[1, n_tr:] = 2.0, 8.0
    rng = np.random.default_rng(1)
    g = Group("BB", "DS", 96, 2021, ["none", "arm1"], None, e, None,
              rng.normal(size=(n, 2)), ["w_a", "w_b"], val_mean=np.array([1.0, 2.0]))
    r = evaluate_split_gate(g, source="test", train_frac=0.7)
    assert r["mse_none"] == pytest.approx(2.0)        # 评估段的 none，不是全段
    assert r["mse_oracle"] == pytest.approx(2.0)      # 评估段逐窗口 min


# --------------------------------------------------------------------------- #
# source 的可用性：test 内部切分必须不依赖 val 逐窗口
# --------------------------------------------------------------------------- #
def test_test_source_works_without_val_windows():
    g = _mk_group(with_val=False, n_test=1000)
    assert not g.has_val_windows
    assert evaluate_split_gate(g, source="test") is not None      # 覆盖全部组的前提
    assert evaluate_split_gate(g, source="val") is None           # val 内部切分则不可用


def test_bad_source_raises():
    with pytest.raises(ValueError, match="source"):
        evaluate_split_gate(_mk_group(), source="train")


# --------------------------------------------------------------------------- #
# 有信号时必须能发现（否则这个诊断没有区分力）
# --------------------------------------------------------------------------- #
def test_split_gate_finds_real_signal():
    """特征直接决定哪个臂更好，且前后段同规律 -> 门控必须赢过固定臂。"""
    n = 2000
    rng = np.random.default_rng(7)
    x = rng.normal(size=(n, 2))
    flag = x[:, 0] > 0
    e = np.empty((2, n))
    e[0] = np.where(flag, 2.0, 1.0) + rng.normal(0, 0.05, n)      # none
    e[1] = np.where(flag, 1.0, 2.0) + rng.normal(0, 0.05, n)      # arm1
    g = Group("BB", "DS", 96, 2021, ["none", "arm1"], None, np.abs(e), None, x,
              ["w_a", "w_b"], val_mean=np.array([1.5, 1.5]))
    r = evaluate_split_gate(g, source="test", train_frac=0.7)
    assert r["gain_gate_vs_best_fixed"] > 10.0
    assert r["gate_acc"] > 0.9


def test_split_gate_does_not_report_signal_on_pure_noise():
    """特征与哪个臂更好完全无关 -> 不能赢过固定臂（允许 ±1% 的噪声带）。"""
    n = 2000
    rng = np.random.default_rng(11)
    x = rng.normal(size=(n, 4))
    e = np.abs(rng.normal(1.0, 0.3, size=(2, n)))
    g = Group("BB", "DS", 96, 2021, ["none", "arm1"], None, e, None, x,
              [f"w_f{i}" for i in range(4)], val_mean=np.array([1.0, 1.0]))
    r = evaluate_split_gate(g, source="test", train_frac=0.7)
    assert r["gain_gate_vs_best_fixed"] < 1.0


# --------------------------------------------------------------------------- #
# 与 summarize / verdict 的接口兼容
# --------------------------------------------------------------------------- #
def test_rows_are_summarizable():
    rows = [evaluate_split_gate(_mk_group(seed=i, n_test=1000), source="test")
            for i in range(10)]
    df = pd.DataFrame([r for r in rows if r is not None])
    s = summarize(df, "split_test")
    assert s["n_groups"] == 10
    for k in ("mean_gain_gate_vs_best_fixed", "mean_gain_oracle_vs_none",
              "wilcoxon_p_across_groups", "mean_gate_acc"):
        assert k in s


def test_split_verdict_wording_tracks_sign():
    neg = {"n_groups": 96, "mean_gain_gate_vs_best_fixed": -0.7,
           "groups_gate_beats_best_fixed_pct": 20.0, "mean_gain_oracle_vs_none": 9.0}
    pos = dict(neg, mean_gain_gate_vs_best_fixed=+1.2)
    assert "排除" in split_gate_verdict(neg, neg)
    assert "转正" in split_gate_verdict(pos, pos)
    # 只要有一个 split 为负，就不能宣称"漂移才是原因"
    assert "排除" in split_gate_verdict(pos, neg)
    assert "无结论" in split_gate_verdict(None, None)


# --------------------------------------------------------------------------- #
# 覆盖缺口表
# --------------------------------------------------------------------------- #
def _cov_groups():
    gs = [_mk_group(with_val=True, seed=i, backbone="DLinear") for i in range(3)]
    gs += [_mk_group(with_val=False, seed=10 + i, backbone="TimesNet") for i in range(2)]
    return gs


def test_coverage_gap_counts():
    df = coverage_gap_table(_cov_groups())
    d = df.set_index("backbone")
    assert d.loc["DLinear", "n_groups_test_side"] == 3
    assert d.loc["DLinear", "n_missing"] == 0
    assert d.loc["TimesNet", "n_missing"] == 2
    assert d.loc["TimesNet", "pct_gating_covered"] == 0.0
    assert d.loc["ALL_gating_covered", "n_groups_test_side"] == 3
    assert d.loc["ALL_gating_missing", "n_groups_test_side"] == 2
    # 缺口原因必须写清楚，不能是空字符串
    assert "order.json" in d.loc["TimesNet", "reason_missing"]


def test_coverage_gap_merges_audit_metrics_and_judges_homogeneity():
    gs = _cov_groups()
    audit = pd.DataFrame([
        {"key": g.key,
         "oracle_gain_real_pct": 10.0,
         "persist_half_life_windows": 9.0,
         "half_life_over_H": 0.06,
         "excess_persist_lagH": -0.01}
        for g in gs
    ])
    df = coverage_gap_table(gs, audit)
    assert "median_half_life_over_H" in df.columns
    mis = df[df["backbone"] == "ALL_gating_missing"].iloc[0]
    assert mis["median_half_life_over_H"] == pytest.approx(0.06)
    assert "缺口不影响结论" in coverage_gap_verdict(df)


def test_coverage_gap_flags_heterogeneous_missing_groups():
    """未覆盖组若在合法延迟下真有信号，必须变成警告而不是放行。"""
    gs = _cov_groups()
    rows = []
    for g in gs:
        bad = not g.has_val_windows
        rows.append({"key": g.key, "oracle_gain_real_pct": 10.0,
                     "persist_half_life_windows": 200.0 if bad else 9.0,
                     "half_life_over_H": 0.9 if bad else 0.06,
                     "excess_persist_lagH": 0.3 if bad else -0.01})
    df = coverage_gap_table(gs, pd.DataFrame(rows))
    assert "缺口可能影响结论" in coverage_gap_verdict(df)


def test_coverage_gap_without_audit_is_still_usable():
    df = coverage_gap_table(_cov_groups(), None)
    assert "median_half_life_over_H" not in df.columns
    assert "无法比较" in coverage_gap_verdict(df)


def test_coverage_gap_no_gap_case():
    gs = [_mk_group(with_val=True, seed=i) for i in range(3)]
    assert "无缺口" in coverage_gap_verdict(coverage_gap_table(gs))


def test_coverage_gap_survives_duplicate_audit_keys():
    """审计表里 key 重复（拼接多个来源）不能让覆盖表崩掉。"""
    gs = _cov_groups()
    one = [{"key": g.key, "oracle_gain_real_pct": 10.0,
            "persist_half_life_windows": 9.0, "half_life_over_H": 0.06,
            "excess_persist_lagH": -0.01} for g in gs]
    df = coverage_gap_table(gs, pd.DataFrame(one + one))      # 每个 key 出现两次
    assert df[df["backbone"] == "ALL_gating_missing"].iloc[0][
        "median_half_life_over_H"] == pytest.approx(0.06)
