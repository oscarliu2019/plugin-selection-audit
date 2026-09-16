"""逐窗口门控的测试：既要能在有信号时发现信号，更要在无信号时**不**报喜。"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features_window import window_features
from src.selector_window import (
    Group,
    _align_pair,
    build_groups,
    evaluate_group,
    evaluate_seed_pair,
    pair_seed_groups,
    parse_cell_id,
    seed_audit_verdict,
    summarize,
    summarize_seed_audit,
    verdict,
)
from src.runner import mark_deterministic_order, save_window_mse_atomic

REPO = Path(__file__).resolve().parents[1]


def _save_val_window_mse(val_dir, cid, arr):
    """写 val 逐窗口误差并打上确定序标记。

    ``build_groups`` 只接受带 ``.order.json`` 的 val 数组（2026-09-14 P0：TSLib 的
    val loader 是 shuffle=True，旧产物顺序被打乱）。测试里的 val 数组当然是确定序的，
    所以这里显式打标；未打标的拒绝路径由 tests/test_val_order_provenance.py 覆盖。
    """
    save_window_mse_atomic(val_dir, cid, arr)
    mark_deterministic_order(val_dir, cid, int(np.asarray(arr).size))


def _group(x_val, e_val, x_test, e_test, arms=("none", "arm1")):
    return Group("BB", "DS", 96, 2021, list(arms), e_val, e_test, x_val, x_test,
                 [f"w_f{i}" for i in range(x_val.shape[1])])


# --------------------------------------------------------------------------- #
# cell_id 解析
# --------------------------------------------------------------------------- #
def test_parse_cell_id_roundtrip():
    m = parse_cell_id("DLinear__ETTh1__h336__fredf_sqrth__s2021")
    assert m == {"backbone": "DLinear", "dataset": "ETTh1", "pred_len": 336,
                 "plugin": "fredf_sqrth", "seed": 2021}
    assert parse_cell_id("garbage") is None
    assert parse_cell_id("a__b__x96__c__s1") is None      # horizon 不是 h 开头


# --------------------------------------------------------------------------- #
# 有信号 / 无信号
# --------------------------------------------------------------------------- #
def test_gate_finds_signal_when_it_exists():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.standard_normal((n, 3))
    base = 1.0 + 0.05 * rng.standard_normal(n)
    e = np.stack([base, np.where(x[:, 0] > 0, base * 0.5, base * 2.0)])
    h = n // 2
    r = evaluate_group(_group(x[:h], e[:, :h], x[h:], e[:, h:]))
    assert r["gain_gate_vs_best_fixed"] > 5.0
    assert r["gate_acc"] > 0.9
    assert 0.3 < r["switch_rate"] < 0.7          # 大约一半窗口该挂插件


def test_gate_does_not_fake_a_win_when_features_are_noise():
    """反向保险：特征与哪个臂更好完全无关时，门控不应显著赢过最佳固定臂。

    如果这条断言炸了，几乎一定是实现里把测试段误差泄漏进了训练。
    """
    rng = np.random.default_rng(1)
    n = 4000
    x = rng.standard_normal((n, 4))                     # 与臂优劣无关的纯噪声特征
    base = 1.0 + 0.05 * rng.standard_normal(n)
    flip = rng.random(n) > 0.5                          # 谁更好由独立随机决定
    e = np.stack([base, np.where(flip, base * 0.8, base * 1.25)])
    h = n // 2
    r = evaluate_group(_group(x[:h], e[:, :h], x[h:], e[:, h:]))
    assert r["gain_gate_vs_best_fixed"] < 2.0, "无信号却赢很多 -> 疑似泄漏"
    assert r["gain_oracle_vs_none"] > 5.0, "oracle 仍应有空间（它可以偷看测试误差）"


def test_gate_falls_back_to_none_when_plugin_is_always_bad():
    """插件永远有害时，回归型门控应该几乎从不挂插件。"""
    rng = np.random.default_rng(2)
    n = 2000
    x = rng.standard_normal((n, 3))
    base = 1.0 + 0.05 * rng.standard_normal(n)
    e = np.stack([base, base * 1.5])
    h = n // 2
    r = evaluate_group(_group(x[:h], e[:, :h], x[h:], e[:, h:]), mode="reg")
    assert r["switch_rate"] < 0.05
    assert r["gain_gate_vs_none"] > -0.5
    assert r["best_fixed_val"] == "none"


# --------------------------------------------------------------------------- #
# 组装：误差与特征的下标对齐 / 截断
# --------------------------------------------------------------------------- #
def test_build_groups_aligns_and_truncates(tmp_path):
    n_val, n_test, nf = 40, 60, 5
    win = tmp_path / "winerr"
    for plug, scale in [("none", 1.0), ("revin", 0.9)]:
        cid = f"DLinear__ETTh1__h96__{plug}__s2021"
        save_window_mse_atomic(win, cid, np.full(n_test, scale, dtype=np.float32))
        _save_val_window_mse(win / "val", cid, np.full(n_val, scale, dtype=np.float32))

    # 特征表比误差长（按最小 pred_len 算的），应被截断而不是报错
    rows = []
    for split, extra in [("val", 7), ("test", 11)]:
        n = (n_val if split == "val" else n_test) + extra
        for i in range(n):
            rows.append({"dataset": "ETTh1", "split": split, "window_index": i,
                         **{f"w_f{j}": float(i + j) for j in range(nf)}})
    feats = pd.DataFrame(rows)

    groups, notes = build_groups(win, feats, verbose=False)
    assert len(groups) == 1, notes
    g = groups[0]
    assert g.arms == ["none", "revin"]
    assert g.e_val.shape == (2, n_val) and g.e_test.shape == (2, n_test)
    assert g.x_val.shape == (n_val, nf) and g.x_test.shape == (n_test, nf)
    # 截断必须从头开始，保证第 i 行特征 <-> 第 i 个误差
    assert g.x_test[0, 0] == 0.0 and g.x_test[-1, 0] == float(n_test - 1)


def test_build_groups_skips_group_without_none_arm(tmp_path):
    win = tmp_path / "winerr"
    for plug in ("revin", "fredf"):
        cid = f"DLinear__ETTh1__h96__{plug}__s2021"
        save_window_mse_atomic(win, cid, np.ones(10, dtype=np.float32))
        _save_val_window_mse(win / "val", cid, np.ones(10, dtype=np.float32))
    feats = pd.DataFrame([{"dataset": "ETTh1", "split": s, "window_index": i, "w_a": float(i)}
                          for s in ("val", "test") for i in range(10)])
    groups, notes = build_groups(win, feats, verbose=False)
    assert groups == [] and any("none" in m for m in notes)


def test_build_groups_skips_when_features_shorter_than_errors(tmp_path):
    win = tmp_path / "winerr"
    for plug in ("none", "revin"):
        cid = f"DLinear__ETTh1__h96__{plug}__s2021"
        save_window_mse_atomic(win, cid, np.ones(50, dtype=np.float32))
        _save_val_window_mse(win / "val", cid, np.ones(50, dtype=np.float32))
    feats = pd.DataFrame([{"dataset": "ETTh1", "split": s, "window_index": i, "w_a": float(i)}
                          for s in ("val", "test") for i in range(20)])
    groups, notes = build_groups(win, feats, verbose=False)
    assert groups == [] and any("特征窗口数" in m for m in notes)


def test_build_groups_rejects_strided_feature_table(tmp_path):
    """stride>1 抽稀的特征表行数可能仍 >= 误差数，旧的位置式截断会静默配错窗口。

    现在按 window_index 取值对齐，缺 0..n-1 中任一下标必须显式跳过。
    """
    win = tmp_path / "winerr"
    n_err = 20
    for plug in ("none", "revin"):
        cid = f"DLinear__ETTh1__h96__{plug}__s2021"
        save_window_mse_atomic(win, cid, np.ones(n_err, dtype=np.float32))
        _save_val_window_mse(win / "val", cid, np.ones(n_err, dtype=np.float32))
    # stride=2 抽稀：window_index = 0,2,4,...,58 —— 共 30 行 > 20 个误差，
    # 行数检查放行，但第 j 行代表窗口 2j，绝不能配到第 j 个误差上。
    feats = pd.DataFrame([{"dataset": "ETTh1", "split": s, "window_index": i, "w_a": float(i)}
                          for s in ("val", "test") for i in range(0, 60, 2)])
    assert len(feats[feats["split"] == "val"]) > n_err, "前提：行数足以骗过行数检查"
    groups, notes = build_groups(win, feats, verbose=False)
    assert groups == [], "抽稀特征表必须被拒绝，而不是静默错位"
    assert any("window_index" in m for m in notes)


def test_build_groups_aligns_by_window_index_not_row_order(tmp_path):
    """特征表行序被打乱时，仍必须按 window_index 正确对齐。"""
    win = tmp_path / "winerr"
    n = 12
    for plug in ("none", "revin"):
        cid = f"DLinear__ETTh1__h96__{plug}__s2021"
        save_window_mse_atomic(win, cid, np.ones(n, dtype=np.float32))
        _save_val_window_mse(win / "val", cid, np.ones(n, dtype=np.float32))
    rows = [{"dataset": "ETTh1", "split": s, "window_index": i, "w_a": float(i)}
            for s in ("val", "test") for i in range(n)]
    shuffled = pd.DataFrame(rows).sample(frac=1.0, random_state=0).reset_index(drop=True)
    groups, _ = build_groups(win, shuffled, verbose=False)
    assert len(groups) == 1
    g = groups[0]
    # w_a == window_index，所以正确对齐后第 i 行必须等于 i
    assert np.array_equal(g.x_test[:, 0], np.arange(n, dtype=float))
    assert np.array_equal(g.x_val[:, 0], np.arange(n, dtype=float))


# --------------------------------------------------------------------------- #
# 汇总与结论文案
# --------------------------------------------------------------------------- #
def test_summarize_and_verdict_branches():
    def mk(gain_bf, oracle, n=12):
        return pd.DataFrame({
            "gain_gate_vs_none": np.linspace(0, 2, n),
            "gain_bestfixed_vs_none": np.linspace(0, 1, n),
            "gain_gate_vs_best_fixed": np.full(n, gain_bf),
            "gain_oracle_vs_none": np.full(n, oracle),
            "switch_rate": np.full(n, 0.3),
            "n_test_windows": np.full(n, 100),
            "wilcoxon_p_gate_vs_best_fixed": np.full(n, 0.01),
            "oracle_realized_pct": np.full(n, 50.0),
            "gate_acc": np.full(n, 0.6),
            "majority_acc": np.full(n, 0.5),
        })

    assert "有信号" in verdict(summarize(mk(2.0, 10.0), "inpool"))
    assert "死路" in verdict(summarize(mk(0.1, 0.3), "inpool"))
    assert "学不到" in verdict(summarize(mk(-1.0, 10.0), "inpool"))
    assert "弱信号" in verdict(summarize(mk(0.2, 10.0), "inpool"))
    assert "无数据" in verdict({"n_groups": 0})
    # 组数不足时必须拒绝下结论：1 个组「100% 为正」是必然事件，不是证据
    assert "样本太少" in verdict(summarize(mk(3.0, 20.0, n=1), "inpool"))
    # 增益够大但几乎没有一组逐窗口显著 -> 只能算弱信号
    weak = summarize(mk(2.0, 10.0), "inpool")
    weak["groups_significant_p05"] = 0
    assert "弱信号" in verdict(weak)

    s = summarize(mk(2.0, 10.0), "inpool")
    assert s["n_groups"] == 12 and s["groups_gate_beats_best_fixed_pct"] == 100.0
    assert s["n_test_windows_total"] == 1200


# --------------------------------------------------------------------------- #
# 特征与误差同源性：真实窗口特征拿去喂门控不应崩
# --------------------------------------------------------------------------- #
def test_real_window_features_feed_gate_end_to_end():
    rng = np.random.default_rng(3)
    L, C, n = 96, 3, 300
    xs, es_none, es_arm = [], [], []
    for i in range(n):
        noisy = i % 2 == 0
        W = (rng.standard_normal((L, C)) if noisy
             else np.stack([np.sin(np.arange(L) * 2 * np.pi / 24 + p) for p in range(C)], 1))
        xs.append(window_features(W))
        es_none.append(1.0)
        es_arm.append(0.5 if noisy else 1.5)     # 插件只在"噪声窗口"上有用
    X = pd.DataFrame(xs).to_numpy(np.float64)
    e = np.stack([np.array(es_none), np.array(es_arm)])
    h = n // 2
    g = _group(X[:h], e[:, :h], X[h:], e[:, h:])
    g.feat_names = list(pd.DataFrame(xs).columns)
    r = evaluate_group(g)
    assert r["gate_acc"] > 0.8
    assert r["gain_gate_vs_best_fixed"] > 5.0


@pytest.mark.parametrize("mode", ["reg", "clf"])
def test_single_class_training_labels_do_not_crash(mode):
    """训练段只有一个臂赢过时（clf 退化成常数门控）不能抛异常。"""
    rng = np.random.default_rng(4)
    n = 200
    x = rng.standard_normal((n, 3))
    e = np.stack([np.ones(n), np.full(n, 2.0)])
    h = n // 2
    r = evaluate_group(_group(x[:h], e[:, :h], x[h:], e[:, h:]), mode=mode)
    assert np.isfinite(r["mse_gate"])


# --------------------------------------------------------------------------- #
# 证伪对照与特征组消融：这两个东西的价值全在"它能不能识破假结果"
# --------------------------------------------------------------------------- #
def _group_with_signal(n: int = 3000, signal: bool = True, seed: int = 0):
    """构造一个组：signal=True 时 f0 决定哪个臂更好；False 时臂的好坏与特征无关。"""
    from src.selector_window import Group
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 6))
    base = 1.0 + 0.05 * rng.standard_normal(n)
    driver = x[:, 0] > 0 if signal else rng.standard_normal(n) > 0
    e_arm = np.where(driver, base * 0.5, base * 2.0)
    e = np.stack([base, e_arm])
    half = n // 2
    return Group("BB", "DS", 96, 2021, ["none", "arm1"],
                 e_val=e[:, :half], e_test=e[:, half:],
                 x_val=x[:half], x_test=x[half:],
                 feat_names=["w_pc_a", "w_ns_a", "w_fr_a", "w_sd_a", "w_ch_a", "w_pc_b"])


def test_controls_confirm_real_signal():
    """有真信号时：门控必须同时赢过『同频随机换臂』和『打乱特征训练的门控』。"""
    from src.selector_window import controls_verdict, evaluate_controls
    r = evaluate_controls(_group_with_signal(signal=True), n_repeat=3)
    assert r["gate_minus_random"] > 5.0, "真信号下门控应显著优于同频随机换臂"
    assert r["gate_minus_shuffled"] > 5.0, "真信号下门控应显著优于打乱特征的门控"
    cs = {"mean_gate_minus_random": r["gate_minus_random"],
          "mean_gate_minus_shuffled": r["gate_minus_shuffled"],
          "mean_gain_shuffled_vs_best_fixed": r["gain_shuffled_vs_best_fixed"],
          "groups_gate_beats_random_pct": 100.0, "groups_gate_beats_shuffled_pct": 100.0}
    assert "✅" in controls_verdict(cs)


def test_controls_expose_fake_signal():
    """无信号时：门控不该显著赢过对照，verdict 必须拒绝下结论。"""
    from src.selector_window import controls_verdict, evaluate_controls
    r = evaluate_controls(_group_with_signal(signal=False), n_repeat=3)
    assert r["gate_minus_random"] < 3.0
    cs = {"mean_gate_minus_random": r["gate_minus_random"],
          "mean_gate_minus_shuffled": r["gate_minus_shuffled"],
          "mean_gain_shuffled_vs_best_fixed": 0.0,
          "groups_gate_beats_random_pct": 50.0, "groups_gate_beats_shuffled_pct": 50.0}
    assert "✅" not in controls_verdict(cs)


def test_controls_verdict_flags_leaky_shuffled_baseline():
    """打乱特征也能赢最佳固定臂时，必须报警而不是庆祝。"""
    from src.selector_window import controls_verdict
    v = controls_verdict({"mean_gate_minus_random": 0.1, "mean_gate_minus_shuffled": 0.1,
                          "mean_gain_shuffled_vs_best_fixed": 2.0,
                          "groups_gate_beats_random_pct": 55.0,
                          "groups_gate_beats_shuffled_pct": 55.0})
    assert "⚠️" in v and "站不住" in v


def test_ablation_points_at_the_group_that_carries_the_signal():
    """信号只藏在 w_pc_a 里，所以去掉 w_pc 组应该掉得最多。"""
    from src.selector_window import FEATURE_GROUPS, ablate_feature_groups
    g = _group_with_signal(signal=True)
    r = ablate_feature_groups(g)
    assert r["gain_full"] > 5.0
    deltas = {k: r[f"delta_drop_{k}"] for k in FEATURE_GROUPS if f"delta_drop_{k}" in r}
    assert len(deltas) == 5
    assert min(deltas, key=deltas.get) == "w_pc", f"应指向 w_pc，实际 {deltas}"
    # 其余组是纯噪声，去掉它们不该造成大幅下降
    others = [v for k, v in deltas.items() if k != "w_pc"]
    assert min(others) > deltas["w_pc"] + 2.0


def test_ablation_skips_absent_feature_groups():
    """特征表里没有某组前缀时，不应生成该组的列（也不能崩）。"""
    from src.selector_window import Group, ablate_feature_groups
    rng = np.random.default_rng(1)
    n = 600
    x = rng.standard_normal((n, 2))
    e = np.stack([np.ones(n), np.where(x[:, 0] > 0, 0.5, 2.0)])
    g = Group("BB", "DS", 96, 2021, ["none", "arm1"], e[:, :300], e[:, 300:],
              x[:300], x[300:], ["w_pc_a", "w_pc_b"])
    r = ablate_feature_groups(g)
    assert "delta_drop_w_ns" not in r and "gain_full" in r


def test_stratify_by_val_size_separates_learnable_groups():
    """分层必须把「没得学」（窗口太少）和「学不到」分开，否则结论会被 ILI 稀释。"""
    from src.selector_window import stratify_by_val_size
    df = pd.DataFrame([
        # 两个小组：门控输给最佳固定臂
        dict(n_val_windows=74, gain_gate_vs_best_fixed=-2.0, gain_gate_vs_none=-1.0,
             gain_oracle_vs_none=9.0, gate_acc=0.2, majority_acc=0.4),
        dict(n_val_windows=120, gain_gate_vs_best_fixed=-3.0, gain_gate_vs_none=-1.5,
             gain_oracle_vs_none=8.0, gate_acc=0.2, majority_acc=0.4),
        # 两个大组：门控赢
        dict(n_val_windows=11425, gain_gate_vs_best_fixed=1.5, gain_gate_vs_none=2.0,
             gain_oracle_vs_none=6.0, gate_acc=0.6, majority_acc=0.4),
        dict(n_val_windows=5175, gain_gate_vs_best_fixed=2.5, gain_gate_vs_none=3.0,
             gain_oracle_vs_none=7.0, gate_acc=0.7, majority_acc=0.4),
    ])
    out = stratify_by_val_size(df)
    small = out["<200（ILI 级，基本不可学）"]
    big = out["≥5000（Weather/ETTm 级）"]
    assert small["n_groups"] == 2 and big["n_groups"] == 2
    assert small["mean_gain_gate_vs_best_fixed"] < 0 < big["mean_gain_gate_vs_best_fixed"]
    assert big["pct_beats_best_fixed"] == 100.0
    # 全量平均会把「赢」抹平到接近 0，这正是需要分层的理由
    assert abs(df["gain_gate_vs_best_fixed"].mean()) < 0.3


def test_stratify_skips_empty_bins():
    from src.selector_window import stratify_by_val_size
    df = pd.DataFrame([dict(n_val_windows=2785, gain_gate_vs_best_fixed=0.5,
                            gain_gate_vs_none=1.0, gain_oracle_vs_none=5.0,
                            gate_acc=0.5, majority_acc=0.4)])
    out = stratify_by_val_size(df)
    assert list(out) == ["1000–5000（ETTh/Electricity 级）"]


def test_clf_gate_survives_a_class_with_a_single_member():
    """回归测试：真实数据里某个臂可能只在 1 个窗口里最优，sklearn 的分层早停会直接报错。

    2026-09-11 的 27 组真实数据上就是这样崩的（least populated class has only 1 member）。
    """
    from src.selector_window import _fit_gate
    rng = np.random.default_rng(0)
    n = 400
    x = rng.standard_normal((n, 4))
    e = np.stack([np.full(n, 1.0), np.full(n, 1.2), np.full(n, 1.3)])
    e[1, :199] = 0.5          # arm1 赢 199 次
    e[2, 199] = 0.1           # arm2 只赢 1 次 -> 分层划分的雷
    for mode in ("clf", "reg"):
        predict, _ = _fit_gate(x, e, ["none", "arm1", "arm2"], 0, mode, seed=1)
        sel = predict(x)
        assert sel.shape == (n,) and set(np.unique(sel)) <= {0, 1, 2}


def test_gate_works_with_very_few_windows():
    """ILI 只有 74 个验证窗口：不能因为留不出 20% 验证集就崩。"""
    from src.selector_window import _fit_gate
    rng = np.random.default_rng(0)
    n = 74
    x = rng.standard_normal((n, 5))
    e = np.stack([np.ones(n), np.where(x[:, 0] > 0, 0.5, 2.0)])
    for mode in ("clf", "reg"):
        predict, _ = _fit_gate(x, e, ["none", "arm1"], 0, mode, seed=1)
        assert predict(x).shape == (n,)


# --------------------------------------------------------------------------- #
# 在线（延迟误差反馈）选择器：这条路的合法性全靠"延迟恰好是 H 个窗口"
# --------------------------------------------------------------------------- #
def test_sliding_mean_is_causal_and_matches_bruteforce():
    from src.selector_window import _sliding_mean
    rng = np.random.default_rng(0)
    e = rng.random((3, 50))
    for w in (1, 5, 200):
        got = _sliding_mean(e, w)
        for k in range(3):
            for t in range(50):
                exp = e[k, max(0, t - w + 1): t + 1].mean()
                assert abs(got[k, t] - exp) < 1e-12, (k, t, w)


def _regime_group(n: int = 2000, pred_len: int = 4, seed: int = 0):
    """构造分段：前半段 arm1 更好，后半段 none 更好。延迟反馈应该能追上这种慢变。"""
    from src.selector_window import Group
    rng = np.random.default_rng(seed)
    base = 1.0 + 0.02 * rng.standard_normal(n)
    good_first = np.arange(n) < n // 2
    e_arm = np.where(good_first, base * 0.5, base * 1.8)
    e = np.stack([base, e_arm])
    x = rng.standard_normal((n, 3))
    return Group("BB", "DS", pred_len, 2021, ["none", "arm1"],
                 e_val=e[:, :10], e_test=e, x_val=x[:10], x_test=x)


def test_online_selector_tracks_regime_switch():
    from src.selector_window import evaluate_online
    r = evaluate_online(_regime_group(), window=100)
    assert r["gain_online_vs_best_fixed"] > 5.0, "分段切换下在线选择应明显优于最佳固定臂"
    assert 0.0 < r["oracle_realized_pct"] <= 100.0
    assert r["delay_windows"] == 4


def test_online_delay_is_exactly_pred_len_and_costs_something():
    """不延迟版本必须 ≥ 延迟版本；delay_cost_pp 就是这个差值，不能是负数。"""
    from src.selector_window import evaluate_online
    r = evaluate_online(_regime_group(pred_len=200), window=50)
    assert r["delay_windows"] == 200
    assert r["gain_nodelay_vs_best_fixed"] >= r["gain_online_vs_best_fixed"] - 1e-9
    assert r["delay_cost_pp"] >= -1e-9


def test_online_selector_does_not_leak_future():
    """把误差序列在时间轴上反转，泄漏实现会依然"赢"；因果实现不会。

    构造：只有最后 5 个窗口 arm1 才变好。因果 + 延迟 H=50 的选择器不可能提前知道，
    所以相对最佳固定臂的收益必须约等于 0（不能显著为正）。
    """
    from src.selector_window import Group, evaluate_online
    n = 1000
    base = np.ones(n)
    e_arm = np.full(n, 3.0)
    e_arm[-5:] = 0.01
    e = np.stack([base, e_arm])
    x = np.zeros((n, 2))
    g = Group("BB", "DS", 50, 2021, ["none", "arm1"], e[:, :10], e, x[:10], x)
    r = evaluate_online(g, window=100)
    assert r["gain_online_vs_best_fixed"] < 0.5, "延迟因果选择器不该提前吃到末尾的好处"


def test_online_verdict_branches():
    from src.selector_window import online_verdict
    good = {"n_groups": 20, "mean_gain_online_vs_best_fixed": 2.0,
            "groups_online_beats_best_fixed_pct": 80.0, "groups_significant_p05": 15,
            "mean_delay_cost_pp": 0.8}
    assert "✅" in online_verdict(good)
    eaten = {"n_groups": 20, "mean_gain_online_vs_best_fixed": -0.5,
             "groups_online_beats_best_fixed_pct": 30.0, "groups_significant_p05": 2,
             "mean_delay_cost_pp": 3.0}
    assert "延迟吃掉" in online_verdict(eaten) or "被 H 步延迟吃掉" in online_verdict(eaten)
    meh = {"n_groups": 20, "mean_gain_online_vs_best_fixed": 0.1,
           "groups_online_beats_best_fixed_pct": 50.0, "groups_significant_p05": 1,
           "mean_delay_cost_pp": 0.1}
    assert "🟡" in online_verdict(meh)


# --------------------------------------------------------------------------- #
# 延迟 × 反馈窗口扫描：论文里那张图的数据源，必须保证单调性与盈亏平衡点的语义
# --------------------------------------------------------------------------- #
def test_sweep_online_shape_and_delay_semantics():
    from src.selector_window import sweep_online
    g = _regime_group(pred_len=8)
    rows = sweep_online(g, windows=(50, 200), delay_ratios=(0.0, 0.5, 1.0, 2.0))
    assert len(rows) == 8
    d = {(r["feedback_window"], r["delay_ratio"]): r for r in rows}
    assert d[(50, 0.0)]["delay_windows"] == 0
    assert d[(50, 1.0)]["delay_windows"] == 8      # 1×H
    assert d[(50, 2.0)]["delay_windows"] == 16
    assert d[(50, 0.0)]["delay_feasible"] is False  # 零延迟是作弊，不可部署
    assert d[(50, 1.0)]["delay_feasible"] is True
    # 延迟越大不可能越好（同一反馈窗口下应当单调不增，允许 1e-9 数值抖动）
    seq = [d[(200, r)]["gain_vs_best_fixed"] for r in (0.0, 0.5, 1.0, 2.0)]
    assert seq[0] >= max(seq[1:]) - 1e-9


def test_sweep_breakeven_ratio_reports_where_it_dies():
    """构造只有零延迟才赢的情形：盈亏平衡延迟必须是 0。"""
    from src.selector_window import Group, sweep_online, summarize_online_sweep, online_sweep_verdict
    n = 600
    rng = np.random.default_rng(3)
    # 每一步在两臂间随机翻转优劣：只有"当前这一步"的信息才有用，延迟一格就失效
    flip = rng.random(n) < 0.5
    e = np.stack([np.where(flip, 0.5, 1.5), np.where(flip, 1.5, 0.5)])
    x = np.zeros((n, 2))
    g = Group("BB", "DS", 8, 2021, ["none", "arm1"], e[:, :10], e, x[:10], x)
    df = pd.DataFrame(sweep_online(g, windows=(50,), delay_ratios=(0.0, 1.0, 2.0)))
    s = summarize_online_sweep(df, ref_window=50)
    assert s["breakeven_delay_ratio"] == 0.0
    assert "❌" in online_sweep_verdict(s) or "⚠️" in online_sweep_verdict(s)


def test_sweep_summary_falls_back_when_ref_window_absent():
    from src.selector_window import sweep_online, summarize_online_sweep
    df = pd.DataFrame(sweep_online(_regime_group(pred_len=4), windows=(37,),
                                   delay_ratios=(0.0, 1.0)))
    s = summarize_online_sweep(df, ref_window=200)   # 200 不在扫描里
    assert s["ref_window"] == 37
    assert set(s["gain_by_delay_ratio"]) == {"0.0", "1.0"}


def test_sweep_verdict_accepts_feasible_win():
    from src.selector_window import online_sweep_verdict
    v = online_sweep_verdict({"breakeven_delay_ratio": 2.0, "feasible_gain_vs_best_fixed": 3.1,
                              "cheat_gain_vs_best_fixed": 4.0, "best_feedback_window": 200})
    assert "✅" in v and "方法主线" in v


def test_sweep_alignment_removes_warmup_artifact():
    """暖机期假象：延迟 d 的前 d 个窗口反馈还没到，必然等于 best_fixed。

    构造一个**确定性的"反馈有害"**场景：arm1 的误差按周期 400 在 0.2 / 4.0 之间切换，
    延迟 200 和 600 都刚好落在反相位上，于是选择器每次都在 arm1 变差的时刻选中它。
    真实收益应当是大负数，且两个延迟档几乎一样。
    但不对齐时，600 的暖机期占 30%、200 只占 10%，被基线稀释得更多，
    于是"延迟越大越好"——这就是第一版扫描里 2H 优于 1H 的全部原因。
    """
    from src.selector_window import Group, sweep_online
    n, per = 2000, 400
    t = np.arange(n)
    e = np.stack([np.ones(n), np.where((t // (per // 2)) % 2 == 0, 0.2, 4.0)])
    e_val = np.stack([np.full(10, 1.0), np.full(10, 1.05)])   # 让 best_fixed = none
    x = np.zeros((n, 2))
    g = Group("BB", "DS", 200, 2021, ["none", "arm1"], e_val, e, np.zeros((10, 2)), x)

    def curve(align: bool) -> list[float]:
        rows = sweep_online(g, windows=(50,), delay_ratios=(1.0, 3.0), align=align)
        return [r["gain_vs_best_fixed"] for r in sorted(rows, key=lambda r: r["delay_ratio"])]

    raw, ali = curve(False), curve(True)
    assert all(v < -30 for v in ali), f"反馈反相位时应当是大负数，实际 {ali}"
    assert raw[1] > raw[0] + 10, f"未对齐时应出现『延迟越大越好』的假象，实际 {raw}"
    # 对齐后：假单调消失（更长的延迟不再显得更好），且两档差距至少收窄一半
    assert ali[1] <= ali[0], f"对齐后更长的延迟不该反而更好，实际 {ali}"
    assert abs(ali[1] - ali[0]) < abs(raw[1] - raw[0]) / 2, f"对齐后差距应显著收窄，{raw} -> {ali}"


def test_sweep_marks_insufficient_when_test_too_short():
    """ILI + h720 这种格子：2×H 已经超过测试窗口数，必须置 NaN 而不是硬算。"""
    from src.selector_window import Group, sweep_online, summarize_online_sweep
    n = 80
    rng = np.random.default_rng(1)
    e = 1.0 + 0.1 * rng.random((2, n))
    x = np.zeros((n, 2))
    g = Group("BB", "ILI", 60, 2021, ["none", "arm1"], e[:, :10], e, x[:10], x)
    rows = sweep_online(g, windows=(50,), delay_ratios=(0.0, 1.0, 2.0))
    by = {r["delay_ratio"]: r for r in rows}
    assert by[0.0]["eval_start"] == 40 == n // 2      # 退让到 n//2
    assert by[1.0]["insufficient"] is True            # 60 > 40
    assert np.isnan(by[1.0]["gain_vs_best_fixed"])
    s = summarize_online_sweep(pd.DataFrame(rows), ref_window=50)
    assert s["n_valid_by_delay_ratio"]["1.0"] == 0
    assert s["n_valid_by_delay_ratio"]["0.0"] == 1
    assert s["aligned_eval_window"] is True


# --------------------------------------------------------------------------- #
# 收益归因：赢 best_fixed_val 可能只是因为 val 选错了臂，而不是时变自适应。
# 这是这条线能不能写成"自适应"贡献的分水岭，必须有测试钉住。
# --------------------------------------------------------------------------- #
def _mismatch_group(n: int = 2000):
    """val 上 arm1 更好，但 test 上 none 才更好（且 test 内部**没有**时变结构）。

    此时任何时变自适应都不该有收益，在线选择器的全部收益都来自"把臂纠回 none"。
    """
    from src.selector_window import Group
    rng = np.random.default_rng(11)
    e_test = np.stack([1.0 + 0.01 * rng.standard_normal(n),
                       1.4 + 0.01 * rng.standard_normal(n)])
    e_val = np.stack([np.full(50, 1.4), np.full(50, 1.0)])      # val 里 arm1 更好
    return Group("BB", "DS", 4, 2021, ["none", "arm1"],
                 e_val, e_test, np.zeros((50, 2)), np.zeros((n, 2)))


def test_online_gain_attributed_to_arm_mismatch_not_adaptivity():
    from src.selector_window import evaluate_online
    r = evaluate_online(_mismatch_group(), window=100)
    assert r["arm_id_mismatch"] is True
    assert r["best_fixed_val"] == "arm1" and r["best_fixed_test"] == "none"
    # 相对 val 选出的臂大幅获胜……
    assert r["gain_online_vs_best_fixed"] > 20
    # ……但相对 test 上最优的常量臂几乎没有增益：不存在时变自适应收益
    assert abs(r["gain_online_vs_best_fixed_test"]) < 1.0
    assert r["gain_bestfixed_test_vs_val"] > 20


def test_online_verdict_flags_mismatch_attribution():
    from src.selector_window import online_verdict
    s = {"n_groups": 20, "mean_gain_online_vs_best_fixed": 3.0,
         "groups_online_beats_best_fixed_pct": 90.0, "groups_significant_p05": 18,
         "mean_delay_cost_pp": 0.3, "mean_gain_online_vs_best_fixed_test": -0.2,
         "groups_arm_id_mismatch_pct": 70.0}
    v = online_verdict(s)
    assert "选臂失配" in v and "不是窗口级时变自适应" in v


def test_online_verdict_credits_real_adaptivity():
    from src.selector_window import online_verdict
    s = {"n_groups": 20, "mean_gain_online_vs_best_fixed": 3.0,
         "groups_online_beats_best_fixed_pct": 90.0, "groups_significant_p05": 18,
         "mean_delay_cost_pp": 0.3, "mean_gain_online_vs_best_fixed_test": 1.8,
         "groups_arm_id_mismatch_pct": 40.0}
    assert "真正的时变自适应" in online_verdict(s)


def test_regime_group_shows_real_adaptivity_beyond_best_fixed_test():
    """分段切换的构造里，时变自适应是真的：相对 test 最优常量臂也必须赢。"""
    from src.selector_window import evaluate_online
    r = evaluate_online(_regime_group(pred_len=4), window=100)
    assert r["gain_online_vs_best_fixed_test"] > 5.0


def test_sweep_carries_best_fixed_test_reference():
    from src.selector_window import sweep_online
    rows = sweep_online(_mismatch_group(), windows=(200,), delay_ratios=(0.0, 1.0))
    for r in rows:
        assert r["gain_vs_best_fixed"] > r["gain_vs_best_fixed_test"]


# --------------------------------------------------------------------------- #
# oracle 审计：本项目的核心方法论点——oracle 上界对"逐窗口重贴臂标签"不变，
# 因此它不能作为"插件选择可学"的证据；真正该测的是最优臂身份的滞后可预测性。
# --------------------------------------------------------------------------- #
def _pure_noise_group(n: int = 3000, K: int = 4, seed: int = 5):
    """所有臂都是同分布独立噪声：不存在任何"该选谁"的结构，
    但 per-window oracle 依然会显示出一个可观的假上界。"""
    from src.selector_window import Group
    rng = np.random.default_rng(seed)
    e = 1.0 + 0.3 * rng.standard_normal((K, n))
    arms = ["none"] + [f"arm{i}" for i in range(1, K)]
    return Group("BB", "DS", 96, 2021, arms, e[:, :50], e,
                 np.zeros((50, 2)), np.zeros((n, 2)))


def test_oracle_is_invariant_to_per_window_arm_relabelling():
    """核心命题的数值证明：oracle 只依赖每列误差的多重集，与臂身份无关。"""
    from src.selector_window import evaluate_argmin_structure
    r = evaluate_argmin_structure(_pure_noise_group(), n_perm=16)
    assert r["oracle_invariant_under_relabel"] is True
    # 纯噪声下 oracle 依然显示出很大的"上界"——这正是文献里容易被误读的地方
    assert r["oracle_gain_real_pct"] > 10.0


def test_argmin_identity_unpredictable_under_pure_noise():
    from src.selector_window import evaluate_argmin_structure
    r = evaluate_argmin_structure(_pure_noise_group(), n_perm=16)
    # 独立噪声：滞后一致率应当贴着 chance = Σ p_k²
    assert abs(r["excess_persist_lag1"]) < 0.02, r
    assert abs(r["excess_persist_lagH"]) < 0.02, r
    assert abs(r["z_lag1"]) < 3.0
    # 重贴标签后的一致率也应当在 chance 附近
    assert abs(r["persist_lag1_relabelled"] - r["chance_persist"]) < 0.02


def test_argmin_identity_persistent_when_regimes_are_long():
    """长分段结构：lag1 与 lagH 都应显著超出 chance。"""
    from src.selector_window import Group, evaluate_argmin_structure
    n = 4000
    rng = np.random.default_rng(6)
    good = (np.arange(n) // 1000) % 2 == 0          # 每 1000 窗口换一次优劣，远长于 H=96
    e = np.stack([np.where(good, 2.0, 0.5), np.where(good, 0.5, 2.0)])
    e = e + 0.01 * rng.standard_normal((2, n))
    g = Group("BB", "DS", 96, 2021, ["none", "arm1"], e[:, :50], e,
              np.zeros((50, 2)), np.zeros((n, 2)))
    r = evaluate_argmin_structure(g, n_perm=8)
    assert r["excess_persist_lag1"] > 0.4
    assert r["excess_persist_lagH"] > 0.3           # 1000 >> H=96，延迟后依然可预测
    assert r["z_lagH"] > 5


def test_argmin_persistence_dies_when_regimes_shorter_than_horizon():
    """分段长度 < H：短程可预测但合法延迟下不可用——这正是我们真实数据里的形态。"""
    from src.selector_window import Group, evaluate_argmin_structure
    n, H = 6000, 200
    rng = np.random.default_rng(8)
    # 马尔可夫切换而不是固定周期：固定周期会和 lag 产生相位共振（周期整除 H 时
    # 滞后一致率反而是 1），那是构造的假象。真实数据里的机制是随机切换，
    # 自相关按 (1-2q)^d 衰减，q=1/20 时到 d=200 已经彻底衰减。
    good = np.empty(n, dtype=bool)
    good[0] = True
    switch = rng.random(n) < 1 / 20
    for t in range(1, n):
        good[t] = ~good[t - 1] if switch[t] else good[t - 1]
    e = np.stack([np.where(good, 2.0, 0.5), np.where(good, 0.5, 2.0)])
    e = e + 0.01 * rng.standard_normal((2, n))
    g = Group("BB", "DS", H, 2021, ["none", "arm1"], e[:, :50], e,
              np.zeros((50, 2)), np.zeros((n, 2)))
    r = evaluate_argmin_structure(g, n_perm=8)
    assert r["excess_persist_lag1"] > 0.3, r        # 相邻窗口高度一致
    assert abs(r["excess_persist_lagH"]) < 0.05, r  # 延迟 200（=10 个周期）后完全失效


def test_argmin_structure_verdict_branches():
    from src.selector_window import argmin_structure_verdict
    base = {"n_groups": 54, "mean_oracle_gain_real_pct": 11.7, "mean_chance_persist": 0.31}
    dead = {**base, "mean_excess_persist_lag1": 0.01,
            "mean_excess_persist_lagH": 0.00, "groups_z_gt2_lagH": 2}
    v = argmin_structure_verdict(dead)
    assert "没有东西可学" in v and "被文献混为一谈" in v
    short = {**base, "mean_excess_persist_lag1": 0.20,
             "mean_excess_persist_lagH": 0.01, "groups_z_gt2_lagH": 5}
    assert "极短程" in argmin_structure_verdict(short)
    alive = {**base, "mean_excess_persist_lag1": 0.30,
             "mean_excess_persist_lagH": 0.22, "groups_z_gt2_lagH": 40}
    assert "✅" in argmin_structure_verdict(alive)


def test_argmin_structure_summary_keys_complete():
    from src.selector_window import evaluate_argmin_structure, summarize_argmin_structure
    df = pd.DataFrame([evaluate_argmin_structure(_pure_noise_group(n=600), n_perm=4)])
    s = summarize_argmin_structure(df)
    for name in ("lag1", "lagH_4", "lagH_2", "lagH", "lag2H"):
        assert f"mean_persist_{name}" in s and f"mean_excess_persist_{name}" in s
    assert s["oracle_invariant_under_relabel_all"] is True


def test_oracle_null_reproducible_across_processes():
    """结果不能依赖 PYTHONHASHSEED：报告里的数字必须能复现。"""
    import json as _json
    import subprocess
    import sys
    code = (
        "import numpy as np, json;"
        "from src.selector_window import Group, evaluate_argmin_structure;"
        "rng=np.random.default_rng(5);"
        "e=1.0+0.3*rng.standard_normal((4,600));"
        "g=Group('BB','DS',96,2021,['none','a','b','c'],e[:,:50],e,"
        "np.zeros((50,2)),np.zeros((600,2)));"
        "print(json.dumps(evaluate_argmin_structure(g,n_perm=8)['persist_lag1_relabelled']))"
    )
    outs = []
    for hs in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": hs}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, cwd=str(REPO))
        assert r.returncode == 0, r.stderr[-2000:]
        outs.append(_json.loads(r.stdout.strip().splitlines()[-1]))
    assert outs[0] == outs[1], outs


def test_half_life_measures_correlation_length_of_best_arm_identity():
    """半衰期是论文的核心数字：必须能在已知相关长度的构造上还原出正确量级。"""
    from src.selector_window import Group, evaluate_argmin_structure
    n = 20000
    rng = np.random.default_rng(21)
    # 马尔可夫切换，切换概率 q=1/64 -> 超出量按 (1-2q)^L 衰减，半衰期 ≈ ln2/(2q) ≈ 22
    good = np.empty(n, dtype=bool)
    good[0] = True
    sw = rng.random(n) < 1 / 64
    for t in range(1, n):
        good[t] = ~good[t - 1] if sw[t] else good[t - 1]
    e = np.stack([np.where(good, 2.0, 0.5), np.where(good, 0.5, 2.0)])
    e = e + 0.01 * rng.standard_normal((2, n))
    g = Group("BB", "DS", 720, 2021, ["none", "arm1"], e[:, :50], e,
              np.zeros((50, 2)), np.zeros((n, 2)))
    r = evaluate_argmin_structure(g, n_perm=4)
    assert 10 < r["persist_half_life_windows"] < 60, r["persist_half_life_windows"]
    assert r["half_life_over_H"] < 0.1          # 远短于 H=720
    # 衰减曲线必须单调不增（允许噪声抖动）
    curve = [r[f"excess_abs{L}"] for L in (1, 2, 4, 8, 16, 32, 64, 128)]
    assert curve[0] > curve[-1] and curve[-1] < 0.05


def test_half_life_is_nan_when_no_short_range_signal():
    from src.selector_window import evaluate_argmin_structure
    r = evaluate_argmin_structure(_pure_noise_group(n=2000), n_perm=4)
    hl = r["persist_half_life_windows"]
    assert hl != hl or hl <= 2, hl      # 纯噪声：要么无法定义，要么立刻衰减


def test_verdict_quotes_half_life_ratio():
    from src.selector_window import argmin_structure_verdict
    s = {"n_groups": 57, "mean_oracle_gain_real_pct": 11.9, "mean_chance_persist": 0.39,
         "mean_excess_persist_lag1": 0.478, "mean_excess_persist_lagH": -0.016,
         "groups_z_gt2_lagH": 13, "median_persist_half_life_windows": 8.3,
         "mean_half_life_over_H": 0.031}
    v = argmin_structure_verdict(s)
    assert "半衰期" in v and "8.3" in v and "3.1%" in v
    assert "一到两个数量级" in v


# --------------------------------------------------------------------------- #
# 成对 seed 审计：headroom 里有多少是训练随机性
# --------------------------------------------------------------------------- #
def _seed_pair(e_a, e_b, arms=("none", "arm1"), pred_len=96, bb="BB", ds="DS"):
    """构造两个只差 seed 的组。特征在 seed 审计里用不到，给个占位。"""
    n = e_a.shape[1]
    x = np.zeros((n, 2))
    ga = Group(bb, ds, pred_len, 2021, list(arms), e_a, e_a, x, x, ["w_f0", "w_f1"])
    gb = Group(bb, ds, pred_len, 2022, list(arms), e_b, e_b, x, x, ["w_f0", "w_f1"])
    return ga, gb


def test_pair_seed_groups_pairs_only_same_cell_different_seed():
    e = np.abs(np.random.default_rng(0).normal(1, 0.2, (2, 60)))
    ga, gb = _seed_pair(e, e)
    other = Group("BB", "OTHER", 96, 2021, ["none", "arm1"], e, e,
                  np.zeros((60, 2)), np.zeros((60, 2)), ["w_f0", "w_f1"])
    pairs = pair_seed_groups([gb, ga, other])
    assert len(pairs) == 1
    assert (pairs[0][0].seed, pairs[0][1].seed) == (2021, 2022)   # 按 seed 升序


def test_pair_seed_groups_ignores_single_seed_cells():
    e = np.abs(np.random.default_rng(1).normal(1, 0.2, (2, 60)))
    ga, _ = _seed_pair(e, e)
    assert pair_seed_groups([ga]) == []


def test_align_pair_matches_arms_by_name_not_index():
    """两个 seed 的 arms 顺序不同时，必须按名字对齐，否则一致率是假象。"""
    rng = np.random.default_rng(2)
    n = 50
    a_none, a_x, a_y = (np.abs(rng.normal(1, .1, n)) for _ in range(3))
    ga = Group("BB", "DS", 96, 2021, ["none", "xx", "yy"],
               np.stack([a_none, a_x, a_y]), np.stack([a_none, a_x, a_y]),
               np.zeros((n, 2)), np.zeros((n, 2)), ["w_f0", "w_f1"])
    # seed B 只有 none 与 yy，且顺序里 yy 在前
    gb = Group("BB", "DS", 96, 2022, ["none", "yy"],
               np.stack([a_none, a_y]), np.stack([a_none, a_y]),
               np.zeros((n, 2)), np.zeros((n, 2)), ["w_f0", "w_f1"])
    ea, eb, arms = _align_pair(ga, gb)
    assert arms == ["none", "yy"]
    assert ea.shape == eb.shape == (2, n)
    # 同一个臂名对应同一条误差曲线
    np.testing.assert_allclose(ea[1], a_y)
    np.testing.assert_allclose(eb[1], a_y)


def test_seed_audit_reports_noise_when_argmin_is_pure_training_noise():
    """核心用例：两个臂**真实水平完全相同**，逐窗口谁赢纯靠训练噪声。

    此时同 seed oracle 仍有可观 headroom（min 两个噪声），但跨 seed oracle 应该
    赚不到钱——这正是「oracle headroom 不等于可学信号」的经验版证据。
    """
    rng = np.random.default_rng(7)
    n = 4000
    diff = np.abs(rng.normal(1.0, 0.25, (2, n)))     # seed A 的两臂
    diff_b = np.abs(rng.normal(1.0, 0.25, (2, n)))   # seed B 独立重训
    ga, gb = _seed_pair(diff, diff_b)
    r = evaluate_seed_pair(ga, gb)
    assert r["self_oracle_gain_pct"] > 3.0                     # 同 seed 上界不小
    assert abs(r["argmin_agree"] - r["chance_agree"]) < 0.05    # 一致率≈chance
    assert r["xseed_oracle_gain_pct"] < 0.5                    # 跨 seed 赚不到
    assert r["noise_frac"] > 0.8                               # headroom 基本是噪声


def test_seed_audit_detects_reproducible_signal():
    """对照用例：存在跨 seed 稳定的窗口级结构时，跨 seed oracle 必须拿到大部分 headroom。"""
    rng = np.random.default_rng(8)
    n = 4000
    who = rng.integers(0, 2, n)            # 每个窗口真正该用哪个臂（与 seed 无关）

    def _mk():
        e = np.abs(rng.normal(1.0, 0.05, (2, n)))
        e[who, np.arange(n)] *= 0.5        # 该赢的臂真的好一大截
        return e
    ga, gb = _seed_pair(_mk(), _mk())
    r = evaluate_seed_pair(ga, gb)
    assert r["argmin_agree"] > 0.9
    assert r["excess_agree"] > 0.3
    assert r["noise_frac"] < 0.2
    assert r["xseed_oracle_gain_pct"] > 10.0


def test_seed_audit_baseline_is_test_best_fixed_not_val():
    """baseline 必须是测试段最佳固定臂：一个臂全程更好时，逐窗口决策不该有收益。"""
    rng = np.random.default_rng(9)
    n = 800
    e = np.stack([np.abs(rng.normal(1.0, 0.02, n)),
                  np.abs(rng.normal(0.5, 0.02, n))])    # arm1 恒优
    e2 = np.stack([np.abs(rng.normal(1.0, 0.02, n)),
                   np.abs(rng.normal(0.5, 0.02, n))])
    r = evaluate_seed_pair(*_seed_pair(e, e2))
    assert r["best_fixed_arm_a"] == r["best_fixed_arm_b"] == "arm1"
    assert r["self_oracle_gain_pct"] < 1.0        # 相对最佳固定臂几乎没有空间
    assert r["xseed_oracle_gain_pct"] < 1.0


def test_summarize_and_verdict_seed_audit_flags_noise():
    rng = np.random.default_rng(10)
    rows = []
    for i in range(6):
        n = 3000
        ga, gb = _seed_pair(np.abs(rng.normal(1, .25, (2, n))),
                            np.abs(rng.normal(1, .25, (2, n))))
        rows.append(evaluate_seed_pair(ga, gb))
    s = summarize_seed_audit(pd.DataFrame(rows))
    assert s["n_pairs"] == 6 and s["seeds"] == [2021, 2022]
    assert s["noise_frac_of_headroom"] > 0.8
    v = seed_audit_verdict(s)
    assert v.startswith("❌") and "训练随机性" in v


def test_seed_audit_flags_low_power_when_two_seeds_give_identical_models():
    """近凸骨干（DLinear）两个 seed 收敛到同一个解时，「可复现」是平凡的，必须自己招认。"""
    rng = np.random.default_rng(11)
    n = 2000
    base = np.abs(rng.normal(1.0, 0.3, (2, n)))
    rows = [evaluate_seed_pair(*_seed_pair(base, base * 1.0001)) for _ in range(4)]
    s = summarize_seed_audit(pd.DataFrame(rows))
    assert s["mean_same_arm_err_corr"] > 0.99
    v = seed_audit_verdict(s)
    assert v.startswith("✅") and "区分力有限" in v


def test_seed_audit_calls_signal_real_when_models_differ_but_argmin_agrees():
    """模型确实变了（同臂误差相关不高），但 argmin 身份仍稳定 -> 「信号是真的，但来得太晚」。"""
    rng = np.random.default_rng(12)
    n = 3000
    who = rng.integers(0, 2, n)

    def _mk():
        # 每个臂加一个较大的、与窗口无关的 seed 噪声，压低同臂误差的跨 seed 相关
        e = np.abs(rng.normal(1.0, 0.6, (2, n)))
        e[who, np.arange(n)] *= 0.25
        return e
    rows = [evaluate_seed_pair(*_seed_pair(_mk(), _mk())) for _ in range(4)]
    s = summarize_seed_audit(pd.DataFrame(rows))
    assert s["mean_same_arm_err_corr"] < 0.99
    assert s["noise_frac_of_headroom"] < 0.5
    v = seed_audit_verdict(s)
    assert v.startswith("✅") and "信号是真的，但它来得太晚" in v


# --------------------------------------------------------------------------- #
# 分层审计：论文核心论点「相关长度 << 预测视界」必须在每一层都成立
# --------------------------------------------------------------------------- #
def _audit_frame(n_per: int = 4, hl_scale: float = 1.0, exc_lagH: float = -0.01):
    """造一张 evaluate_argmin_structure 形状的逐组表。"""
    from src.selector_window import LAGS_ABS
    rows = []
    for bb in ("DLinear", "PatchTST"):
        for ds in ("ETTh1", "ILI"):
            for H in (96, 720):
                for k in range(n_per):
                    hl = 8.0 * hl_scale * (1 + 0.1 * k)
                    r = {"key": f"{bb}__{ds}__h{H}__{k}", "backbone": bb, "dataset": ds,
                         "pred_len": H, "n_arms": 4, "n_test_windows": 2000,
                         "win_rate_top": 0.4, "win_rate_none": 0.2, "win_rate_entropy": 0.9,
                         "chance_persist": 0.28, "persist_lag1": 0.9,
                         "excess_persist_lag1": 0.6, "z_lag1": 70.0,
                         "excess_persist_lagH": exc_lagH, "z_lagH": -1.0 + 4 * (exc_lagH > 0.05),
                         "persist_half_life_windows": hl, "half_life_over_H": hl / H,
                         "oracle_invariant_under_relabel": True,
                         "persist_lag1_relabelled": 0.25, "oracle_gain_real_pct": 10.0}
                    for L in LAGS_ABS:
                        r[f"excess_abs{L}"] = 0.6 * 0.5 ** (L / hl)
                    rows.append(r)
    return pd.DataFrame(rows)


def test_stratify_audit_covers_three_cuts_and_counts_add_up():
    from src.selector_window import stratify_audit
    df = _audit_frame()
    ds = stratify_audit(df)
    assert set(ds["stratum_key"]) == {"backbone", "dataset", "pred_len"}
    # 每种切法的组数之和必须等于总组数——分层不能漏组也不能重复计
    for key in ("backbone", "dataset", "pred_len"):
        assert ds.loc[ds["stratum_key"] == key, "n_groups"].sum() == len(df), key
    assert not ds["underpowered"].any()


def test_stratify_audit_flags_underpowered_stratum_instead_of_dropping_it():
    """只有 1~2 组的层要保留并标记：直接丢掉会让『某个数据集是例外』这种信息消失。"""
    from src.selector_window import stratify_audit
    df = _audit_frame()
    df = pd.concat([df, df.iloc[:1].assign(dataset="Weather")], ignore_index=True)
    ds = stratify_audit(df, min_n=3)
    w = ds[(ds["stratum_key"] == "dataset") & (ds["stratum"] == "Weather")]
    assert len(w) == 1 and bool(w["underpowered"].iloc[0]) is True
    assert int(w["n_groups"].iloc[0]) == 1


def test_strata_verdict_confirms_universality_when_every_stratum_is_short():
    from src.selector_window import stratify_audit, summarize_strata, strata_verdict
    s = summarize_strata(stratify_audit(_audit_frame()))
    assert s["strata_with_excess_lagH_above_005"] == 0
    assert s["strata_with_half_life_over_H_below_25pct"] == s["strata_total_used"]
    assert "✅" in strata_verdict(s)


def test_strata_verdict_refuses_to_generalize_when_one_stratum_keeps_signal():
    """有一层在延迟 H 下仍有信号时，必须降级为『要单独讨论』，不能笼统宣称全都学不到。"""
    from src.selector_window import stratify_audit, summarize_strata, strata_verdict
    df = _audit_frame()
    m = df["dataset"] == "ILI"
    df.loc[m, "excess_persist_lagH"] = 0.30
    df.loc[m, "z_lagH"] = 9.0
    s = summarize_strata(stratify_audit(df))
    v = strata_verdict(s)
    assert s["strata_with_excess_lagH_above_005"] >= 1
    assert "⚠️" in v and "ILI" in s["max_excess_persist_lagH_stratum"]
    assert "单独讨论" in v


def test_strata_reports_horizon_dependence_of_half_life():
    """真实数据的关键形态：半衰期绝对值随 H 变长，但『半衰期/H』随 H 变小。
    这两个秩相关的符号就是主图要讲的故事，必须被算出来。"""
    from src.selector_window import stratify_audit, summarize_strata
    rows = []
    base = _audit_frame()
    for H, hl in ((96, 8.0), (192, 10.0), (336, 11.0), (720, 14.0)):
        sub = base[base["pred_len"] == 96].copy()
        sub["pred_len"] = H
        sub["persist_half_life_windows"] = hl
        sub["half_life_over_H"] = hl / H
        rows.append(sub)
    s = summarize_strata(stratify_audit(pd.concat(rows, ignore_index=True)))
    assert s["horizon_hl_spearman"] > 0.9      # 绝对半衰期随 H 上升
    assert s["horizon_ratio_spearman"] < -0.9  # 但比值随 H 下降
    assert s["hl_at_max_H"] > s["hl_at_min_H"]


def test_spearman_matches_known_values():
    from src.selector_window import _spearman
    a = np.array([1.0, 2, 3, 4, 5])
    assert abs(_spearman(a, a) - 1.0) < 1e-12
    assert abs(_spearman(a, -a) + 1.0) < 1e-12
    # 常量输入：分母为 0，必须返回 nan 而不是抛除零错
    v = _spearman(np.array([1.0, 1, 1]), np.array([1.0, 2, 3]))
    assert v != v
    # 样本太少也返回 nan
    assert _spearman(np.array([1.0, 2]), np.array([1.0, 2])) != \
           _spearman(np.array([1.0, 2]), np.array([1.0, 2]))


# --------------------------------------------------------------------------- #
# 2026-09-14 code review 的三个 P1 修复项的回归测试。
# 这三条都是"口径"错误：数字算得出来、看起来也合理，但结算区间/组集合/文案与统计量
# 不一致。所以测试必须用**独立构造、结论可手算**的数据，不能拿实现自己的输出当期望值。
# --------------------------------------------------------------------------- #
def _warmup_heavy_group(n: int = 400, H: int = 200, arm_late: float = 2.0,
                        arm_early: float = 0.5):
    """暖机占比 50% 的组：延迟 H 恰好等于半个测试段，且反馈**反相位**。

    arm1 的误差在 t<H 时是 ``arm_early``、t≥H 时是 ``arm_late``；none 恒为 1.0。
    于是延迟 H 的选择器在评测区间 [H, n) 里读到的全是前半段（arm1 看起来很好）的
    反馈，必然一路选中 arm1，而 arm1 在这段区间里正是最差的。val 上让 none 取胜，
    所以 best_fixed = none、逐点误差恒为 1.0，一切都能手算。
    """
    from src.selector_window import Group
    t = np.arange(n)
    e = np.stack([np.ones(n), np.where(t < H, arm_early, arm_late)])
    e_val = np.stack([np.full(20, 1.0), np.full(20, 1.5)])      # best_fixed = none
    return Group("BB", "DS", H, 2021, ["none", "arm1"],
                 e_val, e, np.zeros((20, 2)), np.zeros((n, 2)))


def test_online_settles_delayed_and_nodelay_on_the_same_interval():
    """P1（2026-09-14）：延迟版与零延迟版必须在**同一段公共区间**上结算。

    构造是全确定性的，期望值可以手算：评测区间 = [200, 400)，
    该区间里 best_fixed(none) 恒为 1.0、延迟选择器被反相位反馈骗得一路选 arm1（恒 2.0），
    所以 mse_online 必须正好是 2.0、相对 best_fixed 是 −100%、换臂率 100%。
    旧实现在完整 [0, 400) 上取均值，前 200 个窗口被强行填成 best_fixed，
    于是 mse_online 被稀释成 1.5、增益只剩 −50%、换臂率 0.5——正是"暖机把指标压向 0"。
    """
    from src.selector_window import evaluate_online
    r = evaluate_online(_warmup_heavy_group(), window=50)
    assert r["eval_start"] == 200 and r["n_eval_windows"] == 200
    assert abs(r["mse_best_fixed"] - 1.0) < 1e-12
    assert abs(r["mse_online"] - 2.0) < 1e-12, "延迟版必须只在 [H, n) 上结算"
    assert r["gain_online_vs_best_fixed"] < -95.0, r["gain_online_vs_best_fixed"]
    # 评测区间里已经没有暖机窗口，所以每一个窗口都真的换了臂
    assert abs(r["switch_rate"] - 1.0) < 1e-12
    assert r["insufficient"] is False
    # 零延迟版在同一区间上只吃到很小的转向代价（窗口 50 的滑动均值 ~16 个窗口才翻转）
    assert -12.0 < r["gain_nodelay_vs_best_fixed"] < 0.0
    assert r["delay_cost_pp"] > 80.0, r["delay_cost_pp"]


def test_online_delay_cost_is_zero_when_eval_interval_is_neutral():
    """P1（2026-09-14）：暖机期里的"免费收益"不得计入 delay_cost_pp。

    构造：前 H 个窗口 arm1 明显更好（0.2 vs 1.0），评测区间 [H, n) 里两臂**完全相同**
    （都是 1.0）。既然公共评测区间里没有任何可赚的差异，延迟版与零延迟版必须一模一样，
    delay_cost_pp 恒等于 0。旧实现把零延迟版在暖机段白捡的 0.2 也平均进去，
    于是 delay_cost_pp ≈ +40pp —— 纯粹是"零延迟版多评了一段延迟版没评的区间"。
    """
    from src.selector_window import Group, evaluate_online
    n, H = 400, 200
    t = np.arange(n)
    e = np.stack([np.ones(n), np.where(t < H, 0.2, 1.0)])
    e_val = np.stack([np.full(20, 1.0), np.full(20, 1.5)])
    g = Group("BB", "DS", H, 2021, ["none", "arm1"], e_val, e,
              np.zeros((20, 2)), np.zeros((n, 2)))
    r = evaluate_online(g, window=50)
    assert r["eval_start"] == 200
    assert abs(r["mse_online"] - 1.0) < 1e-12
    assert abs(r["mse_online_nodelay"] - 1.0) < 1e-12
    assert abs(r["delay_cost_pp"]) < 1e-9, r["delay_cost_pp"]
    assert abs(r["gain_nodelay_vs_best_fixed"]) < 1e-9
    # oracle 也必须在同一区间上算：区间内两臂相同，因此上界为 0
    assert abs(r["mse_oracle"] - 1.0) < 1e-12


def test_online_summary_excludes_groups_whose_delay_eats_half_the_test_split():
    """P1（2026-09-14）：H 已吃掉过半测试窗口的组要标 insufficient 并从汇总里剔除。

    这种组（Exchange/ILI + h720）的评测区间里仍然残留暖机窗口，它的"增益≈0、
    delay_cost 巨大"是稀释产物。汇总必须只用可比的那一组，均值等于它自己的值。
    """
    from src.selector_window import evaluate_online, summarize_online
    ok = evaluate_online(_warmup_heavy_group(n=1000, H=100), window=50)
    bad = evaluate_online(_warmup_heavy_group(n=100, H=80), window=50)
    assert ok["insufficient"] is False
    assert bad["insufficient"] is True and bad["eval_start"] == 50
    # 两组的延迟代价必须差得足够远，否则这条测试对"有没有剔除"没有区分力
    assert abs(ok["delay_cost_pp"] - bad["delay_cost_pp"]) > 1.0
    s = summarize_online(pd.DataFrame([ok, bad]))
    assert (s["n_groups"], s["n_groups_all"], s["n_groups_insufficient"]) == (1, 2, 1)
    assert abs(s["mean_delay_cost_pp"] - ok["delay_cost_pp"]) < 1e-12
    assert abs(s["mean_gain_online_vs_best_fixed"] - ok["gain_online_vs_best_fixed"]) < 1e-12


def _sweep_frame_with_dropouts() -> pd.DataFrame:
    """手写一张扫描表：5 个"短"组全档有效，5 个"长 horizon"组在 1×H 掉档。

    数字是刻意设计的，所有均值可以口算：
      * 短组（公共支撑）：ratio 0.0 → +2.0，0.5 → −1.0，1.0 → −2.0
      * 长组：ratio 0.0 → +6.0，0.5 → +3.0，1.0 → NaN（测试窗口不够）
    naive（跳过 NaN）曲线：+4.0 / +1.0 / −2.0  -> breakeven = 0.5×H（被掉组撑高）
    公共支撑曲线：      +2.0 / −1.0 / −2.0  -> breakeven = 0（真实值）
    """
    gains = {"short": {0.0: 2.0, 0.5: -1.0, 1.0: -2.0},
             "long": {0.0: 6.0, 0.5: 3.0, 1.0: float("nan")}}
    rows = []
    for kind, k in [(k_, i) for k_ in ("short", "long") for i in range(5)]:
        for r, v in gains[kind].items():
            rows.append({"key": f"{kind}{k}", "backbone": "BB", "dataset": "DS",
                         "pred_len": 96 if kind == "short" else 720,
                         "n_test_windows": 2000, "eval_start": 192,
                         "n_eval_windows": 1808, "feedback_window": 200,
                         "delay_ratio": r, "delay_windows": int(r * 96),
                         "delay_feasible": r >= 1.0, "insufficient": v != v,
                         "gain_vs_best_fixed": v, "gain_vs_none": v,
                         "gain_vs_best_fixed_test": v, "gain_oracle_vs_none": 10.0})
    return pd.DataFrame(rows)


def test_sweep_summary_uses_common_support_across_delay_ratios():
    """P1（2026-09-14）：延迟曲线的每个点必须来自同一批组，否则 breakeven 被掉组撑高。

    期望值全部来自上面手写的表，与实现无关：公共支撑只剩 5 个短组，
    ratio=0.5 的均值必须是 −1.0（naive 会因为混进 5 个 +3.0 的长组变成 +1.0），
    因此 breakeven 必须是 0 而不是 0.5×H。
    """
    from src.selector_window import summarize_online_sweep
    s = summarize_online_sweep(_sweep_frame_with_dropouts(), ref_window=200)
    assert (s["n_groups"], s["n_groups_all"]) == (5, 10)
    assert s["n_groups_dropped_no_common_support"] == 5
    assert abs(s["gain_by_delay_ratio"]["0.0"] - 2.0) < 1e-9
    assert abs(s["gain_by_delay_ratio"]["0.5"] + 1.0) < 1e-9, "0.5 档必须只用公共支撑"
    assert abs(s["gain_by_delay_ratio"]["1.0"] + 2.0) < 1e-9
    assert s["breakeven_delay_ratio"] == 0.0, "掉组不能把盈亏平衡延迟撑高到 0.5×H"
    assert abs(s["gain_by_delay_and_window"]["0.5"]["200"] + 1.0) < 1e-9
    # 作弊上界与合法设定也必须在公共支撑上算
    assert abs(s["cheat_gain_vs_best_fixed"] - 2.0) < 1e-9
    assert abs(s["feasible_gain_vs_best_fixed"] + 2.0) < 1e-9
    # 掉组的事实本身仍要留在 summary 里（这是取公共支撑的证据）
    assert s["n_valid_by_delay_ratio"] == {"0.0": 10, "0.5": 10, "1.0": 5}


def test_sweep_summary_is_invariant_to_adding_a_group_that_drops_out():
    """同一性质的端到端版本：往扫描里加一个在长延迟档掉档的组，曲线不该被它改写。

    这是"公共支撑"的定义性质：只在部分延迟档有效的组必须整组退出聚合，
    所以两次汇总的曲线、breakeven 必须逐位相同。旧实现下 ratio=0 的均值会被它改掉。
    """
    from src.selector_window import summarize_online_sweep, sweep_online
    long_g = _regime_group(n=80, pred_len=60)                  # 2×H、1×H 都超过 n//2=40
    short_rows = sweep_online(_regime_group(n=2000, pred_len=8),
                              windows=(50,), delay_ratios=(0.0, 1.0, 2.0))
    long_rows = sweep_online(long_g, windows=(50,), delay_ratios=(0.0, 1.0, 2.0))
    assert any(r["insufficient"] for r in long_rows), "构造失败：长组没有掉档"
    only = summarize_online_sweep(pd.DataFrame(short_rows), ref_window=50)
    both = summarize_online_sweep(pd.DataFrame(short_rows + long_rows), ref_window=50)
    assert both["n_groups"] == only["n_groups"] == 1
    assert both["n_groups_dropped_no_common_support"] == 1
    assert both["gain_by_delay_ratio"] == only["gain_by_delay_ratio"]
    assert both["breakeven_delay_ratio"] == only["breakeven_delay_ratio"]


def _strata_summary(rho_hl: float, rho_ratio: float, hl_lo: float = 4.5,
                    hl_hi: float = 14.1) -> dict:
    """手写一份 summarize_strata 形状的字典，只为检验文案与统计量是否一致。"""
    return {"protocol": "audit_strata", "n_strata": 12, "strata_total_used": 12,
            "strata_with_half_life_over_H_below_25pct": 12,
            "strata_with_excess_lagH_above_005": 0,
            "median_half_life_windows_range": [hl_lo, hl_hi],
            "max_excess_persist_lagH": -0.004,
            "max_excess_persist_lagH_stratum": "dataset=ETTh1",
            "max_median_half_life_over_H": 0.19,
            "max_median_half_life_over_H_stratum": "pred_len=24",
            "horizon_hl_spearman": rho_hl, "horizon_ratio_spearman": rho_ratio,
            "hl_at_min_H": hl_lo, "hl_at_max_H": hl_hi, "H_min": 24.0, "H_max": 720.0}


def test_strata_verdict_does_not_call_half_life_flat_when_it_grows_with_H():
    """P1（2026-09-14）：文案必须与 horizon_hl_spearman 的符号绑定。

    落盘 summary 的真实取值：rho(H, 半衰期) = +0.74，H=24 时 4.5 个窗口、
    H=720 时 14.1 个窗口（长了 3 倍）。旧实现在打印完 +0.74 之后仍然无条件接上
    「相关长度基本不随预测视界一起变长」，同一句话里自相矛盾。
    正确表述：绝对半衰期在增长，但慢于 H，所以「半衰期/H」随 H 下降。
    """
    from src.selector_window import strata_verdict
    v = strata_verdict(_strata_summary(0.74, -0.95))
    assert "+0.74" in v and "-0.95" in v
    assert "基本不随预测视界一起变长" not in v, "半衰期明明随 H 长了 3 倍"
    assert "绝对值随 H 增长" in v and "增长慢于 H 本身" in v
    assert "「半衰期/H」随 H 下降" in v and "H 越大越无望" in v


def test_strata_verdict_allows_flat_wording_only_when_rho_is_near_zero():
    from src.selector_window import strata_verdict
    v = strata_verdict(_strata_summary(0.05, -0.9, hl_lo=8.0, hl_hi=8.2))
    assert "基本不随预测视界一起变长" in v
    assert "绝对值随 H 增长" not in v


def test_strata_verdict_refuses_hopeless_claim_when_ratio_rises_with_H():
    """「H 越大越无望」只能由 horizon_ratio_spearman 为负来支撑。"""
    from src.selector_window import strata_verdict
    v = strata_verdict(_strata_summary(0.9, 0.8))
    assert "「半衰期/H」随 H 下降：H 越大越无望" not in v
    assert "不能宣称 H 越大越无望" in v
    assert "长视界反而相对更有利" in v
    # 秩相关算不出来（层太少 / 取值恒定）时也不许硬下结论
    nan_v = strata_verdict(_strata_summary(float("nan"), float("nan")))
    assert "无法判定" in nan_v and "H 越大越无望" not in nan_v


def test_strata_verdict_wording_matches_end_to_end_horizon_statistics():
    """端到端版：半衰期随 H 增长的逐组表 -> 分层 -> 汇总 -> 文案，三者不能互相打脸。"""
    from src.selector_window import stratify_audit, summarize_strata, strata_verdict
    base = _audit_frame()
    parts = []
    for H, hl in ((96, 8.0), (192, 10.0), (336, 11.0), (720, 14.0)):
        sub = base[base["pred_len"] == 96].copy()
        sub["pred_len"] = H
        sub["persist_half_life_windows"] = hl
        sub["half_life_over_H"] = hl / H
        parts.append(sub)
    s = summarize_strata(stratify_audit(pd.concat(parts, ignore_index=True)))
    assert s["horizon_hl_spearman"] > 0.9 and s["horizon_ratio_spearman"] < -0.9
    v = strata_verdict(s)
    assert "基本不随预测视界一起变长" not in v
    assert "绝对值随 H 增长" in v and "H 越大越无望" in v
