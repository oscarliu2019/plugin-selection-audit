"""`src/selector.py` 自检。

四件必须成立的事：
1. 标签构造正确（含 τ 阈值下「不该挂」的判定）；
2. LODO **真的没有跨数据集泄漏**（每折训练集不含该数据集的任何行）；
3. 在特征确实携带信号的合成场景里，selector 的决策收益要 ≥ 最强固定策略，
   且 regret 严格小于「永远不挂」；
4. （2026-09-14 code review 回归）`best_fixed_lodo` 是**折外**的可部署强基线：
   每折只用训练折的块选固定臂，换掉留出折的误差不得改变选臂结果；
   而 `always_*` 里 mean_mse 最小的那一行只是事后上界，note 必须写明不可部署。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import selector as SEL


# --------------------------------------------------------------------------- #
# 合成场景：插件最优选择完全由一个特征决定
# --------------------------------------------------------------------------- #
def make_case(n_datasets: int = 6, horizons=(96, 336), backbones=("DLinear", "PatchTST"),
              seed: int = 0, noise: float = 0.0005) -> tuple[pd.DataFrame, pd.DataFrame]:
    """构造 (features, results)：ns_mean_drift 高 → revin 最优，低 → fredf 最优，中间 → 不该挂。"""
    rng = np.random.default_rng(seed)
    frows, rrows = [], []
    for i in range(n_datasets):
        drift = (i + 0.5) / n_datasets                     # 0..1 均匀铺开
        for h in horizons:
            frows.append({
                "dataset": f"D{i}", "pred_len": h,
                "ns_mean_drift": drift,
                "fr_topk_energy": 1.0 - drift,
                "pc_perm_entropy": 0.8 + 0.05 * rng.standard_normal(),
                "hz_mean_shift": drift * h / 96.0,
            })
            for bk in backbones:
                base = 0.4 + 0.1 * (h == 336)
                gains = {
                    "none": 0.0,
                    "revin": 0.05 * (drift - 0.5),          # drift>0.5 才有收益
                    "fredf": 0.05 * (0.5 - drift),
                    "san_lite": -0.01,                      # 恒定有害的对照
                }
                for pl, g in gains.items():
                    rrows.append({"dataset": f"D{i}", "pred_len": h, "backbone": bk,
                                  "plugin": pl, "seed": 2021,
                                  "mse": base * (1 - g) + rng.normal(0, noise)})
    return pd.DataFrame(frows), pd.DataFrame(rrows)


# --------------------------------------------------------------------------- #
# 1. 数据构造与标签
# --------------------------------------------------------------------------- #
def test_build_dataset_shapes_and_onehot():
    feats, res = make_case()
    d = SEL.build_dataset(feats, res)
    assert len(d.X) == 6 * 2 * 2 == len(d.y) == len(d.mse)
    assert {"bb_DLinear", "bb_PatchTST"} <= set(d.X.columns)
    assert "dataset" not in d.X.columns and "backbone" not in d.X.columns
    assert "pred_len" in d.X.columns          # horizon 必须作为特征保留
    assert not d.X.isna().any().any()
    assert list(d.blocks.columns) == SEL.BLOCK_KEYS


def test_labels_follow_relative_gain_and_tau():
    feats, res = make_case()
    d_strict = SEL.build_dataset(feats, res, tau=0.02)   # 高阈值 → 更多「不该挂」
    d_loose = SEL.build_dataset(feats, res, tau=0.0)
    assert (d_strict.y == "none").sum() > (d_loose.y == "none").sum()
    assert (d_loose.y == "none").sum() == 0
    # 极端格子：drift 最大的数据集必须选 revin，最小的必须选 fredf
    lab = pd.concat([d_loose.blocks, d_loose.y.rename("label")], axis=1)
    assert set(lab[lab.dataset == "D5"]["label"]) == {"revin"}
    assert set(lab[lab.dataset == "D0"]["label"]) == {"fredf"}


def test_rel_gain_sign_convention():
    feats, res = make_case()
    d = SEL.build_dataset(feats, res)
    assert np.allclose(d.rel_gain["none"], 0.0)
    assert d.rel_gain["san_lite"].mean() < 0        # 恒定有害插件的相对增益为负


def test_build_dataset_requires_control_column():
    feats, res = make_case()
    with pytest.raises(ValueError, match="对照组"):
        SEL.build_dataset(feats, res[res.plugin != "none"])


def test_build_dataset_detects_missing_features():
    feats, res = make_case()
    with pytest.raises(ValueError, match="没有对应特征行"):
        SEL.build_dataset(feats[feats.dataset != "D3"], res)


# --------------------------------------------------------------------------- #
# 2. LODO 无泄漏
# --------------------------------------------------------------------------- #
def test_lodo_has_no_dataset_leakage(monkeypatch):
    """拦截每次 fit，确认训练行的数据集集合与该折测试数据集互斥。"""
    feats, res = make_case()
    d = SEL.build_dataset(feats, res)
    seen: list[tuple[str, set[str]]] = []
    orig = SEL.make_model

    def spy(kind="gbdt", seed=26):
        model = orig(kind, seed)
        real_fit = model.fit

        def fit(X, y):
            idx = X.index
            seen.append((str(sorted(set(d.groups.iloc[idx])))))
            return real_fit(X, y)

        model.fit = fit  # type: ignore[method-assign]
        return model

    monkeypatch.setattr(SEL, "make_model", spy)
    preds, _ = SEL.lodo_predict(d, n_repeats_importance=1)
    folds = sorted(preds["fold"].unique())
    assert len(folds) == 6
    for fold, train_sets in zip(folds, seen):
        assert fold not in train_sets
    # 每个决策单元恰好被预测一次
    assert len(preds) == len(d.X)
    assert preds.groupby(SEL.BLOCK_KEYS).size().max() == 1


def test_lodo_learns_the_signal():
    feats, res = make_case(n_datasets=8)
    d = SEL.build_dataset(feats, res, tau=0.0)
    preds, imp = SEL.lodo_predict(d, n_repeats_importance=4)
    acc = SEL.accuracy_report(preds)
    assert acc["accuracy"] > acc["majority_baseline"]
    assert acc["n"] == len(d.X)
    assert set(acc["per_fold_accuracy"]) == set(f"D{i}" for i in range(8))
    # 决定标签的两个特征必须排在 permutation importance 前列
    top = imp.head(4)["feature"].tolist()
    assert {"ns_mean_drift", "fr_topk_energy"} & set(top)


def test_logreg_backend_runs():
    feats, res = make_case(n_datasets=5)
    d = SEL.build_dataset(feats, res, tau=0.0)
    preds, _ = SEL.lodo_predict(d, kind="logreg", n_repeats_importance=1)
    assert len(preds) == len(d.X)
    imp = SEL.global_importance(d, kind="logreg")
    assert "logreg_abs_coef_mean" in imp.columns and (imp.iloc[:, 1] >= 0).all()


def test_make_model_rejects_unknown_kind():
    with pytest.raises(ValueError, match="未知模型类型"):
        SEL.make_model("transformer")


# --------------------------------------------------------------------------- #
# 3. 决策收益
# --------------------------------------------------------------------------- #
def test_decision_gain_policies_and_bounds():
    feats, res = make_case(n_datasets=8)
    d = SEL.build_dataset(feats, res, tau=0.0)
    preds, _ = SEL.lodo_predict(d, n_repeats_importance=1)
    g = SEL.decision_gain(d, preds).set_index("policy")

    assert {"always_none", "always_revin", "always_fredf", "always_san_lite",
            "selector_lodo", "best_fixed_lodo", "oracle"} == set(g.index)
    assert g.loc["always_none", "mean_rel_gain_pct"] == pytest.approx(0.0)
    assert g.loc["oracle", "mean_regret_vs_oracle_pct"] == pytest.approx(0.0)
    # oracle 是下界，任何策略不可能比它更好
    assert g["mean_mse"].min() == pytest.approx(g.loc["oracle", "mean_mse"])
    # selector 应当优于最强固定策略，且 regret 明显小于「永远不挂」
    best_fixed = g[g.index.str.startswith("always_")]["mean_mse"].min()
    assert g.loc["selector_lodo", "mean_mse"] <= best_fixed
    assert g.loc["selector_lodo", "mean_regret_vs_oracle_pct"] < \
        g.loc["always_none", "mean_regret_vs_oracle_pct"]
    assert 0.0 <= g.loc["selector_lodo", "harmful_rate"] <= 1.0
    # note 只标两类：事后上界（always_* 里最小的一行）与可部署强基线 best_fixed_lodo
    assert g.loc["best_fixed_lodo", "note"].startswith("最强固定策略(可部署基线")
    post_hoc = g[g.note.str.startswith("事后最优固定臂")]
    assert len(post_hoc) == 1 and post_hoc.index[0].startswith("always_")
    # 折外基线不可能好过在同一批块上事后挑出来的固定臂
    assert g.loc["best_fixed_lodo", "mean_mse"] >= best_fixed - 1e-12


def test_decision_gain_fills_pct_choose_none_for_fixed_policies():
    """回归：2026-09-14 的报告里 always_* 四行的 pct_choose_none 打成了 NaN，
    看起来像 bug。固定策略的这一列不是缺失值，而是恒等于 100（always_none）或 0。"""
    feats, res = make_case(n_datasets=6)
    d = SEL.build_dataset(feats, res, tau=0.0)
    preds, _ = SEL.lodo_predict(d, n_repeats_importance=1)
    g = SEL.decision_gain(d, preds).set_index("policy")
    assert g["pct_choose_none"].notna().all(), g["pct_choose_none"]
    assert g.loc["always_none", "pct_choose_none"] == pytest.approx(100.0)
    for p in ("always_revin", "always_fredf", "always_san_lite"):
        assert g.loc[p, "pct_choose_none"] == pytest.approx(0.0)


def test_decision_gain_alignment_is_key_based():
    """打乱预测行序不能改变 selector 的收益（对齐必须按块 key，不是按行号）。"""
    feats, res = make_case(n_datasets=6)
    d = SEL.build_dataset(feats, res, tau=0.0)
    preds, _ = SEL.lodo_predict(d, n_repeats_importance=1)
    g1 = SEL.decision_gain(d, preds).set_index("policy")
    g2 = SEL.decision_gain(d, preds.sample(frac=1.0, random_state=3)).set_index("policy")
    assert g1.loc["selector_lodo", "mean_mse"] == pytest.approx(g2.loc["selector_lodo", "mean_mse"])


# --------------------------------------------------------------------------- #
# 4. 消融与端到端
# --------------------------------------------------------------------------- #
def test_feature_group_ablation_marks_decisive_group():
    feats, res = make_case(n_datasets=8)
    d = SEL.build_dataset(feats, res, tau=0.0)
    abl = SEL.feature_group_ablation(d)
    assert abl.iloc[0]["ablation"].startswith("full")
    row = abl[abl.ablation == "-ns_非平稳度"].iloc[0]
    # 去掉决定性的非平稳度特征组，决策收益必须下降
    assert row["mean_rel_gain_pct"] < abl.iloc[0]["mean_rel_gain_pct"]
    assert row["n_features"] < abl.iloc[0]["n_features"]


def test_run_writes_all_artifacts(tmp_path):
    feats, res = make_case(n_datasets=6)
    s = SEL.run(feats, res, tmp_path, tau=0.0)
    for f in ["lodo_predictions.csv", "feature_importance_permutation.csv",
              "feature_importance_model.csv", "decision_gain.csv",
              "relative_gain_matrix.csv", "feature_group_ablation.csv"]:
        assert (tmp_path / f).exists(), f
    assert s["n_blocks"] == 6 * 2 * 2
    assert 0.0 <= s["accuracy"] <= 1.0
    assert 0 < len(s["top_features"]) <= 10  # 特征数少于 10 时取全部


# --------------------------------------------------------------------------- #
# 5. best_fixed_lodo：折外的可部署强基线（2026-09-14 code review 回归）
# --------------------------------------------------------------------------- #
# 手工构造的 MSE 表（每个数据集两个块，值刻意做成「事后最优臂 ≠ 折外最优臂」）：
#   arms:      none    A      B
#   D0        1.00    0.95   0.01     ← 只有 D0 上 B 好得离谱
#   D1        1.00    0.50   0.95
#   D2        1.00    0.50   0.95
# 事后（在全部块上）取均值：none=1.0、A=0.65、B=(0.01+0.95+0.95)/3≈0.6367 → 事后最优是 B；
# 而折外选臂：留出 D0 时训练折只有 D1/D2（A=0.50 < B=0.95）→ 必须选 A，
# 留出 D1 时训练折是 D0/D2（A=0.725 > B=0.48）→ 选 B。两者结论不同，这正是本组用例的区分力。
_MSE_TABLE = {"D0": (1.00, 0.95, 0.01), "D1": (1.00, 0.50, 0.95), "D2": (1.00, 0.50, 0.95)}
_ARMS = ("none", "A", "B")


def _fixed_data(table: dict[str, tuple[float, ...]], horizons=(96, 336)) -> SEL.SelectorData:
    """用给定的 (数据集 → 各臂 MSE) 表直接拼一个 SelectorData（不经过模型，期望值可手算）。"""
    blocks, rows = [], []
    for ds, vals in table.items():
        for h in horizons:
            blocks.append({"dataset": ds, "pred_len": h, "backbone": "DLinear"})
            rows.append(dict(zip(_ARMS, vals)))
    mse = pd.DataFrame(rows, columns=list(_ARMS))
    blk = pd.DataFrame(blocks)
    rel = mse.apply(lambda col: (mse["none"] - col) / mse["none"])
    y = pd.Series(rel.drop(columns=["none"]).idxmax(axis=1), name="label")
    return SEL.SelectorData(X=pd.DataFrame({"dummy": np.arange(len(mse), dtype=float)}),
                            y=y, groups=blk["dataset"], mse=mse, rel_gain=rel, blocks=blk)


def _trivial_preds(data: SEL.SelectorData, choice: str = "none") -> pd.DataFrame:
    """一个恒定预测的假 selector：只为了让 decision_gain 能算出 selector 那一行。"""
    return data.blocks.assign(y_true=data.y.to_numpy(), y_pred=choice,
                              fold=data.blocks["dataset"].to_numpy())


def test_best_fixed_lodo_picks_arm_from_training_folds_only():
    """折外选臂：留出折上最好的臂**不能**被选中，选的必须是训练折上最好的臂。

    退回旧实现（只有事后的 always_* 行）时本用例连 `lodo_best_fixed` 都找不到，直接失败。
    """
    d = _fixed_data(_MSE_TABLE)
    arms, picked = SEL.lodo_best_fixed(d)
    ds = d.blocks["dataset"].to_numpy()

    # 留出 D0：训练折 D1/D2 上 A(0.50) 优于 B(0.95) → 选 A，尽管 D0 自己上 B 只有 0.01
    assert set(arms[ds == "D0"]) == {"A"}
    assert set(picked[ds == "D0"]) == {0.95}
    # 留出 D1：训练折 D0/D2 上 B((0.01+0.95)/2=0.48) 优于 A(0.725) → 选 B
    assert set(arms[ds == "D1"]) == {"B"} and set(arms[ds == "D2"]) == {"B"}
    assert set(picked[ds == "D1"]) == {0.95}
    # 关键：选出来的臂不是「该块自己的最优臂」（那才是事后口径）
    own_best = d.mse.idxmin(axis=1).to_numpy()
    assert (arms.to_numpy() != own_best).any()


def test_best_fixed_lodo_is_invariant_to_held_out_fold_errors():
    """只改留出折自己的误差，不得改变该折的选臂结果（这就是「不看留出折」的可检验定义）。

    事后实现会随着 D0 的误差翻转选臂，因此这条断言对旧口径必然失败。
    """
    d = _fixed_data(_MSE_TABLE)
    arm0 = SEL.lodo_best_fixed(d)[0]
    ds = d.blocks["dataset"].to_numpy()

    for new_d0 in [(1.00, 0.95, 9.99), (1.00, 0.95, 0.0001), (1.00, 0.10, 0.20)]:
        tweaked = _fixed_data({**_MSE_TABLE, "D0": new_d0})
        arms = SEL.lodo_best_fixed(tweaked)[0]
        # D0 是留出折 → 选臂只由 D1/D2 决定，必须与原表完全一致
        assert list(arms[ds == "D0"]) == list(arm0[ds == "D0"]) == ["A", "A"], new_d0


def test_decision_gain_reports_lodo_baseline_and_marks_post_hoc_row():
    """decision_gain 里 best_fixed_lodo 的数值可手算，且事后那一行的 note 写明不可部署。"""
    d = _fixed_data(_MSE_TABLE)
    g = SEL.decision_gain(d, _trivial_preds(d)).set_index("policy")

    # 折外基线：D0→A(0.95)、D1→B(0.95)、D2→B(0.95)，六个块全是 0.95
    assert g.loc["best_fixed_lodo", "mean_mse"] == pytest.approx(0.95)
    # 事后最优固定臂是 always_B（(0.01+0.95+0.95)/3），比折外基线好——它是上界不是基线
    assert g.loc["always_B", "mean_mse"] == pytest.approx((0.01 + 0.95 + 0.95) / 3)
    assert g.loc["always_B", "mean_mse"] < g.loc["best_fixed_lodo", "mean_mse"]
    assert g.loc["always_B", "note"].startswith("事后最优固定臂")
    assert "不可部署" in g.loc["always_B", "note"]
    assert g.loc["always_A", "note"] == "" and g.loc["always_none", "note"] == ""
    # 折外基线一次都没选对照组；相对增益/regret 的口径与其它行一致
    assert g.loc["best_fixed_lodo", "pct_choose_none"] == pytest.approx(0.0)
    assert g.loc["best_fixed_lodo", "mean_rel_gain_pct"] == pytest.approx(5.0)
    assert g.loc["best_fixed_lodo", "n_blocks"] == 6


def test_best_fixed_lodo_may_choose_the_control_arm():
    """所有插件都有害时，可部署的最强固定策略就是「不挂」——对照组是合法选项。

    同时验证：此时事后那一行是 always_none，note 必须点明「胜者即对照组」，
    否则论文会把对照组当成 selector 要打败的「最强固定策略」。
    """
    d = _fixed_data({"D0": (1.0, 1.1, 1.2), "D1": (1.0, 1.1, 1.2), "D2": (1.0, 1.1, 1.2)})
    arms, picked = SEL.lodo_best_fixed(d)
    assert set(arms) == {"none"} and set(picked) == {1.0}

    g = SEL.decision_gain(d, _trivial_preds(d)).set_index("policy")
    assert g.loc["best_fixed_lodo", "mean_mse"] == pytest.approx(1.0)
    assert g.loc["best_fixed_lodo", "pct_choose_none"] == pytest.approx(100.0)
    assert g.loc["best_fixed_lodo", "harmful_rate"] == pytest.approx(0.0)
    assert "本次胜者即对照组" in g.loc["always_none", "note"]


def test_best_fixed_lodo_holds_out_all_blocks_of_a_dataset():
    """LODO 的分组单位是数据集：同一数据集的所有 horizon 必须一起留出。

    构造上让每个数据集内部两个 horizon 的最优臂相反：若实现按「行」而不是按「数据集」
    留出，某个 horizon 会靠同数据集另一个 horizon 的信息选臂，本用例会失败。
    """
    blocks, rows = [], []
    per_ds = {"D0": [(1.0, 0.5, 1.5), (1.0, 1.5, 0.5)],       # 内部两个 horizon 互相矛盾
              "D1": [(1.0, 0.9, 0.6), (1.0, 0.9, 0.6)],
              "D2": [(1.0, 0.9, 0.6), (1.0, 0.9, 0.6)]}
    for ds, vals in per_ds.items():
        for h, v in zip((96, 336), vals):
            blocks.append({"dataset": ds, "pred_len": h, "backbone": "DLinear"})
            rows.append(dict(zip(_ARMS, v)))
    mse = pd.DataFrame(rows, columns=list(_ARMS))
    blk = pd.DataFrame(blocks)
    rel = mse.apply(lambda col: (mse["none"] - col) / mse["none"])
    d = SEL.SelectorData(X=pd.DataFrame({"dummy": np.arange(len(mse), dtype=float)}),
                         y=pd.Series(rel.drop(columns=["none"]).idxmax(axis=1)),
                         groups=blk["dataset"], mse=mse, rel_gain=rel, blocks=blk)
    arms = SEL.lodo_best_fixed(d)[0]
    # 留出 D0 的两个块时训练折只有 D1/D2（B=0.6 优于 A=0.9）→ 两块都选 B
    assert list(arms[blk["dataset"] == "D0"]) == ["B", "B"]


def test_selector_and_lodo_baseline_share_the_same_out_of_fold_protocol():
    """论文要引用的一对比较必须同口径：selector_lodo 与 best_fixed_lodo 都只用训练折信息。"""
    feats, res = make_case(n_datasets=8)
    d = SEL.build_dataset(feats, res, tau=0.0)
    preds, _ = SEL.lodo_predict(d, n_repeats_importance=1)
    g = SEL.decision_gain(d, preds).set_index("policy")
    # 合成场景里 selector 拿到了特征信号，应当打败折外最强固定臂
    assert g.loc["selector_lodo", "mean_mse"] < g.loc["best_fixed_lodo", "mean_mse"]
    # 折外基线永远不可能好过「在同一批块上事后挑臂」的上界（同一批臂池、更少的信息）
    post_hoc_best = g[g.index.str.startswith("always_")]["mean_mse"].min()
    assert g.loc["best_fixed_lodo", "mean_mse"] >= post_hoc_best - 1e-12
    # 它必须落在真实臂池里（合成场景中 revin/fredf 收益对称，折外常常挑到对照组 none）
    arms = set(SEL.lodo_best_fixed(d)[0])
    assert arms <= set(d.mse.columns) and arms
    assert 0.0 <= g.loc["best_fixed_lodo", "harmful_rate"] <= 1.0
