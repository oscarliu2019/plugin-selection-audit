"""复杂度特征 → 插件选择：轻量、可解释的 selector。

为什么不用深度模型
------------------
本文的卖点是「可迁移的选择规则/知识」，不是又一个网络。因此 selector 必须：
- **可解释**：梯度提升树的特征重要性 / 逻辑回归系数可以直接写进论文；
- **极轻**：训练样本量只有 (数据集 × horizon × 骨干) 量级（≤144 行），
  深度模型在这个样本量上只会过拟合；
- **零算力增量**：特征来自原始序列（O(n log n)），推理是一次树遍历。

决策单元与标签
--------------
决策单元 = 一个 (数据集, horizon, 骨干) 格；
标签 = 该格上**相对增益最大**的插件；若最大增益小于阈值 `tau`（默认 0.5%），
标签取对照组 `none` —— 即「不该挂」也是一个合法决策，这正是本文的核心问题。

防泄漏
------
**leave-one-dataset-out (LODO)** 交叉验证：留出的那个数据集的**全部** horizon/骨干
都不参与训练。这是审稿人必查的点：若按行随机切分，同一数据集的其它 horizon
会把该数据集的特征分布泄漏给训练集，指标会虚高。

决策收益评估（把 empirical study 升格为方法贡献）
------------------------------------------------
下列策略在同一批格子上比较：
- `always_none`           永远不挂（对照）
- `always_<plugin>`       永远挂某个固定插件；**注意**这些行里 mean_mse 最小的那一条是
  在同一批评估块上事后挑出来的，属于 arm 级别的事后上界（与 `oracle` 同类），不是基线
- `best_fixed_lodo`       ⭐ 可部署的强固定基线：每个 LODO 折**只用训练折**的块挑固定臂，
  再在留出折上评估（与 selector 的折外口径一致，论文里 selector 要打败的就是这一行）
- `selector`              按 selector 的 LODO 预测选择
- `oracle`                事后诸葛亮（上界）
指标：平均 MSE、相对对照组的平均相对增益、相对 oracle 的 regret、以及有害决策比例。

用法
----
    python -m src.selector --features artifacts/features.csv \
        --results results/results.csv --out artifacts/selector --model gbdt
    python -m src.selector --demo         # 用合成结果自检（无 GPU 也能跑）
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BLOCK_KEYS = ["dataset", "pred_len", "backbone"]


# --------------------------------------------------------------------------- #
# 数据准备
# --------------------------------------------------------------------------- #
@dataclass
class SelectorData:
    X: pd.DataFrame              # 特征（含骨干 one-hot）
    y: pd.Series                 # 最优插件标签
    groups: pd.Series            # 数据集名（LODO 的分组）
    mse: pd.DataFrame            # 每格 × 每插件的真实 MSE（用于策略评估）
    rel_gain: pd.DataFrame       # 每格 × 每插件的相对增益
    blocks: pd.DataFrame         # 块的 key


def build_dataset(
    features: pd.DataFrame,
    results: pd.DataFrame,
    control: str = "none",
    tau: float = 0.005,
    metric: str = "mse",
) -> SelectorData:
    """把特征表与结果表拼成 selector 的训练数据。

    features: 一行 = (dataset, pred_len)；results: 一行 = 一个 cell（多 seed 会被平均）。
    """
    agg = results.groupby(BLOCK_KEYS + ["plugin"])[metric].mean().reset_index()
    mse = agg.pivot_table(index=BLOCK_KEYS, columns="plugin", values=metric).dropna(how="any")
    if control not in mse.columns:
        raise ValueError(f"结果表里缺少对照组 {control!r}")
    rel = mse.apply(lambda col: (mse[control] - col) / mse[control])

    best_plugin = rel.drop(columns=[control]).idxmax(axis=1)
    best_gain = rel.drop(columns=[control]).max(axis=1)
    y = pd.Series(np.where(best_gain >= tau, best_plugin, control), index=mse.index, name="label")

    blocks = mse.index.to_frame(index=False)
    feat_cols = [c for c in features.columns if c not in ("dataset", "pred_len")]
    X = blocks.merge(features[["dataset", "pred_len"] + feat_cols], on=["dataset", "pred_len"], how="left")
    missing = X[feat_cols].isna().all(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} 个格子没有对应特征行，请先补齐 features.csv")
    # 骨干作为分类变量做 one-hot（selector 必须知道挂在谁身上）
    X = pd.concat([X, pd.get_dummies(X["backbone"], prefix="bb").astype(float)], axis=1)
    X = X.drop(columns=["dataset", "backbone"])
    if "hz_pred_len" in X.columns:  # 与 block key 的 pred_len 完全重复
        X = X.drop(columns=["hz_pred_len"])
    X = X.fillna(X.median(numeric_only=True))
    return SelectorData(
        X=X.reset_index(drop=True),
        y=y.reset_index(drop=True),
        groups=blocks["dataset"].reset_index(drop=True),
        mse=mse.reset_index(drop=True),
        rel_gain=rel.reset_index(drop=True),
        blocks=blocks,
    )


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def make_model(kind: str = "gbdt", seed: int = 26):
    """两种可解释模型：梯度提升树（默认）与多项逻辑回归。"""
    if kind == "gbdt":
        return GradientBoostingClassifier(
            n_estimators=120, learning_rate=0.08, max_depth=2, subsample=0.9, random_state=seed
        )
    if kind == "logreg":
        return Pipeline([
            ("scale", StandardScaler()),
            # 注意: sklearn>=1.7 起移除了 multi_class 参数, 多分类默认即 multinomial
            ("clf", LogisticRegression(max_iter=2000, C=1.0, random_state=seed)),
        ])
    raise ValueError(f"未知模型类型 {kind!r}（可选: gbdt / logreg）")


# --------------------------------------------------------------------------- #
# LODO 交叉验证
# --------------------------------------------------------------------------- #
def lodo_predict(data: SelectorData, kind: str = "gbdt", seed: int = 26,
                 n_repeats_importance: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """leave-one-dataset-out 预测 + 逐折 permutation importance。

    返回 (predictions, importance)。predictions 每行对应一个决策单元。
    """
    preds: list[dict[str, Any]] = []
    imps: list[pd.Series] = []
    for ds in sorted(data.groups.unique()):
        te = (data.groups == ds).to_numpy()
        tr = ~te
        if data.y[tr].nunique() < 2:
            print(f"[selector] 折 {ds}: 训练集只有一个类别，退化为常数预测")
        model = make_model(kind, seed)
        model.fit(data.X[tr], data.y[tr])
        yhat = model.predict(data.X[te])
        for i, idx in enumerate(np.where(te)[0]):
            preds.append({
                **data.blocks.iloc[idx].to_dict(),
                "y_true": data.y.iloc[idx], "y_pred": yhat[i],
                "fold": ds,
            })
        try:
            pi = permutation_importance(
                model, data.X[te], data.y[te], n_repeats=n_repeats_importance,
                random_state=seed, scoring="accuracy",
            )
            imps.append(pd.Series(pi.importances_mean, index=data.X.columns, name=ds))
        except Exception as exc:  # 某折只有单一类别时 permutation importance 无意义
            print(f"[selector] 折 {ds} 的 permutation importance 跳过: {exc}")
    imp = pd.concat(imps, axis=1) if imps else pd.DataFrame(index=data.X.columns)
    importance = pd.DataFrame({
        "feature": data.X.columns,
        "perm_importance_mean": imp.mean(axis=1).to_numpy() if not imp.empty else np.nan,
        "perm_importance_std": imp.std(axis=1).to_numpy() if not imp.empty else np.nan,
    }).sort_values("perm_importance_mean", ascending=False)
    return pd.DataFrame(preds), importance


def global_importance(data: SelectorData, kind: str = "gbdt", seed: int = 26) -> pd.DataFrame:
    """在全部数据上拟合一次，输出模型自带的特征重要性/系数（写论文用）。"""
    model = make_model(kind, seed)
    model.fit(data.X, data.y)
    if kind == "gbdt":
        vals = model.feature_importances_
        col = "gbdt_importance"
    else:
        clf = model.named_steps["clf"]
        vals = np.abs(clf.coef_).mean(axis=0)
        col = "logreg_abs_coef_mean"
    return pd.DataFrame({"feature": data.X.columns, col: vals}).sort_values(col, ascending=False)


# --------------------------------------------------------------------------- #
# 决策收益评估
# --------------------------------------------------------------------------- #
def lodo_best_fixed(data: SelectorData, control: str = "none") -> tuple[pd.Series, np.ndarray]:
    """⭐ 可部署的「最强固定臂」基线：逐 LODO 折**只用训练折**的块选臂，再在留出折上评估。

    2026-09-14 code review 修复项。原来 ``decision_gain`` 只有一条被标注「最强固定策略」
    的 ``always_*`` 行，而它是在**同一批评估块**上按 ``mean_mse`` 事后取最小值选出来的
    （用到了留出块的测试误差），因此它是 arm 级别的**事后上界**（与 ``oracle`` 同类），
    不是可部署基线：真实部署时没人能先看到留出块的测试误差再决定挂哪个插件。
    论文若把它当作 selector 要打败的基线，口径与 selector 的 LODO 折外口径不一致
    （selector 吃亏、审稿人一个对照就能推翻）。

    ``src/selector_window.py`` 早就区分了 ``best_fixed_val``（选臂用验证集，真正的强基线）
    与 ``best_fixed_test``（事后），块级这里补上同样的区分：选臂只允许看**训练折**，
    与 selector 的 LODO 完全同口径，两者因此可以直接比较。

    返回 ``(每块选中的臂, 每块对应的 MSE)``，行序与 ``data.mse`` / ``data.blocks`` 对齐。
    候选臂包含对照组 ``control``——「一个插件都不挂」本身就是合法且可部署的固定决策。
    """
    mse = data.mse
    arms = list(mse.columns)
    m = mse.to_numpy(dtype=float)
    groups = pd.Series(data.groups).to_numpy()
    if len(groups) != len(m):
        raise ValueError(f"groups 与 mse 行数不一致: {len(groups)} vs {len(m)}")
    chosen = np.empty(len(m), dtype=object)
    picked = np.empty(len(m), dtype=float)
    for ds in sorted(set(groups)):
        te = groups == ds
        tr = ~te
        # 只有一个数据集时没有任何训练折可用 → 退回对照组，绝不允许偷看留出折来选臂
        j = int(np.argmin(m[tr].mean(axis=0))) if tr.any() else arms.index(control)
        chosen[te] = arms[j]
        picked[te] = m[te, j]
    return pd.Series(chosen, index=mse.index, name="best_fixed_lodo_arm"), picked


def decision_gain(data: SelectorData, preds: pd.DataFrame, control: str = "none") -> pd.DataFrame:
    """比较 selector / 永远挂某插件 / 折外最强固定臂 / 永远不挂 / oracle 各类策略。"""
    mse = data.mse
    plugins = list(mse.columns)
    ctrl = mse[control].to_numpy(float)
    oracle = mse.min(axis=1).to_numpy(float)

    def _row(name: str, chosen_mse: np.ndarray, choices: pd.Series | None = None) -> dict[str, Any]:
        rel = (ctrl - chosen_mse) / ctrl
        regret = (chosen_mse - oracle) / ctrl
        out = {
            "policy": name,
            "mean_mse": float(np.mean(chosen_mse)),
            "mean_rel_gain_pct": float(np.mean(rel) * 100),
            "median_rel_gain_pct": float(np.median(rel) * 100),
            "mean_regret_vs_oracle_pct": float(np.mean(regret) * 100),
            "harmful_rate": float(np.mean(chosen_mse > ctrl + 1e-12)),
            "n_blocks": int(len(chosen_mse)),
        }
        if choices is not None:
            out["pct_choose_none"] = float(np.mean(choices.to_numpy() == control) * 100)
        return out

    # 固定策略的 pct_choose_none 不是「缺失」，而是恒等于 0 或 100：显式传进去，
    # 否则这一列在报告里打出整片 NaN，看起来像 bug（2026-09-14 报告里就是这样）。
    idx = mse.index
    rows = [_row("always_none", ctrl, pd.Series(control, index=idx))]
    for p in plugins:
        if p == control:
            continue
        rows.append(_row(f"always_{p}", mse[p].to_numpy(float), pd.Series(p, index=idx)))

    # selector：预测顺序与 data.blocks 对齐
    key = data.blocks.astype(str).agg("|".join, axis=1)
    pk = preds[BLOCK_KEYS].astype(str).agg("|".join, axis=1)
    choice = pd.Series(preds["y_pred"].to_numpy(), index=pk).reindex(key)
    sel_mse = np.array([mse.iloc[i][choice.iloc[i]] for i in range(len(mse))], dtype=float)
    rows.append(_row("selector_lodo", sel_mse, choice))

    # ⭐ 折外最强固定臂：唯一与 selector 同口径（只用训练折信息）的固定策略基线。
    # 放在 selector 旁边，方便论文直接引用「selector vs 可部署强基线」这一对比较。
    bf_arms, bf_mse = lodo_best_fixed(data, control)
    rows.append(_row("best_fixed_lodo", bf_mse, bf_arms))

    # oracle 上界
    rows.append(_row("oracle", oracle, mse.idxmin(axis=1)))
    df = pd.DataFrame(rows)

    # note 的口径必须写清楚（2026-09-14 code review 修复项）：
    # always_* 里 mean_mse 最小的那一行是**在同一批评估块上事后选出**的（arm 级事后上界），
    # 历史上它被标成「最强固定策略」，很容易被论文误引成 selector 该打败的基线。
    # 真正可部署的强基线是 best_fixed_lodo（选臂只看训练折）。
    post_hoc = df[df.policy.str.startswith("always_")].sort_values("mean_mse").iloc[0]["policy"]
    post_hoc_note = "事后最优固定臂(上界：在同一批评估块上按 mean_mse 选出，不可部署)"
    if post_hoc == f"always_{control}":
        # startswith("always_") 也会把对照组选进来，此时这一行说的其实是「不挂最好」
        post_hoc_note += "；本次胜者即对照组"
    notes = np.where(df.policy == post_hoc, post_hoc_note, "")
    notes = np.where(df.policy == "best_fixed_lodo",
                     "最强固定策略(可部署基线：每折只用训练折的块选臂，与 selector 同口径)",
                     notes)
    df["note"] = notes
    return df


def accuracy_report(preds: pd.DataFrame) -> dict[str, Any]:
    ok = (preds["y_true"] == preds["y_pred"])
    per_fold = preds.assign(ok=ok).groupby("fold")["ok"].mean()
    return {
        "accuracy": float(ok.mean()),
        "n": int(len(preds)),
        "majority_baseline": float(preds["y_true"].value_counts(normalize=True).max()),
        "per_fold_accuracy": per_fold.round(3).to_dict(),
    }


# --------------------------------------------------------------------------- #
# 特征组消融
# --------------------------------------------------------------------------- #
FEATURE_GROUPS = {
    "pc_窗口复杂度": "pc_",
    "ns_非平稳度": "ns_",
    "ch_通道相关谱": "ch_",
    "fr_频域集中度": "fr_",
    "sd_趋势季节自相关": "sd_",
    "hz_horizon条件": "hz_",
    "bb_骨干身份": "bb_",
}


def feature_group_ablation(data: SelectorData, kind: str = "gbdt", seed: int = 26,
                           control: str = "none") -> pd.DataFrame:
    """逐组去掉特征后重跑 LODO，量化每组特征对决策收益的贡献。"""
    rows = []
    full_preds, _ = lodo_predict(data, kind, seed, n_repeats_importance=1)
    full_gain = decision_gain(data, full_preds, control)
    full_sel = full_gain[full_gain.policy == "selector_lodo"].iloc[0]
    rows.append({"ablation": "full(全部特征)", "n_features": data.X.shape[1],
                 "lodo_accuracy": accuracy_report(full_preds)["accuracy"],
                 "mean_rel_gain_pct": full_sel["mean_rel_gain_pct"],
                 "mean_regret_vs_oracle_pct": full_sel["mean_regret_vs_oracle_pct"]})
    for name, prefix in FEATURE_GROUPS.items():
        keep = [c for c in data.X.columns if not c.startswith(prefix)]
        if len(keep) == data.X.shape[1] or not keep:
            continue
        sub = SelectorData(data.X[keep], data.y, data.groups, data.mse, data.rel_gain, data.blocks)
        preds, _ = lodo_predict(sub, kind, seed, n_repeats_importance=1)
        g = decision_gain(sub, preds, control)
        srow = g[g.policy == "selector_lodo"].iloc[0]
        rows.append({"ablation": f"-{name}", "n_features": len(keep),
                     "lodo_accuracy": accuracy_report(preds)["accuracy"],
                     "mean_rel_gain_pct": srow["mean_rel_gain_pct"],
                     "mean_regret_vs_oracle_pct": srow["mean_regret_vs_oracle_pct"]})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 端到端
# --------------------------------------------------------------------------- #
def run(features: pd.DataFrame, results: pd.DataFrame, out_dir: str | Path,
        kind: str = "gbdt", control: str = "none", tau: float = 0.005,
        seed: int = 26, ablation: bool = True) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = build_dataset(features, results, control=control, tau=tau)
    preds, imp = lodo_predict(data, kind, seed)
    gain = decision_gain(data, preds, control)
    gimp = global_importance(data, kind, seed)

    preds.to_csv(out / "lodo_predictions.csv", index=False)
    imp.to_csv(out / "feature_importance_permutation.csv", index=False)
    gimp.to_csv(out / "feature_importance_model.csv", index=False)
    gain.to_csv(out / "decision_gain.csv", index=False)
    data.rel_gain.assign(**data.blocks).to_csv(out / "relative_gain_matrix.csv", index=False)
    if ablation:
        abl = feature_group_ablation(data, kind, seed, control)
        abl.to_csv(out / "feature_group_ablation.csv", index=False)

    acc = accuracy_report(preds)
    summary = {"model": kind, "tau": tau, "n_blocks": len(data.X),
               "label_distribution": data.y.value_counts().to_dict(), **acc,
               "decision_gain": gain.to_dict("records"),
               "top_features": imp.head(10)["feature"].tolist()}
    print(f"[selector] 决策单元 {len(data.X)} 个，标签分布 {summary['label_distribution']}")
    print(f"[selector] LODO 准确率 {acc['accuracy']:.3f}（多数类基线 {acc['majority_baseline']:.3f}）")
    # 论文要引用的对比必须是「折外 vs 折外」：selector_lodo vs best_fixed_lodo。
    # always_* 里最小的那一行是事后上界，不能当基线（2026-09-14 code review 修复项）。
    g = gain.set_index("policy")["mean_mse"]
    if {"selector_lodo", "best_fixed_lodo"} <= set(g.index):
        print(f"[selector] 折外口径对比：selector_lodo {g['selector_lodo']:.4f} vs "
              f"best_fixed_lodo（可部署强基线）{g['best_fixed_lodo']:.4f}")
    print(f"[selector] 输出目录 {out}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="复杂度特征 → 插件选择器")
    ap.add_argument("--features", default="artifacts/features.csv")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--out", default="artifacts/selector")
    ap.add_argument("--model", default="gbdt", choices=["gbdt", "logreg"])
    ap.add_argument("--control", default="none")
    ap.add_argument("--tau", type=float, default=0.005, help="小于该相对增益就判定为『不该挂』")
    ap.add_argument("--no-ablation", action="store_true", help="跳过特征组消融（较慢）")
    ap.add_argument("--demo", action="store_true", help="用合成结果 + 已算好的真实特征自检")
    args = ap.parse_args()

    if args.demo:
        from .config import MatrixConfig
        from .synth import make_results

        feats = pd.read_csv(args.features)
        results = make_results(feats, MatrixConfig.load())
        print(f"[selector] demo: 合成结果 {results.shape}")
    else:
        feats = pd.read_csv(args.features)
        results = pd.read_csv(args.results)

    s = run(feats, results, args.out, args.model, args.control, args.tau,
            ablation=not args.no_ablation)
    print("\n=== 决策收益对比 ===")
    print(pd.DataFrame(s["decision_gain"]).to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\n=== Top-10 特征（permutation importance, LODO 平均）===")
    print(pd.read_csv(Path(args.out) / "feature_importance_permutation.csv").head(10)
          .to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
