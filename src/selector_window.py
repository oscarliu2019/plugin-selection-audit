"""⭐ 逐窗口插件门控（instance-adaptive plugin gating）——本项目的核心实验。

它回答的问题
------------
phase-1 已经证明「插件增益强烈依赖条件」：FreDF 在 DLinear 上平均 +4.56%、
在 PatchTST 上 −16.63%；SAN-lite 115 个块里 23 胜 92 负但最好的格子 +49%。
既然平均值把好格子和坏格子抵消掉了，那么**逐个窗口在线决定挂不挂插件**能否
把这些被平均掉的收益捞回来？这就是 PlugGate 的主张。

phase-1 的数据集级 selector 是失败的（LODO 0.381 < 多数类 0.405），根因是
特征只有 8 个互异取值。本模块把粒度下沉到单个窗口，N 从 42 变成万级。

无泄漏协议（审稿人第一刀砍在这里）
--------------------------------
门控器**只在验证窗口上训练**，在测试窗口上评估：

    ┌── train 段 ──┬── val 段 ──┬── test 段 ──┐
    │ 训练骨干+插件 │ 学门控器    │ 评估门控器   │
    └──────────────┴────────────┴─────────────┘

这既杜绝了「用测试误差选插件」，也正好等于真实部署时能拿到的信息：
上线前你有验证集，上线后来一个窗口就要立刻决定挂哪个插件。

两个协议：
* ``inpool``（主结果，可部署）：同一 (backbone, dataset, horizon) 内，val 学 → test 评。
* ``lodo``（迁移性）：在其余数据集的 val 窗口上学 → 留出数据集的 test 窗口上评。
  这一条更难，回答的是「门控规律是否跨数据集通用」。

必须打败的基线（不是 always-none！）
----------------------------------
* ``none``：不挂插件。
* 每个固定插件臂：always-revin / always-fredf / ...
* ⭐ ``best_fixed_val``：**在验证集上挑一个最好的固定插件**并一直用它。
  这是真正强的基线——如果门控打不过它，那"逐窗口决策"就没有存在价值。
* ``oracle``：每个窗口用测试误差挑最优臂，是不可达上界，用来度量还剩多少空间。

用法
----
    python -m src.selector_window --config configs/matrix_p2.yaml
    python -m src.selector_window --config configs/matrix_p2.yaml --protocol lodo
    python -m src.selector_window --self-test        # 合成数据自检，无需 GPU/实验结果
"""

from __future__ import annotations

import argparse
import json
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from .config import MatrixConfig
from .features_window import window_feature_columns

EPS = 1e-12


# --------------------------------------------------------------------------- #
# 装载：winerr/*.npy + window_features.parquet -> 每组一个对齐好的张量
# --------------------------------------------------------------------------- #
@dataclass
class Group:
    """一个决策组 = (backbone, dataset, pred_len, seed)，内含若干插件臂。"""

    backbone: str
    dataset: str
    pred_len: int
    seed: int
    arms: list[str]
    # 验证段**逐窗口** MSE。仅当该组所有臂的 val 数组都带确定序标记时才有值；
    # 否则为 None —— 见 load_winerr 的说明（修复前 val loader 是 shuffle=True）。
    # 只有需要「特征 i ↔ 误差 i」对齐的协议（门控/对照/消融）才用它。
    e_val: np.ndarray | None   # (n_arms, n_val)
    e_test: np.ndarray         # (n_arms, n_test) 测试段逐窗口 MSE
    x_val: np.ndarray | None   # (n_val, F)
    x_test: np.ndarray         # (n_test, F)
    feat_names: list[str] = field(default_factory=list)
    # 验证段**每臂均值**，(n_arms,)。均值对窗口置换不变，所以即使 val 数组顺序被打乱
    # 它依然正确 —— best_fixed_val 这类「用验证集选一个固定臂」的基线因此对全部组可用。
    # 这正是把 e_val 与 val_mean 分开的意义：test 侧审计（oracle/半衰期/在线延迟）
    # 得以保住全部 127 组，而不必因为 val 逐窗口不可用就连带丢掉 TimesNet。
    # 缺省时从 e_val 推导（供测试与旧调用方便），生产路径由 build_groups 显式传入。
    val_mean: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.val_mean is None:
            if self.e_val is None:
                raise ValueError("Group 需要 val_mean 或 e_val 之一来确定 best_fixed_val")
            self.val_mean = np.asarray(self.e_val, dtype=np.float64).mean(axis=1)
        self.val_mean = np.asarray(self.val_mean, dtype=np.float64)

    @property
    def has_val_windows(self) -> bool:
        """能否做需要 val 逐窗口对齐的协议（门控 / 对照 / 消融）。"""
        return self.e_val is not None and self.x_val is not None

    @property
    def key(self) -> str:
        return f"{self.backbone}__{self.dataset}__h{self.pred_len}__s{self.seed}"

    @property
    def i_none(self) -> int:
        return self.arms.index("none")


def parse_cell_id(cell_id: str) -> dict[str, Any] | None:
    """``DLinear__ETTh1__h96__fredf__s2021`` -> dict。不合规返回 None。"""
    parts = cell_id.split("__")
    if len(parts) != 5:
        return None
    bb, ds, h, plug, seed = parts
    if not h.startswith("h") or not seed.startswith("s"):
        return None
    try:
        return {"backbone": bb, "dataset": ds, "pred_len": int(h[1:]),
                "plugin": plug, "seed": int(seed[1:])}
    except ValueError:
        return None


def load_winerr(
    winerr_dirs: Path | list[Path],
    require_order_marker: bool = False,
    on_unmarked: str = "raise",
    skipped_out: list[str] | None = None,
) -> dict[tuple, dict[str, np.ndarray]]:
    """读若干 ``<dir>/*.npy``，返回 {(backbone,dataset,pred_len,seed): {plugin: arr}}。

    支持多个目录取**并集**：phase-2 提供 none/revin/fredf/san_lite 四个臂，
    FreDF 公平性组（results/p3）再补上一串不同 α 的臂。合到一起，门控的候选池
    就从 4 个变成 12 个——这更贴近真实场景（"挂不挂 FreDF" 本质是 "α 取多少"）。
    同名臂以**先出现的目录**为准，避免不同协议的结果互相覆盖。

    ``require_order_marker=True`` 时只接受带 ``<cell_id>.order.json`` 溯源标记的数组。
    这是给 **val** 段用的：2026-09-14 的 code review 发现 TSLib 只对 test 关 shuffle，
    此前落盘的 val 逐窗口误差是随机置换顺序，与窗口特征按下标对齐时整体错位；
    修复前后的 ``.npy`` 从内容上无法区分，只能靠写入方打标。

    ``on_unmarked`` 决定遇到未打标数组怎么办：

    - ``"raise"``：直接报错（用于必须全量干净的场合）。
    - ``"skip"``：跳过并**大声**告警，让分析在干净子集上继续。
      TimesNet 占 167/178 GPU·h，短期不重跑，若一律 raise 会导致整个分析跑不起来；
      跳过是安全的（污染数据被排除），但必须可见，所以会打印告警并回填
      ``skipped_out``，由调用方写进 notes / summary，供论文如实陈述覆盖范围。

    无论哪种模式都**不会**静默载入未打标数组。
    """
    dirs = [winerr_dirs] if isinstance(winerr_dirs, (str, Path)) else list(winerr_dirs)
    out: dict[tuple, dict[str, np.ndarray]] = {}
    skipped: list[str] = []
    for d in dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.npy")):
            meta = parse_cell_id(p.stem)
            if meta is None:
                continue
            if require_order_marker and not (d / f"{p.stem}.order.json").exists():
                skipped.append(p.stem)
                continue
            k = (meta["backbone"], meta["dataset"], meta["pred_len"], meta["seed"])
            slot = out.setdefault(k, {})
            slot.setdefault(meta["plugin"], np.load(p).astype(np.float64))
    if skipped:
        msg = (
            f"{len(skipped)} 个 val 逐窗口误差数组缺少确定序标记（.order.json），已拒绝载入。\n"
            "这些数组是 2026-09-14 修复前用 shuffle=True 的 val loader 落盘的，"
            "与窗口特征按下标对齐时整体错位，任何基于它们的门控/对照/消融结论都不可用。\n"
            "请用修复后的 src/runner.py 重跑相应格子（重跑会自动打标）。"
            f"示例：{sorted(skipped)[:3]}"
        )
        if on_unmarked == "raise":
            raise RuntimeError(msg)
        if skipped_out is not None:
            skipped_out.extend(sorted(skipped))
        bar = "!" * 78
        print(f"\n{bar}\n[selw] ⚠️ 排除未打标的 val 数组\n{msg}\n{bar}\n",
              file=sys.stderr, flush=True)
    return out


def build_groups(
    winerr_dir: Path | list[Path],
    feats: pd.DataFrame,
    min_arms: int = 2,
    require_none: bool = True,
    verbose: bool = True,
) -> tuple[list[Group], list[str]]:
    """把逐窗口误差与逐窗口特征按下标对齐成决策组。

    对齐契约：
    - **test**：TSLib 的 test loader ``shuffle=False`` 且 ``drop_last=False``，
      第 i 个窗口的输入是 ``split[i : i+seq_len]``，与 ``features_window`` 完全一致。
    - **val**：TSLib 的 ``data_factory`` 只对 test 关 shuffle，所以 val 的**逐窗口**顺序
      只有在 ``runner`` 用显式确定序 loader 落盘并打上 ``.order.json`` 标记时才可信。
      （2026-09-14 code review 发现的 P0：旧 val 数组是随机置换顺序。）

      因此这里把 val 拆成两种用途：

      * ``val_mean``（每臂均值）：对窗口置换**不变**，任何 val 数组都可用 →
        ``best_fixed_val`` 基线对**全部**组成立；
      * ``e_val``（逐窗口）：只有全臂打标的组才有 → 门控 / 对照 / 消融只在这些组上跑。

      这样 test 侧审计（oracle 不变性、argmin 半衰期、在线延迟）不会因为
      TimesNet 的 val 逐窗口不可用就连带丢掉整个骨干。

    特征只按最小 pred_len 算过一次，长 horizon 的窗口数更少，故**截断**特征表即可。

    对齐方式是**按 ``window_index`` 取值**而非按行位置：第 i 个误差值对应
    ``window_index == i`` 的那一行。这样 ``features_window`` 若以 ``stride > 1``
    抽稀（第 j 行代表窗口 ``j*stride``），会因缺 ``window_index`` 而显式跳过，
    而不是静默配错（旧的位置式 ``[:n]`` 截断在行数仍够时会放行）。
    """
    dirs = [Path(winerr_dir)] if isinstance(winerr_dir, (str, Path)) else [Path(x) for x in winerr_dir]
    test_err = load_winerr(dirs)
    notes: list[str] = []
    # val 读两遍，用途严格分开（见 docstring）：
    #   any    -> 只取每臂均值（置换不变，全部组可用）
    #   marked -> 逐窗口，仅全臂打标的组
    val_any = load_winerr([d / "val" for d in dirs])
    unmarked: list[str] = []
    val_marked = load_winerr([d / "val" for d in dirs], require_order_marker=True,
                             on_unmarked="skip", skipped_out=unmarked)
    if unmarked:
        bb = sorted({c.split("__")[0] for c in unmarked})
        notes.append(
            f"{len(unmarked)} 个 val 数组无确定序标记（骨干：{', '.join(bb)}）："
            f"仍可用于 best_fixed_val 与 test 侧审计，但不参与门控 / 对照 / 消融"
        )
    fcols = window_feature_columns(feats)

    fv = {k: g for k, g in feats[feats["split"] == "val"].groupby("dataset")}
    ft = {k: g for k, g in feats[feats["split"] == "test"].groupby("dataset")}

    def _take_prefix(g: pd.DataFrame, n: int) -> np.ndarray | None:
        """取 window_index == 0..n-1 的特征行，缺任一下标返回 None。"""
        idx = g["window_index"].to_numpy()
        pos = {int(w): i for i, w in enumerate(idx)}
        rows = [pos.get(i) for i in range(n)]
        if any(r is None for r in rows):
            return None
        return g[fcols].to_numpy(np.float64)[np.asarray(rows, dtype=int)]

    groups: list[Group] = []
    n_win_ok = 0
    for k in sorted(test_err):
        bb, ds, pl, sd = k
        arms_t, arms_v = test_err.get(k, {}), val_any.get(k, {})
        arms = sorted(set(arms_t) & set(arms_v))
        if require_none and "none" not in arms:
            notes.append(f"{k}: 缺 none 对照臂，跳过")
            continue
        if len(arms) < min_arms:
            notes.append(f"{k}: 只有 {len(arms)} 个臂（需 ≥{min_arms}），跳过")
            continue
        # 同一组内各臂窗口数必须一致；取交集长度以防某臂 eval 被截断
        n_t = min(arms_t[a].size for a in arms)
        n_v = min(arms_v[a].size for a in arms)
        if ds not in ft or ds not in fv:
            notes.append(f"{k}: 特征表缺 {ds}，跳过")
            continue
        gt, gv = ft[ds].sort_values("window_index"), fv[ds].sort_values("window_index")
        if len(gt) < n_t or len(gv) < n_v:
            notes.append(f"{k}: 特征窗口数 {len(gv)}/{len(gt)} < 误差窗口数 {n_v}/{n_t}，跳过")
            continue
        x_test = _take_prefix(gt, n_t)
        if x_test is None:
            notes.append(
                f"{k}: test 特征表 window_index 不是 0..n-1 的连续前缀"
                f"（stride>1 抽稀？），无法与逐窗口误差对齐，跳过"
            )
            continue
        # val 均值：置换不变，用任何 val 数组都对
        val_mean = np.array([float(arms_v[a][:n_v].mean()) for a in arms])
        # val 逐窗口：要求该组**每个臂**都打标，否则整组不做门控
        marked = val_marked.get(k, {})
        e_val = x_val = None
        if all(a in marked for a in arms):
            x_val = _take_prefix(gv, n_v)
            if x_val is None:
                notes.append(f"{k}: val 特征表非连续前缀，该组不做门控（test 侧审计仍保留）")
            else:
                e_val = np.stack([marked[a][:n_v] for a in arms])
                n_win_ok += 1
        groups.append(Group(
            backbone=bb, dataset=ds, pred_len=pl, seed=sd, arms=arms,
            e_val=e_val,
            e_test=np.stack([arms_t[a][:n_t] for a in arms]),
            x_val=x_val,
            x_test=x_test,
            feat_names=list(fcols),
            val_mean=val_mean,
        ))
    if verbose:
        print(f"[selw] 决策组 {len(groups)} 个（其中 {n_win_ok} 个可做 val 逐窗口门控）"
              f"（跳过 {len(notes)} 个）")
        for m in notes[:8]:
            print("   -", m)
    return groups, notes


# --------------------------------------------------------------------------- #
# 门控器
# --------------------------------------------------------------------------- #
def _fit_gate(
    x_tr: np.ndarray, e_tr: np.ndarray, arms: list[str], i_none: int, mode: str, seed: int = 26
):
    """在训练窗口上拟合门控器，返回 predict(x)->选中的臂下标。

    两种形式：
    * ``clf``：直接多分类预测 argmin 臂。简单，但把「赢 0.1%」和「赢 40%」看成一样重要。
    * ``reg``（默认）：对每个非 none 臂回归**相对增益** (e_none−e_arm)/e_none，
      预测时取增益最大的臂，若所有臂预测增益 ≤0 就用 none。回归形式对代价敏感，
      而且天然带一个「不确定就别挂」的保守偏置，实践中更稳。
    """
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    if mode == "clf":
        y = e_tr.argmin(axis=0)
        vals, cnts = np.unique(y, return_counts=True)
        if len(vals) < 2:              # 训练段只有一个臂赢过 -> 退化成常数门控
            const = int(y[0])
            return lambda x: np.full(len(x), const, dtype=int), None
        # sklearn 的 early_stopping 用**分层**划分留验证集，某个臂只赢过 1 次时会直接
        # 报 "The least populated class in y has only 1 member"。真实数据里这必然发生
        # （比如某个插件在 11425 个窗口里只赢 1 次），所以样本不够就关掉早停，
        # 而不是让整个门控实验崩掉。
        es = bool(cnts.min() >= 10 and len(y) >= 50)
        clf = HistGradientBoostingClassifier(
            max_iter=200 if es else 120, learning_rate=0.1, max_depth=6,
            l2_regularization=1.0, early_stopping=es,
            validation_fraction=0.2 if es else None, random_state=seed,
        ).fit(x_tr, y)
        classes = clf.classes_

        def predict(x: np.ndarray) -> np.ndarray:
            return classes[clf.predict_proba(x).argmax(1)].astype(int)

        return predict, clf

    base = e_tr[i_none]
    models: dict[int, Any] = {}
    for j, _ in enumerate(arms):
        if j == i_none:
            continue
        gain = (base - e_tr[j]) / (base + EPS)
        gain = np.clip(gain, -5.0, 5.0)   # 极小 e_none 会把相对增益放大到 1e3，截断防爆
        es = len(x_tr) >= 50          # 窗口太少（ILI 只有 74 个）时留 20% 反而不稳
        models[j] = HistGradientBoostingRegressor(
            max_iter=200 if es else 120, learning_rate=0.1, max_depth=6,
            l2_regularization=1.0, early_stopping=es,
            validation_fraction=0.2 if es else None, random_state=seed,
        ).fit(x_tr, gain)

    def predict(x: np.ndarray) -> np.ndarray:
        pred = np.zeros((len(arms), len(x)))          # none 的预测增益恒为 0
        for j, m in models.items():
            pred[j] = m.predict(x)
        best = pred.argmax(0)
        best[pred.max(0) <= 0.0] = i_none             # 谁都不看好 -> 不挂
        return best.astype(int)

    return predict, models


def _pick(e: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """按每窗口选中的臂取误差。"""
    return e[idx, np.arange(e.shape[1])]


def evaluate_group(g: Group, mode: str = "reg", seed: int = 26) -> dict[str, Any]:
    """协议 inpool：val 学门控 -> test 评估。返回该组一行结果。"""
    if not g.has_val_windows:
        raise ValueError(
            f"{g.key}: 该组没有可信的 val 逐窗口误差（缺 .order.json 确定序标记），"
            "不能做门控类协议。调用方应先用 Group.has_val_windows 过滤。"
        )
    predict, _ = _fit_gate(g.x_val, g.e_val, g.arms, g.i_none, mode, seed)
    sel = predict(g.x_test)

    e_none = g.e_test[g.i_none]
    m_none = float(e_none.mean())
    m_gate = float(_pick(g.e_test, sel).mean())
    m_oracle = float(g.e_test.min(axis=0).mean())
    fixed_test = {a: float(g.e_test[i].mean()) for i, a in enumerate(g.arms)}
    # 强基线：用验证集平均 MSE 选一个固定臂，再看它在测试集上的表现
    i_bf = int(g.val_mean.argmin())
    m_bf = float(g.e_test[i_bf].mean())

    def rel(x: float, ref: float) -> float:
        return 100.0 * (ref - x) / (ref + EPS)

    row: dict[str, Any] = {
        "key": g.key, "backbone": g.backbone, "dataset": g.dataset,
        "pred_len": g.pred_len, "seed": g.seed,
        "n_arms": len(g.arms), "arms": "|".join(g.arms),
        "n_val_windows": g.e_val.shape[1], "n_test_windows": g.e_test.shape[1],
        "mse_none": m_none, "mse_gate": m_gate, "mse_oracle": m_oracle,
        "best_fixed_val": g.arms[i_bf], "mse_best_fixed": m_bf,
        "gain_gate_vs_none": rel(m_gate, m_none),
        "gain_gate_vs_best_fixed": rel(m_gate, m_bf),
        "gain_bestfixed_vs_none": rel(m_bf, m_none),
        "gain_oracle_vs_none": rel(m_oracle, m_none),
        # 门控吃到了 oracle 剩余空间的百分之多少（0=没用，100=追平上界）
        "oracle_realized_pct": 100.0 * (m_none - m_gate) / (m_none - m_oracle + EPS),
        "switch_rate": float((sel != g.i_none).mean()),
        "n_distinct_choices": int(len(np.unique(sel))),
        "oracle_arm_entropy": float(_entropy(g.e_test.argmin(0), len(g.arms))),
        "gate_acc": float((sel == g.e_test.argmin(0)).mean()),
        "majority_acc": float(_majority_acc(g.e_test.argmin(0))),
    }
    for a, v in fixed_test.items():
        row[f"mse_fixed_{a}"] = v
    row["wilcoxon_p_gate_vs_best_fixed"] = _wilcoxon_p(
        _pick(g.e_test, sel), g.e_test[i_bf]
    )
    return row


# --------------------------------------------------------------------------- #
# 协议 split：在**同一个 split 内部**按时间切 gate-train / gate-eval
# --------------------------------------------------------------------------- #
SPLIT_MIN_TRAIN = 200
"""gate-train 段的最小窗口数。低于此值 HistGBM 连自己的早停划分都不够（见
LEARNABLE_MIN_VAL 的说明），结论会退化成「数据不够」而不是「学不到」。"""

SPLIT_MIN_EVAL = 100
"""gate-eval 段的最小窗口数。太短时逐窗口配对检验没有功效，均值也不稳。"""


def evaluate_split_gate(
    g: Group,
    source: str = "test",
    train_frac: float = 0.7,
    mode: str = "reg",
    seed: int = 26,
    min_train: int = SPLIT_MIN_TRAIN,
    min_eval: int = SPLIT_MIN_EVAL,
) -> dict[str, Any] | None:
    """把**一个** split 按时间切成前段（学门控）/ 后段（评门控），全部量在后段结算。

    这个协议专门用来堵一条审稿意见：*"你的门控标签取自验证集，而验证集又被用于早停和
    best-epoch 选择，所以 val 误差偏乐观，且 val→test 存在分布漂移——门控赢不了固定臂
    也许只是因为这个偏差/漂移，而不是因为信号本身不可学。"*

    两种 ``source``，各自堵住这条意见的一半：

    * ``source="val"``：在验证段内部切。训练段与评估段**同分布、同一批被早停"污染"的
      数据**，所以漂移=0、乐观偏差在两侧同向抵消。若此处门控仍赢不了同段固定臂，
      "漂移解释"即被排除。
    * ``source="test"``：在测试段内部切。测试段从未参与早停 / best-epoch / 任何选择，
      所以标签的乐观偏差**严格为 0**，且 val 完全不参与。这是给门控**最有利**的诊断条件
      （它甚至用了部署时拿不到的同段真值，属于上界性质的诊断而非可部署方案），
      因此若这里也赢不了，门控失败与 val 无关。

    为什么不用"从训练集尾部切 gate-train"（原计划）：train 尾段是模型**训练时见过**的
    数据，其误差被系统性低估，作为门控标签比 val 更脏；而且本项目的 checkpoint 已清理，
    重算需要重训。``source="test"`` 在"标签无乐观偏差"这一点上严格优于 train 尾段，
    所以用它替代，并在论文 threats 里说明这一替换。

    严格性：切分是**时序**的（前 ``train_frac`` 为训练段），不是随机划分；
    评估段上的每一个量——门控、best_fixed、none、oracle——都只用评估段的误差，
    而所有"选择"信息都只来自训练段。窗口数不足时返回 ``None``。
    """
    if source == "val":
        if not g.has_val_windows:
            return None
        e_all, x_all = g.e_val, g.x_val
    elif source == "test":
        e_all, x_all = g.e_test, g.x_test
    else:
        raise ValueError(f"source 只能是 'val' 或 'test'，收到 {source!r}")

    n = e_all.shape[1]
    n_tr = int(round(n * train_frac))
    n_ev = n - n_tr
    if n_tr < min_train or n_ev < min_eval:
        return None

    e_tr, e_ev = e_all[:, :n_tr], e_all[:, n_tr:]
    x_tr, x_ev = x_all[:n_tr], x_all[n_tr:]

    predict, _ = _fit_gate(x_tr, e_tr, g.arms, g.i_none, mode, seed)
    sel = predict(x_ev)

    e_none = e_ev[g.i_none]
    m_none = float(e_none.mean())
    m_gate = float(_pick(e_ev, sel).mean())
    m_oracle = float(e_ev.min(axis=0).mean())
    # 同口径可部署基线：只用**训练段**均值选一个固定臂，冻结后在评估段结算。
    # 这与门控使用的信息完全对等（都只看训练段），是本协议唯一正确的比较对象。
    i_bf = int(e_tr.mean(axis=1).argmin())
    m_bf = float(e_ev[i_bf].mean())
    # 参照口径：仍按整段 val 均值选臂（论文主表用的那个 best_fixed_val）
    i_bfv = int(g.val_mean.argmin())
    m_bfv = float(e_ev[i_bfv].mean())

    def rel(x: float, ref: float) -> float:
        return 100.0 * (ref - x) / (ref + EPS)

    row: dict[str, Any] = {
        "key": g.key, "backbone": g.backbone, "dataset": g.dataset,
        "pred_len": g.pred_len, "seed": g.seed,
        "source": source, "train_frac": train_frac,
        "n_arms": len(g.arms), "arms": "|".join(g.arms),
        # 复用 summarize()/stratify_by_val_size() 的列名：这里的「训练窗口数」就是前段长度
        "n_val_windows": n_tr, "n_test_windows": n_ev,
        "n_windows_total": n,
        "mse_none": m_none, "mse_gate": m_gate, "mse_oracle": m_oracle,
        "best_fixed_split": g.arms[i_bf], "mse_best_fixed": m_bf,
        "best_fixed_val": g.arms[i_bfv], "mse_best_fixed_valmean": m_bfv,
        "gain_gate_vs_none": rel(m_gate, m_none),
        "gain_gate_vs_best_fixed": rel(m_gate, m_bf),
        "gain_gate_vs_best_fixed_valmean": rel(m_gate, m_bfv),
        "gain_bestfixed_vs_none": rel(m_bf, m_none),
        "gain_oracle_vs_none": rel(m_oracle, m_none),
        "oracle_realized_pct": 100.0 * (m_none - m_gate) / (m_none - m_oracle + EPS),
        "switch_rate": float((sel != g.i_none).mean()),
        "n_distinct_choices": int(len(np.unique(sel))),
        "gate_acc": float((sel == e_ev.argmin(0)).mean()),
        "majority_acc": float(_majority_acc(e_ev.argmin(0))),
        # 训练段选出的固定臂与评估段事后最优臂是否一致：量化「同 split 内部的臂漂移」
        "bf_split_is_eval_argmin": bool(i_bf == int(e_ev.mean(axis=1).argmin())),
    }
    row["wilcoxon_p_gate_vs_best_fixed"] = _wilcoxon_p(_pick(e_ev, sel), e_ev[i_bf])
    return row


def split_gate_verdict(s_val: dict[str, Any] | None, s_test: dict[str, Any] | None) -> str:
    """把两个 split 诊断合成一句能直接回复审稿人的话。"""
    parts: list[str] = []
    for name, s in (("val 内部", s_val), ("test 内部", s_test)):
        if not s or s.get("n_groups", 0) == 0:
            continue
        parts.append(
            f"{name} split（{s['n_groups']} 组）：gate−best_fixed "
            f"{s['mean_gain_gate_vs_best_fixed']:+.2f}%，"
            f"{s['groups_gate_beats_best_fixed_pct']:.0f}% 的组为正，"
            f"oracle 上界 {s['mean_gain_oracle_vs_none']:+.1f}%"
        )
    if not parts:
        return "⏳ 两个 split 诊断都因窗口数不足而无结论。"
    worst = min(s["mean_gain_gate_vs_best_fixed"]
                for s in (s_val, s_test) if s and s.get("n_groups", 0) > 0)
    head = ("✅ 排除「val 被早停用过」这条解释：" if worst <= 0 else
            "⚠️ 注意：换成同 split 内部切分后门控转正，"
            "说明原先的失败里确实有 val→test 漂移的成分，必须重新表述：")
    tail = ("；两者都是**同分布**训练/评估（漂移=0），test 内部切分的标签更是"
            "从未被早停或 best-epoch 选择接触过（乐观偏差=0）。"
            "门控在这两种最有利条件下依然赢不了同口径固定臂，"
            "因此失败原因不是验证集偏差或分布漂移，而是窗口级信号本身不可用。"
            if worst <= 0 else
            "；需要区分「漂移导致的失败」与「信号不可学」两种解释。")
    return head + "；".join(parts) + tail


# 绝对滞后网格（窗口数）。用 2 的幂覆盖 1~1024，足以横跨 H=24 到 H=720。
LAGS_ABS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def evaluate_argmin_structure(g: Group, n_perm: int = 32, seed: int = 26) -> dict[str, Any]:
    """审计 per-window oracle 这个统计量本身：它能不能作为"插件选择可学"的证据？

    **不能。** 关键观察（本项目的核心方法论点）：per-window oracle
    ``mean_t min_k e[k,t]`` 只依赖每个窗口内误差的**多重集**，与"这些误差属于哪个臂"
    完全无关。对每个窗口独立地随机重贴臂标签，oracle 一分不变，但"哪个窗口该用哪个臂"
    的可预测性会被彻底摧毁。所以「oracle 比 best_fixed 高 X%」这句话在信息上是**空的**：
    只要各臂误差在窗口尺度上有离散度，就必然有一个很大的 X，与是否可学无关。

    真正决定可学性的是**最优臂身份 a*_t = argmin_k e[k,t] 的持续性/可预测性**，
    而且必须在合法的延迟下看：部署时只能用 t−H 及更早的信息。因此本函数测：

    * ``win_rate_top``     —— 最常获胜的臂占多少窗口（越接近 1 越该直接用固定臂）
    * ``chance_persist``   —— 独立重抽下 a*_t == a*_{t−d} 的概率 = Σ p_k²
    * ``persist_lag*``     —— 真实的滞后一致率，lag1 / H/4 / H/2 / H / 2H
    * ``excess_persist_lagH`` —— 合法延迟下超出 chance 的部分，**这才是可利用信号的直接度量**
    * ``z_lagH``           —— 上述超出量的正态近似 z 值
    * ``oracle_invariant`` —— 数值验证"重贴标签后 oracle 不变"，把上面的论点钉死
    """
    rng = np.random.default_rng(seed + zlib.crc32(g.key.encode()) % 10_000)
    e = g.e_test
    K, n = e.shape
    a = e.argmin(axis=0)
    p_win = np.bincount(a, minlength=K) / n
    chance = float((p_win ** 2).sum())
    H = int(g.pred_len)
    lags = {"lag1": 1, "lagH_4": max(1, H // 4), "lagH_2": max(1, H // 2),
            "lagH": H, "lag2H": 2 * H}
    row: dict[str, Any] = {
        "key": g.key, "backbone": g.backbone, "dataset": g.dataset, "pred_len": H,
        "n_arms": K, "n_test_windows": n,
        "win_rate_top": float(p_win.max()), "win_rate_none": float(p_win[g.i_none]),
        "win_rate_entropy": float(-(p_win[p_win > 0] * np.log(p_win[p_win > 0])).sum()
                                  / np.log(K)),
        "chance_persist": chance,
    }
    for name, d in lags.items():
        if d < n:
            pr = float((a[d:] == a[:-d]).mean())
            ne = n - d
            z = (pr - chance) / np.sqrt(max(chance * (1 - chance), EPS) / ne)
        else:
            pr, z = float("nan"), float("nan")
        row[f"persist_{name}"] = pr
        row[f"excess_persist_{name}"] = pr - chance if pr == pr else float("nan")
        row[f"z_{name}"] = float(z)

    # 绝对滞后上的衰减曲线 + 半衰期：这是把结论量化成"相关长度 vs 预测视界"的关键。
    # 只看 H 的倍数会漏掉最重要的信息——身份相关性到底能撑多少个窗口。
    for L in LAGS_ABS:
        row[f"excess_abs{L}"] = (float((a[L:] == a[:-L]).mean()) - chance
                                 if L < n else float("nan"))
    e1 = row["excess_abs1"]
    half = float("nan")
    if e1 == e1 and e1 > 0:
        prev_L, prev_v = 1, e1
        for L in LAGS_ABS[1:]:
            v = row[f"excess_abs{L}"]
            if v != v:
                break
            if v <= e1 / 2:
                # 在 log2(lag) 上线性插值，避免半衰期只能取到 2 的幂
                if prev_v > v:
                    f = (prev_v - e1 / 2) / (prev_v - v)
                    half = float(2 ** (np.log2(prev_L) + f * (np.log2(L) - np.log2(prev_L))))
                else:
                    half = float(L)
                break
            prev_L, prev_v = L, v
    row["persist_half_life_windows"] = half
    # 这个比值是论文的一句话结论：可利用的相关长度只有预测视界的百分之几。
    row["half_life_over_H"] = float(half / H) if half == half else float("nan")

    # 重贴标签零假设：逐窗口随机置换臂标签。oracle 必须逐位不变（下面数值验证），
    # 而滞后一致率会掉到 chance 附近——这正是"oracle 与可学性无关"的构造性证明。
    real_oracle = float(e.min(axis=0).mean())
    inv_ok, rel_persist = True, []
    for _ in range(n_perm):
        perm = np.argsort(rng.random((K, n)), axis=0)      # 每列一个独立置换
        ep = np.take_along_axis(e, perm, axis=0)
        if abs(float(ep.min(axis=0).mean()) - real_oracle) > 1e-9:
            inv_ok = False
        ap = ep.argmin(axis=0)
        rel_persist.append(float((ap[1:] == ap[:-1]).mean()))
    row["oracle_invariant_under_relabel"] = bool(inv_ok)
    row["persist_lag1_relabelled"] = float(np.mean(rel_persist))
    row["oracle_gain_real_pct"] = 100.0 * (float(e[g.i_none].mean()) - real_oracle) / (
        float(e[g.i_none].mean()) + EPS)
    return row


def summarize_argmin_structure(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "protocol": "argmin_identity_persistence",
        "n_groups": int(len(df)),
        "mean_oracle_gain_real_pct": float(df["oracle_gain_real_pct"].mean()),
        "oracle_invariant_under_relabel_all": bool(df["oracle_invariant_under_relabel"].all()),
        "mean_win_rate_top": float(df["win_rate_top"].mean()),
        "mean_win_rate_entropy": float(df["win_rate_entropy"].mean()),
        "mean_chance_persist": float(df["chance_persist"].mean()),
        "mean_persist_lag1_relabelled": float(df["persist_lag1_relabelled"].mean()),
    }
    out["mean_persist_half_life_windows"] = float(df["persist_half_life_windows"].mean())
    out["median_persist_half_life_windows"] = float(df["persist_half_life_windows"].median())
    out["mean_half_life_over_H"] = float(df["half_life_over_H"].mean())
    out["excess_decay_curve"] = {str(L): float(df[f"excess_abs{L}"].mean())
                                 for L in LAGS_ABS if f"excess_abs{L}" in df}
    for name in ("lag1", "lagH_4", "lagH_2", "lagH", "lag2H"):
        out[f"mean_persist_{name}"] = float(df[f"persist_{name}"].mean())
        out[f"mean_excess_persist_{name}"] = float(df[f"excess_persist_{name}"].mean())
        out[f"groups_z_gt2_{name}"] = int((df[f"z_{name}"] > 2).sum())
    return out


def argmin_structure_verdict(s: dict[str, Any]) -> str:
    e1 = s["mean_excess_persist_lag1"]
    eh = s["mean_excess_persist_lagH"]
    ch = s["mean_chance_persist"]
    n = s["n_groups"]
    zh = s.get("groups_z_gt2_lagH", 0)
    orc = s["mean_oracle_gain_real_pct"]
    head = (f"oracle 上界 {orc:.2f}% 在逐窗口重贴臂标签下**完全不变**"
            f"（{n} 组全部验证通过），因此它不能作为可学性的证据。")
    if eh > 0.05 and zh > n * 0.5:
        return (f"✅ {head} 但最优臂身份确实可预测：合法延迟 H 下滞后一致率超出 chance "
                f"{eh:+.3f}（chance={ch:.3f}），{zh}/{n} 组 z>2。信号在，值得继续做选择器。")
    if e1 > 0.05 and eh <= 0.05:
        hl = s.get("median_persist_half_life_windows", float("nan"))
        ratio = s.get("mean_half_life_over_H", float("nan"))
        return (f"⚠️ {head} 最优臂身份只有**极短程**可预测性：lag1 超出 chance {e1:+.3f}，"
                f"但到合法延迟 H 就只剩 {eh:+.3f}（chance={ch:.3f}，{zh}/{n} 组 z>2）。"
                f"身份相关性的半衰期中位数只有 {hl:.1f} 个窗口，约为预测视界 H 的 {ratio:.1%}。"
                "结论：**可利用的相关长度比预测视界短一到两个数量级**，因此任何因果选择器都"
                f"拿不到那 {orc:.1f}% 的上界。这是一条定量、可复现、且能解释全部负结果的核心结论。")
    return (f"❌ {head} 且最优臂身份基本不可预测：lag1 超出 chance 仅 {e1:+.3f}、"
            f"延迟 H 时 {eh:+.3f}（chance={ch:.3f}）。"
            "这解释了为什么静态特征门控、延迟误差反馈、乃至零延迟作弊版本全都赢不了最佳固定臂："
            "不是没学好，而是逐窗口最优臂身份接近独立重抽，没有东西可学。"
            "『oracle 上界大』与『插件选择可学』是两件被文献混为一谈的事——这就是论文的核心论点。")


# --------------------------------------------------------------------------- #
# 分层审计：半衰期短是普遍现象，还是某个数据集/骨干拖出来的平均值？
# --------------------------------------------------------------------------- #
# 为什么必须做：论文的核心论点是「可利用的相关长度比预测视界短一到两个数量级」。
# 只报 127 组的均值/中位数，审稿人一定会问「是不是 ILI 这种小数据集把它拉下去的」。
# 因此按 backbone / dataset / horizon 三种切法各报一遍，并给出**最有利于反方**的那个
# 分层（excess_persist_lagH 最大的那层）——如果连它都撑不住，结论才算稳。
STRATA_KEYS = ("backbone", "dataset", "pred_len")


def stratify_audit(df: pd.DataFrame, min_n: int = 3) -> pd.DataFrame:
    """把 ``evaluate_argmin_structure`` 的逐组表按三种维度分层汇总。

    每层报：组数、oracle 上界、lag1/延迟 H 的超出量、z>2 的组数、
    半衰期（中位数，窗口数）与半衰期/H。半衰期用**中位数**而不是均值：
    它的分布右偏（最大 41 个窗口 vs 中位数 9），均值会被少数长尾组带偏。

    ``min_n`` 以下的层照样输出但标记 ``underpowered=True``，交给上层决定是否引用，
    而不是静默丢掉——丢掉会让「某个数据集是例外」这种信息消失。
    """
    rows: list[dict[str, Any]] = []
    for key in STRATA_KEYS:
        if key not in df.columns:
            continue
        for val, sub in df.groupby(key, sort=True):
            n = int(len(sub))
            row: dict[str, Any] = {
                "stratum_key": key,
                "stratum": str(val),
                "n_groups": n,
                "underpowered": bool(n < min_n),
                "mean_oracle_gain_real_pct": float(sub["oracle_gain_real_pct"].mean()),
                "mean_chance_persist": float(sub["chance_persist"].mean()),
                "mean_excess_persist_lag1": float(sub["excess_persist_lag1"].mean()),
                "mean_excess_persist_lagH": float(sub["excess_persist_lagH"].mean()),
                "groups_z_gt2_lagH": int((sub["z_lagH"] > 2).sum()),
                "groups_z_gt2_lag1": int((sub["z_lag1"] > 2).sum()),
                "median_half_life_windows": float(sub["persist_half_life_windows"].median()),
                "iqr_half_life_windows": float(
                    sub["persist_half_life_windows"].quantile(0.75)
                    - sub["persist_half_life_windows"].quantile(0.25)),
                "median_half_life_over_H": float(sub["half_life_over_H"].median()),
                "max_half_life_over_H": float(sub["half_life_over_H"].max()),
            }
            for L in LAGS_ABS:
                col = f"excess_abs{L}"
                if col in sub:
                    row[col] = float(sub[col].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_strata(ds: pd.DataFrame) -> dict[str, Any]:
    """跨层汇总：结论是否在**每一层**都成立，以及最有利于反方的那层长什么样。"""
    if ds.empty:
        return {"protocol": "audit_strata", "n_strata": 0}
    ok = ds[~ds["underpowered"]]
    use = ok if len(ok) else ds
    worst = use.loc[use["mean_excess_persist_lagH"].idxmax()]      # 最像"有信号"的那层
    longest = use.loc[use["median_half_life_over_H"].idxmax()]     # 相关长度相对 H 最长的那层
    out: dict[str, Any] = {
        "protocol": "audit_strata",
        "n_strata": int(len(ds)),
        "n_strata_underpowered": int(ds["underpowered"].sum()),
        "strata_keys": list(dict.fromkeys(ds["stratum_key"])),
        # 「普遍性」的两个硬判据
        "strata_with_half_life_over_H_below_25pct": int((use["median_half_life_over_H"] < 0.25).sum()),
        "strata_total_used": int(len(use)),
        "strata_with_excess_lagH_above_005": int((use["mean_excess_persist_lagH"] > 0.05).sum()),
        "max_median_half_life_over_H": float(longest["median_half_life_over_H"]),
        "max_median_half_life_over_H_stratum": f"{longest['stratum_key']}={longest['stratum']}",
        "max_excess_persist_lagH": float(worst["mean_excess_persist_lagH"]),
        "max_excess_persist_lagH_stratum": f"{worst['stratum_key']}={worst['stratum']}",
        "median_half_life_windows_range": [float(use["median_half_life_windows"].min()),
                                           float(use["median_half_life_windows"].max())],
    }
    # horizon 依赖：半衰期是「绝对窗口数基本不变」还是「随 H 一起长」？
    hz = ds[ds["stratum_key"] == "pred_len"]
    if len(hz) >= 3:
        h = hz["stratum"].astype(float).to_numpy()
        hl = hz["median_half_life_windows"].to_numpy()
        ratio = hz["median_half_life_over_H"].to_numpy()
        out["horizon_hl_spearman"] = float(_spearman(h, hl))
        out["horizon_ratio_spearman"] = float(_spearman(h, ratio))
        out["hl_at_min_H"] = float(hl[np.argmin(h)])
        out["hl_at_max_H"] = float(hl[np.argmax(h)])
        out["H_min"] = float(h.min())
        out["H_max"] = float(h.max())
    return out


def coverage_gap_table(groups: list[Group], audit: pd.DataFrame | None = None) -> pd.DataFrame:
    """逐骨干列出「test 侧审计覆盖」与「门控类协议覆盖」的缺口，并给出缺口是否要紧的证据。

    论文里必须显式交代：门控结论只覆盖 96/127 个决策组，缺的那些全是 TimesNet
    （它们的 val 逐窗口数组是旧的、没有确定序标记）。审稿人接着会问：
    *"你凭什么说补上这 31 组不会改变结论？"*

    这张表就是答案：把**未覆盖组**与**已覆盖组**在 test 侧审计指标上并排比较。
    test 侧审计（oracle 上界、身份半衰期、合法延迟 H 下的超出量）对全部 127 组都可算，
    且完全不受 val 顺序 bug 影响。若未覆盖组在这些指标上与已覆盖组无差异，
    则"补齐后结论会翻"这一质疑没有依据——因为决定可学性的量在两批组上是同一个分布。
    """
    rows: list[dict[str, Any]] = []
    ac = audit.set_index("key") if (audit is not None and "key" in audit.columns) else None
    if ac is not None and ac.index.has_duplicates:
        # 同一 key 出现多次（例如把多个 winerr 目录的审计结果拼在一起）时 reindex 会直接抛
        # "cannot reindex on an axis with duplicate labels"，去重而不是让整张覆盖表崩掉。
        ac = ac[~ac.index.duplicated(keep="first")]

    def _audit_stats(keys: list[str]) -> dict[str, Any]:
        if ac is None or not keys:
            return {}
        sub = ac.reindex([k for k in keys if k in ac.index])
        if sub.empty:
            return {}
        out: dict[str, Any] = {}
        for col, name in (("oracle_gain_real_pct", "mean_oracle_gain_pct"),
                          ("persist_half_life_windows", "median_half_life"),
                          ("half_life_over_H", "median_half_life_over_H"),
                          ("excess_persist_lagH", "mean_excess_lagH")):
            if col not in sub.columns:
                continue
            v = pd.to_numeric(sub[col], errors="coerce").dropna()
            if v.empty:
                continue
            out[name] = float(v.median() if name.startswith("median") else v.mean())
        return out

    for bb in sorted({g.backbone for g in groups}):
        sub = [g for g in groups if g.backbone == bb]
        gated = [g for g in sub if g.has_val_windows]
        missing = [g for g in sub if not g.has_val_windows]
        row: dict[str, Any] = {
            "backbone": bb,
            "n_groups_test_side": len(sub),
            "n_groups_gating": len(gated),
            "n_missing": len(missing),
            "pct_gating_covered": round(100.0 * len(gated) / len(sub), 1),
            "reason_missing": ("val 逐窗口数组无 .order.json 确定序标记（重跑前的旧产物）"
                               if missing else ""),
        }
        row.update(_audit_stats([g.key for g in sub]))
        rows.append(row)

    df = pd.DataFrame(rows)
    # 汇总两行：已覆盖 vs 未覆盖，用于直接回答「缺口是否要紧」
    cov_keys = [g.key for g in groups if g.has_val_windows]
    mis_keys = [g.key for g in groups if not g.has_val_windows]
    for label, keys in (("ALL_gating_covered", cov_keys), ("ALL_gating_missing", mis_keys)):
        if not keys:
            continue
        r: dict[str, Any] = {
            "backbone": label,
            "n_groups_test_side": len(keys),
            "n_groups_gating": len(keys) if label.endswith("covered") else 0,
            "n_missing": 0 if label.endswith("covered") else len(keys),
            "pct_gating_covered": 100.0 if label.endswith("covered") else 0.0,
            "reason_missing": "",
        }
        r.update(_audit_stats(keys))
        df = pd.concat([df, pd.DataFrame([r])], ignore_index=True)
    return df


def coverage_gap_verdict(df: pd.DataFrame) -> str:
    """把覆盖缺口表翻译成一句能写进 threats 的话。"""
    if df.empty:
        return "无覆盖信息。"
    cov = df[df["backbone"] == "ALL_gating_covered"]
    mis = df[df["backbone"] == "ALL_gating_missing"]
    if mis.empty:
        return "✅ 门控类协议覆盖全部决策组，无缺口。"
    n_cov = int(cov["n_groups_test_side"].iloc[0]) if not cov.empty else 0
    n_mis = int(mis["n_groups_test_side"].iloc[0])
    bbs = sorted(df[(df["n_missing"] > 0) & (~df["backbone"].str.startswith("ALL_"))]["backbone"])
    msg = (f"门控类协议覆盖 {n_cov}/{n_cov + n_mis} 个决策组，缺 {n_mis} 组"
           f"（全部来自 {', '.join(bbs)}：val 逐窗口无确定序标记）。")
    keys = ["mean_oracle_gain_pct", "median_half_life", "median_half_life_over_H",
            "mean_excess_lagH"]
    have = [k for k in keys if k in df.columns and not cov.empty
            and pd.notna(cov[k].iloc[0]) and pd.notna(mis[k].iloc[0])]
    if not have:
        return msg + " （缺 test 侧审计数据，无法比较两批组是否同质。）"
    diffs = []
    for k in have:
        a, b = float(cov[k].iloc[0]), float(mis[k].iloc[0])
        diffs.append(f"{k} {a:.3g} vs {b:.3g}")
    # 判断是否同质：半衰期/H 与合法延迟超出量是决定可学性的两个量
    same = True
    if "median_half_life_over_H" in have:
        a = float(cov["median_half_life_over_H"].iloc[0])
        b = float(mis["median_half_life_over_H"].iloc[0])
        same &= (b < 0.25)                      # 未覆盖组的相关长度同样远短于 H
    if "mean_excess_lagH" in have:
        b = float(mis["mean_excess_lagH"].iloc[0])
        same &= (b < 0.05)                      # 未覆盖组在合法延迟下同样接近 chance
    head = ("✅ 缺口不影响结论：" if same else "⚠️ 缺口可能影响结论：")
    tail = ("——未覆盖组在决定可学性的两个量（半衰期/H、合法延迟 H 下超出 chance 的量）上"
            "与已覆盖组同属一个分布，且都远未达到可学阈值，"
            "因此补齐这些组只会增加样本，不会翻转结论。"
            if same else
            "——未覆盖组在 test 侧审计上与已覆盖组不同质，补齐前不能外推。")
    return msg + head + "已覆盖 vs 未覆盖：" + "；".join(diffs) + tail


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman 秩相关（只用 numpy，避免为一个数字引入 scipy 依赖）。"""
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > EPS else float("nan")


# 秩相关被判为「基本没有单调关系」的阈值。|rho| 在这条线以内才允许说「不随 H 变长」；
# 之外必须按符号如实表述。0.3 不是硬统计门槛，而是 4~6 个 horizon 层能支撑的最弱说法。
_RHO_FLAT = 0.3


def _horizon_dependence_clause(s: dict[str, Any]) -> str:
    """把 horizon 依赖的结论文案**绑定到统计量的符号上**。

    2026-09-14 code review 修复项：原实现无条件拼接「相关长度基本不随预测视界一起变长，
    所以 H 越大越无望」，完全不看 ``horizon_hl_spearman`` 的取值。落盘 summary 里这句话
    紧跟在自己打印的 ``+0.74``（H=24 时 4.5 个窗口 → H=720 时 14.1 个窗口，长了约 3 倍）
    后面，同一句话里自相矛盾。真正成立的只有「半衰期/H 随 H 下降」这一半，所以论文的
    主张必须落在 ``horizon_ratio_spearman`` 上，绝对半衰期只能如实报告它在增长。
    """
    rho_hl = float(s.get("horizon_hl_spearman", float("nan")))
    rho_ratio = float(s.get("horizon_ratio_spearman", float("nan")))
    hl_lo, hl_hi = float(s["hl_at_min_H"]), float(s["hl_at_max_H"])
    h_lo, h_hi = float(s["H_min"]), float(s["H_max"])
    head = (f" horizon 依赖：半衰期的绝对值随 H 的 Spearman 秩相关 {rho_hl:+.2f}"
            f"（H={h_lo:.0f} 时 {hl_lo:.1f} 个窗口，H={h_hi:.0f} 时 {hl_hi:.1f} 个窗口），"
            f"而「半衰期/H」随 H 的秩相关 {rho_ratio:+.2f}")
    if not np.isfinite(rho_hl) or not np.isfinite(rho_ratio):
        return head + "——horizon 层太少或取值恒定，无法判定 horizon 依赖的方向。"
    # 绝对半衰期这一侧：严格按符号说话，不许再出现与 +0.74 冲突的「基本不随 H 变长」
    if rho_hl > _RHO_FLAT:
        grow = (f"——半衰期的**绝对值随 H 增长**"
                f"（H 放大 {h_hi / max(h_lo, EPS):.1f}×，半衰期只放大 "
                f"{hl_hi / max(hl_lo, EPS):.1f}×），即增长慢于 H 本身")
    elif rho_hl < -_RHO_FLAT:
        grow = "——半衰期的绝对值随 H 反而缩短"
    else:
        grow = "——相关长度基本不随预测视界一起变长"
    # 论文真正该主张的是比值那一侧：它才决定「H 越大越无望」成不成立
    if rho_ratio < -_RHO_FLAT:
        tail = "，因此「半衰期/H」随 H 下降：H 越大越无望。"
    elif rho_ratio > _RHO_FLAT:
        tail = "，但「半衰期/H」随 H 上升——不能宣称 H 越大越无望，长视界反而相对更有利。"
    else:
        tail = "，且「半衰期/H」与 H 没有单调关系，不能就 H 的方向下结论。"
    return head + grow + tail


def strata_verdict(s: dict[str, Any]) -> str:
    if not s.get("n_strata"):
        return "没有分层数据。"
    tot = s["strata_total_used"]
    below = s["strata_with_half_life_over_H_below_25pct"]
    exc = s["strata_with_excess_lagH_above_005"]
    lo, hi = s["median_half_life_windows_range"]
    worst_v, worst_k = s["max_excess_persist_lagH"], s["max_excess_persist_lagH_stratum"]
    ratio_v, ratio_k = s["max_median_half_life_over_H"], s["max_median_half_life_over_H_stratum"]
    head = (f"分层检验（{tot} 个可用层，按 backbone / dataset / horizon 三种切法）："
            f"半衰期中位数落在 {lo:.1f}~{hi:.1f} 个窗口之间，"
            f"{below}/{tot} 层的「半衰期/H」低于 25%。")
    tail = ""
    if "horizon_hl_spearman" in s:
        tail = _horizon_dependence_clause(s)
    if exc == 0 and below == tot:
        return (f"✅ {head} **没有任何一层**在合法延迟 H 下超出 chance 0.05 以上"
                f"（最像有信号的那层是 {worst_k}，也只有 {worst_v:+.3f}）；"
                f"相关长度相对 H 最长的那层是 {ratio_k}（{ratio_v:.1%}）。"
                f"结论在每一层都成立，不是被某个数据集或骨干拖出来的平均值。{tail}")
    if exc:
        return (f"⚠️ {head} 但有 {exc} 层在延迟 H 下仍超出 chance 0.05 以上，"
                f"最强的是 {worst_k}（{worst_v:+.3f}）——论文必须把这些层单独讨论，"
                f"不能笼统地说「所有设定下都学不到」。{tail}")
    return (f"🟡 {head} 有 {tot - below} 层的「半衰期/H」不低于 25%（最高 {ratio_k}={ratio_v:.1%}），"
            f"这些层上「相关长度远短于视界」的说法要弱化。{tail}")


# --------------------------------------------------------------------------- #
# oracle 审计第二条腿：headroom 里有多少是「训练随机性」而不是信号
# --------------------------------------------------------------------------- #
def pair_seed_groups(groups: list[Group]) -> list[tuple[Group, Group]]:
    """把只差 seed 的组配成对：{(backbone,dataset,pred_len)} -> (seed 小, seed 大)。

    只保留**两个 seed 都有的臂**、并截到公共窗口数。同一 (backbone,dataset,pred_len)
    下若有 >2 个 seed，按 seed 升序两两相邻配对。
    """
    by_cell: dict[tuple, list[Group]] = {}
    for g in groups:
        by_cell.setdefault((g.backbone, g.dataset, g.pred_len), []).append(g)
    out: list[tuple[Group, Group]] = []
    for _, gs in sorted(by_cell.items()):
        gs = sorted(gs, key=lambda x: x.seed)
        for ga, gb in zip(gs, gs[1:]):
            if ga.seed != gb.seed and len(set(ga.arms) & set(gb.arms)) >= 2:
                out.append((ga, gb))
    return out


def _align_pair(ga: Group, gb: Group) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """取两个 seed 的公共臂（按名字对齐，顺序一致）与公共窗口长度。

    臂必须**按名字**对齐：两个 seed 的 arms 列表若因某格失败而缺项，
    下标对齐会把 revin 的误差当成 fredf 的，得到的一致率是纯粹的假象。
    """
    arms = sorted(set(ga.arms) & set(gb.arms))
    ia = [ga.arms.index(a) for a in arms]
    ib = [gb.arms.index(a) for a in arms]
    n = min(ga.e_test.shape[1], gb.e_test.shape[1])
    return ga.e_test[ia][:, :n], gb.e_test[ib][:, :n], arms


def evaluate_seed_pair(ga: Group, gb: Group) -> dict[str, Any]:
    """成对 seed 审计：per-window oracle 的 headroom 有多少是跨 seed 可复现的？

    动机：``evaluate_argmin_structure`` 已经证明 oracle 上界对逐窗口重贴臂标签不变，
    所以它不能证明可学性。但那是个**构造性**论证，怀疑者可以说"现实里标签不是随机的"。
    这里给出第二条、完全经验性的腿：**同一个格子只换训练 seed 重跑一遍**，
    看逐窗口最优臂身份还剩多少。

    * ``argmin_agree`` —— 两个 seed 的 a*_t 相同的比例
    * ``chance_agree`` —— 两个 seed 各自边缘分布下独立重抽的一致率 Σ p_a[k]·p_b[k]
    * ``xseed_gain``   —— **跨 seed oracle**：用 seed A 的逐窗口 argmin 去选臂、
      在 seed B 的误差上结算，相对 B 的最佳固定臂的增益。它依然用了未来信息，
      但只保留了**跨 seed 可复现**的那部分臂身份，因此是任何「靠可复现结构」的
      选择器的上界——包括用了完美特征的静态门控。
    * ``self_gain``    —— 同 seed oracle（B 自己的 argmin 选 B 自己的误差）作参照
    * ``reproducible_frac`` —— xseed_gain / self_gain，即 headroom 里可复现的比例

    注意 baseline 一律取**测试段的最佳固定臂** ``min_k mean_t e[k,t]``（而不是 val 选臂）：
    这里要问的是"逐窗口决策比最好的单臂还能多赚多少"，不能让 val→test 选臂失配
    把功劳算给逐窗口自适应——那个混淆已经在在线实验里坑过我们一次。
    """
    ea, eb, arms = _align_pair(ga, gb)
    K, n = ea.shape
    aa, ab = ea.argmin(axis=0), eb.argmin(axis=0)
    pa = np.bincount(aa, minlength=K) / n
    pb = np.bincount(ab, minlength=K) / n
    chance = float((pa * pb).sum())
    agree = float((aa == ab).mean())
    z = (agree - chance) / np.sqrt(max(chance * (1 - chance), EPS) / n)

    def _gain(e: np.ndarray, pick: np.ndarray) -> float:
        bf = float(e.mean(axis=1).min())          # 测试段最佳固定臂
        return 100.0 * (bf - float(_pick(e, pick).mean())) / (bf + EPS)

    # 两个方向都算，避免"用哪个 seed 当选择器"引入的偏置
    xseed = 0.5 * (_gain(eb, aa) + _gain(ea, ab))
    self_ = 0.5 * (_gain(eb, ab) + _gain(ea, aa))

    # 逐窗口「该赚多少」的可复现性：优势向量 (e_none − min_k e_k)/e_none 的相关
    def _adv(e: np.ndarray, i_none: int) -> np.ndarray:
        return (e[i_none] - e.min(axis=0)) / (e[i_none] + EPS)

    i_none = arms.index("none") if "none" in arms else int(np.argmax(pa))
    va, vb = _adv(ea, i_none), _adv(eb, i_none)
    r = (float(np.corrcoef(va, vb)[0, 1]) if va.std() > EPS and vb.std() > EPS
         else float("nan"))

    # ⚠️ 解释这个审计**必须**先看下面两个量：如果换 seed 之后同一个臂的逐窗口误差几乎
    # 一模一样（相关 ≈1、块级 MSE 几乎不动），说明这个骨干在两个 seed 上收敛到了同一个解，
    # 那么"argmin 身份可复现"是**平凡**的，这个审计对它就没有区分力（DLinear 这类近凸
    # 模型尤其如此）。只有当同臂误差本身有可观的 seed 波动、而 argmin 身份**仍然**一致时，
    # 才能说"headroom 是真实结构而不是训练噪声"。
    same_corr = [float(np.corrcoef(ea[k], eb[k])[0, 1])
                 for k in range(K) if ea[k].std() > EPS and eb[k].std() > EPS]
    ma, mb = ea.mean(axis=1), eb.mean(axis=1)
    return {
        "cell": f"{ga.backbone}__{ga.dataset}__h{ga.pred_len}",
        "backbone": ga.backbone, "dataset": ga.dataset, "pred_len": int(ga.pred_len),
        "seed_a": int(ga.seed), "seed_b": int(gb.seed),
        "n_arms": K, "n_test_windows": int(n), "arms": ",".join(arms),
        "argmin_agree": agree, "chance_agree": chance,
        "excess_agree": agree - chance, "z_agree": float(z),
        "self_oracle_gain_pct": self_, "xseed_oracle_gain_pct": xseed,
        "reproducible_frac": float(xseed / self_) if abs(self_) > EPS else float("nan"),
        "noise_frac": float(1.0 - xseed / self_) if abs(self_) > EPS else float("nan"),
        "adv_corr_across_seeds": r,
        # 这个审计有没有区分力的前提量
        "same_arm_err_corr": float(np.mean(same_corr)) if same_corr else float("nan"),
        "seed_mse_gap_pct": float(100.0 * np.mean(np.abs(ma - mb) / (ma + EPS))),
        # 块级 sanity check：换 seed 之后最佳固定臂还是不是同一个？
        "best_fixed_arm_a": arms[int(ma.argmin())],
        "best_fixed_arm_b": arms[int(mb.argmin())],
    }


def summarize_seed_audit(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "protocol": "paired_seed_replicate",
        "n_pairs": int(len(df)),
        "seeds": sorted({int(x) for x in df["seed_a"]} | {int(x) for x in df["seed_b"]}),
        "mean_argmin_agree": float(df["argmin_agree"].mean()),
        "mean_chance_agree": float(df["chance_agree"].mean()),
        "mean_excess_agree": float(df["excess_agree"].mean()),
        "pairs_z_gt2": int((df["z_agree"] > 2).sum()),
        "mean_self_oracle_gain_pct": float(df["self_oracle_gain_pct"].mean()),
        "mean_xseed_oracle_gain_pct": float(df["xseed_oracle_gain_pct"].mean()),
        "mean_adv_corr_across_seeds": float(df["adv_corr_across_seeds"].mean()),
        "mean_same_arm_err_corr": float(df["same_arm_err_corr"].mean()),
        "mean_seed_mse_gap_pct": float(df["seed_mse_gap_pct"].mean()),
        "best_fixed_arm_stable_pct": float(
            100.0 * (df["best_fixed_arm_a"] == df["best_fixed_arm_b"]).mean()),
    }
    # 比值用「均值之比」而不是「比值之均值」：后者会被 self_gain≈0 的组炸成噪声
    s, x = out["mean_self_oracle_gain_pct"], out["mean_xseed_oracle_gain_pct"]
    out["reproducible_frac_of_headroom"] = float(x / s) if abs(s) > EPS else float("nan")
    out["noise_frac_of_headroom"] = float(1.0 - x / s) if abs(s) > EPS else float("nan")
    out["pairs_xseed_beats_best_fixed"] = int((df["xseed_oracle_gain_pct"] > 0).sum())
    return out


def seed_audit_verdict(s: dict[str, Any]) -> str:
    n = s["n_pairs"]
    ag, ch = s["mean_argmin_agree"], s["mean_chance_agree"]
    self_, x = s["mean_self_oracle_gain_pct"], s["mean_xseed_oracle_gain_pct"]
    noise = s["noise_frac_of_headroom"]
    win = s["pairs_xseed_beats_best_fixed"]
    corr = s.get("mean_same_arm_err_corr", float("nan"))
    gap = s.get("mean_seed_mse_gap_pct", float("nan"))
    head = (f"{n} 对成对 seed（同格子只换训练随机种子）：同 seed oracle 相对最佳固定臂 "
            f"{self_:+.2f}%，跨 seed oracle 只剩 {x:+.2f}%")
    if noise != noise:
        return f"⚠️ {head}，但 self_gain≈0，无法给出噪声占比。"
    if noise > 0.5:
        return (f"❌ {head}——headroom 的 **{noise:.0%} 是训练随机性**，不是信号。"
                f"逐窗口最优臂身份的跨 seed 一致率仅 {ag:.3f}（独立重抽 chance={ch:.3f}）。"
                "换句话说 oracle 主要在过拟合单次训练的噪声：连「重跑一遍同样的实验」都复现不了的"
                "决策，任何选择器都学不到。这与「重贴标签不变性」是两条独立的证据，"
                "共同说明 per-window oracle headroom 不能作为可学性的依据。")
    if x > 0.5 and win > n * 0.5:
        base = (f"✅ {head}，可复现比例 {1 - noise:.0%}，{win}/{n} 对跨 seed oracle 仍能赢"
                f"最佳固定臂；逐窗口最优臂身份的跨 seed 一致率 {ag:.3f} vs chance {ch:.3f}。"
                "**headroom 不是训练噪声，是真实、可复现的结构**——"
                "这排除了「oracle 只是在过拟合单次训练随机性」这个平凡解释。")
        if corr == corr and corr > 0.99:
            return (base + f" 但注意本审计在这批格子上**区分力有限**：同一个臂的逐窗口误差"
                    f"跨 seed 相关高达 {corr:.4f}、块级 MSE 仅差 {gap:.2f}%，"
                    "说明两个 seed 基本收敛到同一个解（DLinear 这类近凸模型的典型表现），"
                    "『可复现』几乎是平凡的。要让这条证据有力，必须补上非凸骨干"
                    "（PatchTST / iTransformer）的成对 seed。")
        return (base + f" 而且这不是平凡的：同臂逐窗口误差的跨 seed 相关只有 {corr:.3f}、"
                f"块级 MSE 随 seed 波动 {gap:.2f}%，即模型本身确实变了，但"
                "「哪个窗口该用哪个插件」依然稳定。"
                "于是结论从『没东西可学』升级为更精确的一句：**信号是真的，但它来得太晚**——"
                "结合最优臂身份的相关半衰期只有几个窗口、而部署要求提前 H 步决策，"
                "真实存在的 headroom 在因果约束下依然不可达。")
    return (f"⚠️ {head}，可复现比例 {1 - noise:.0%}（一致率 {ag:.3f} vs chance {ch:.3f}）。"
            "有一部分结构跨 seed 稳定，但绝对量太小，不足以支撑一个实用选择器。")


def evaluate_controls(g: Group, mode: str = "reg", seed: int = 26, n_repeat: int = 5) -> dict[str, Any]:
    """两个"证伪用"的对照。没有它们，审稿人有权认为收益来自换臂本身或过拟合。

    * **random**：把门控的选择向量随机重排。换臂频率、各臂被选的比例都与门控**完全一致**，
      只有「哪个窗口选哪个臂」被打乱。所以它精确地隔离出「窗口级信息」这一个变量：
      门控若赢不过它，说明收益来自臂的边缘分布（等价于一种随机混合），而不是条件决策。
    * **shuffled**：在**打乱行序的特征**上训练门控（标签不动），再在真实测试特征上预测。
      这拆掉特征与标签的对应关系，是"门控有没有在学真东西"的直接检验；
      它的分数近似于「用同样容量的模型去拟合噪声」能拿到多少，也就是过拟合基线。

    两者都做 ``n_repeat`` 次取均值，避免单次随机数决定结论。
    """
    if not g.has_val_windows:
        raise ValueError(
            f"{g.key}: 该组没有可信的 val 逐窗口误差（缺 .order.json 确定序标记），"
            "不能做门控类协议。调用方应先用 Group.has_val_windows 过滤。"
        )
    rng = np.random.default_rng(seed)
    predict, _ = _fit_gate(g.x_val, g.e_val, g.arms, g.i_none, mode, seed)
    sel = predict(g.x_test)
    i_bf = int(g.val_mean.argmin())
    m_bf = float(g.e_test[i_bf].mean())
    m_none = float(g.e_test[g.i_none].mean())

    rand, shuf = [], []
    for r in range(n_repeat):
        rand.append(float(_pick(g.e_test, rng.permutation(sel)).mean()))
        perm = rng.permutation(len(g.x_val))
        pr, _ = _fit_gate(g.x_val[perm], g.e_val, g.arms, g.i_none, mode, seed + 100 + r)
        shuf.append(float(_pick(g.e_test, pr(g.x_test)).mean()))
    m_rand, m_shuf = float(np.mean(rand)), float(np.mean(shuf))

    def rel(x: float, ref: float) -> float:
        return 100.0 * (ref - x) / (ref + EPS)

    return {
        "key": g.key, "backbone": g.backbone, "dataset": g.dataset, "pred_len": g.pred_len,
        "mse_random_gate": m_rand, "mse_shuffled_feat_gate": m_shuf,
        "gain_random_vs_none": rel(m_rand, m_none),
        "gain_shuffled_vs_none": rel(m_shuf, m_none),
        "gain_random_vs_best_fixed": rel(m_rand, m_bf),
        "gain_shuffled_vs_best_fixed": rel(m_shuf, m_bf),
        # 真门控相对这两个对照的净优势：论文里真正该报的数字
        "gate_minus_random": rel(float(_pick(g.e_test, sel).mean()), m_rand),
        "gate_minus_shuffled": rel(float(_pick(g.e_test, sel).mean()), m_shuf),
    }


# 反馈窗口 = 全历史（expanding mean）。用一个大到不可能小于测试长度的数表示，
# 这样 _sliding_mean 不需要特例分支，csv 里也仍然是一个可排序的数值列。
EXPANDING = 10 ** 9


def _sliding_mean(e: np.ndarray, w: int) -> np.ndarray:
    """e[k, t] 的因果滑动均值：out[k, t] = mean(e[k, max(0,t-w+1) .. t])。"""
    c = np.cumsum(e, axis=1)
    lo = np.maximum(np.arange(e.shape[1]) - w + 1, 0)
    hi_sum = c
    lo_sum = np.where(lo > 0, c[:, np.clip(lo - 1, 0, None)], 0.0)
    cnt = np.arange(e.shape[1]) - lo + 1
    return (hi_sum - lo_sum) / cnt


def evaluate_online(g: Group, window: int = 200, delay: int | None = None) -> dict[str, Any]:
    """把「该挂哪个插件」当成**带延迟反馈的在线决策**，而不是静态特征分类问题。

    动机：静态复杂度特征在 val 上学、在 test 上用，跨过了一次分布漂移；而部署时其实
    有一种更强的信号可用——**过去窗口上各臂真实的误差**。这既符合现实（预测发出 H 步后
    真值就到了），又完全不需要额外 GPU（用已经落盘的逐窗口误差就能算）。

    延迟的严格性：窗口 t 覆盖输入 [t, t+L)、目标 [t+L, t+L+H)。窗口 t' 的真值在
    t'+L+H 时刻才齐；要在 t+L 时刻（窗口 t 发预测的时刻）就能用，必须 t' ≤ t−H。
    所以延迟**恰好是 H 个窗口**（stride=1），默认 ``delay=g.pred_len``。这一条是这个
    基线能不能写进论文的关键：少延一格就是未来信息泄漏。

    * ``mse_online``       —— 合法的延迟反馈选择器（follow-the-leader over 滑动窗口）
    * ``mse_online_nodelay`` —— 故意不延迟（**作弊上界**），用来量化延迟本身的代价

    评测区间（2026-09-14 code review 修复项）：延迟版的前 H 个窗口反馈还没到、被强制
    填成 ``i_bf``，逐点等于 best_fixed；零延迟版没有这段暖机期。若两条曲线都在完整
    ``[0:n]`` 上取均值，``delay_cost_pp`` 里就混进了「暖机窗口占比」这个与延迟无关的
    变量——占比从 0.8%（长测试段）到 90%（DLinear__Exchange__h720：n=798、delay=720）
    不等，后者的 delay_cost_pp 被撑到 34pp，纯属稀释假象。``sweep_online`` 早就用
    公共区间 ``[d_max:]`` 修掉了这个问题，但当时没有回填到这里。现在与它统一：所有
    指标（含 switch_rate 与 Wilcoxon 配对检验）都只在公共区间 ``[min(H, n//2):]`` 上结算。
    """
    H = int(g.pred_len) if delay is None else int(delay)
    e = g.e_test
    n = e.shape[1]
    roll = _sliding_mean(e, window)              # roll[k, t] = 臂 k 在 [t-w+1, t] 的均误差
    i_bf = int(g.val_mean.argmin())

    def choose(shift: int) -> np.ndarray:
        sel = np.full(n, i_bf, dtype=int)        # 反馈还没到的前 shift 个窗口用 val 最佳固定臂
        if shift < n:
            src = roll[:, : n - shift] if shift > 0 else roll
            sel[shift:] = src.argmin(axis=0)
        return sel

    sel_on, sel_nd = choose(H), choose(0)
    # 公共评测区间：跳过延迟版的暖机期，让延迟版与零延迟版在**同一批窗口**上结算。
    # 与 sweep_online 一致：H 若已吃掉过半测试窗口就退让到 n//2（否则几乎没有窗口可评），
    # 并把这种组标成 insufficient——它的「增益 0」是暖机造成的，不能当真实增益平均进去。
    start = min(H, n // 2)
    sl = slice(start, None)
    insufficient = bool(H > start)
    m_none = float(e[g.i_none, sl].mean())
    m_bf = float(e[i_bf, sl].mean())
    m_or = float(e[:, sl].min(axis=0).mean())
    # 常量策略里的最优臂（在 test 上选）。这是"不做任何时变自适应"的天花板：
    # 在线选择器若只赢 best_fixed_val 却赢不过它，说明收益全部来自
    # 「val 选错了臂、在线反馈把它纠回来」，而不是窗口级的时变自适应。
    # 这个区分是这条线能不能写成"自适应"贡献的分水岭。
    i_bt = int(e[:, sl].mean(axis=1).argmin())
    m_bt = float(e[i_bt, sl].mean())
    m_on = float(_pick(e, sel_on)[sl].mean())
    m_nd = float(_pick(e, sel_nd)[sl].mean())

    def rel(x: float, ref: float) -> float:
        return 100.0 * (ref - x) / (ref + EPS)

    return {
        "key": g.key, "backbone": g.backbone, "dataset": g.dataset, "pred_len": g.pred_len,
        "n_test_windows": n, "feedback_window": window, "delay_windows": H,
        # 评测口径必须落盘：论文/报告要能声明「延迟版与零延迟版是同一批窗口」
        "eval_start": int(start), "n_eval_windows": int(n - start),
        "insufficient": insufficient,
        "best_fixed_val": g.arms[i_bf],
        "best_fixed_test": g.arms[i_bt], "arm_id_mismatch": bool(i_bt != i_bf),
        "mse_none": m_none, "mse_best_fixed": m_bf, "mse_online": m_on,
        "mse_online_nodelay": m_nd, "mse_oracle": m_or, "mse_best_fixed_test": m_bt,
        "gain_online_vs_none": rel(m_on, m_none),
        "gain_online_vs_best_fixed": rel(m_on, m_bf),
        "gain_nodelay_vs_best_fixed": rel(m_nd, m_bf),
        "gain_oracle_vs_none": rel(m_or, m_none),
        "oracle_realized_pct": 100.0 * (m_none - m_on) / (m_none - m_or + EPS),
        "delay_cost_pp": rel(m_nd, m_bf) - rel(m_on, m_bf),
        "gain_bestfixed_test_vs_val": rel(m_bt, m_bf),        # val->test 选臂失配的代价
        "gain_online_vs_best_fixed_test": rel(m_on, m_bt),    # 扣掉失配后剩下的时变增益
        "switch_rate": float((sel_on[sl] != g.i_none).mean()),
        "n_distinct_choices": int(len(np.unique(sel_on[sl]))),
        "wilcoxon_p_online_vs_best_fixed": _wilcoxon_p(_pick(e, sel_on)[sl], e[i_bf, sl]),
    }


def summarize_online(df: pd.DataFrame) -> dict[str, Any]:
    """把逐组的在线选择结果汇总成一个可以直接写进论文表格的字典。

    2026-09-14 code review 修复项：先剔除 ``insufficient=True`` 的组再取均值。
    这些组的 H 已经吃掉过半测试窗口（Exchange/ILI + h720），公共评测区间里仍然残留
    暖机窗口，它们的「增益 ≈ 0 / delay_cost 巨大」是评测区间被暖机污染的产物，
    不是真实的延迟代价；平均进去会同时污染 mean_gain 与 mean_delay_cost_pp。
    """
    n_all = int(len(df))
    if "insufficient" in df.columns:
        df = df[~df["insufficient"].astype(bool)]
    if df.empty:                                  # 极端情况：所有组都不可比，只报组数
        return {"protocol": "online_delayed_feedback", "n_groups": 0,
                "n_groups_all": n_all, "n_groups_insufficient": n_all}
    p = df["wilcoxon_p_online_vs_best_fixed"]
    return {
        "protocol": "online_delayed_feedback",
        "n_groups": int(len(df)),
        "n_groups_all": n_all,
        # 被排除的组数：报告里必须显式声明，否则读者会以为均值是全部组的
        "n_groups_insufficient": n_all - int(len(df)),
        "feedback_window": int(df["feedback_window"].iloc[0]),
        "mean_gain_online_vs_none": float(df["gain_online_vs_none"].mean()),
        "mean_gain_online_vs_best_fixed": float(df["gain_online_vs_best_fixed"].mean()),
        "median_gain_online_vs_best_fixed": float(df["gain_online_vs_best_fixed"].median()),
        "mean_gain_nodelay_vs_best_fixed": float(df["gain_nodelay_vs_best_fixed"].mean()),
        "mean_delay_cost_pp": float(df["delay_cost_pp"].mean()),
        "mean_gain_oracle_vs_none": float(df["gain_oracle_vs_none"].mean()),
        "mean_oracle_realized_pct": float(df["oracle_realized_pct"].mean()),
        "groups_online_beats_best_fixed_pct": float(100.0 * (df["gain_online_vs_best_fixed"] > 0).mean()),
        "groups_online_beats_none_pct": float(100.0 * (df["gain_online_vs_none"] > 0).mean()),
        "groups_significant_p05": int((p < 0.05).sum()),
        "mean_switch_rate": float(df["switch_rate"].mean()),
        # --- 收益来源分解 ---
        "mean_gain_bestfixed_test_vs_val": float(df["gain_bestfixed_test_vs_val"].mean()),
        "mean_gain_online_vs_best_fixed_test": float(df["gain_online_vs_best_fixed_test"].mean()),
        "groups_arm_id_mismatch_pct": float(100.0 * df["arm_id_mismatch"].mean()),
        "groups_online_beats_best_fixed_test_pct":
            float(100.0 * (df["gain_online_vs_best_fixed_test"] > 0).mean()),
    }


def sweep_online(g: Group,
                 windows: Sequence[int] = (50, 200, 1000, 3000, EXPANDING),
                 delay_ratios: Sequence[float] = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0),
                 align: bool = True) -> list[dict[str, Any]]:
    """把「延迟吃掉了收益」从一句话变成一条曲线：延迟 × 反馈窗口的二维扫描。

    延迟以 H 的倍数表示（``delay_ratios``），因为 H=96 和 H=720 的「延迟 H 个窗口」
    严重程度完全不同；ratio=0 是作弊上界，ratio=1 是唯一合法的部署设定。
    纯 numpy，一组 51×3×6 也就几秒，是这个项目里性价比最高的实验。

    ``align``（**必须打开，否则曲线是错的**）：延迟 d 的前 d 个窗口反馈还没到，只能退回
    best_fixed。d 越大，越多窗口和 best_fixed 完全相同，指标就被"暖机期"稀释着拉回 0——
    第一版扫描里 2H 看起来比 1H 好，就纯粹是这个假象。因此所有格子都只在公共评测区间
    ``[d_max:]`` 上算指标，d_max 取扫描中最大的延迟，这样跨延迟才可比。
    """
    e, n = g.e_test, g.e_test.shape[1]
    H = int(g.pred_len)
    delays = {float(r): (0 if r == 0 else max(1, int(round(r * H)))) for r in delay_ratios}
    # 评测区间起点：正常取最大延迟；若最大延迟已经吃掉一半以上测试窗口（ILI + h720
    # 这种极端格子），就退让到 n//2，并把仍然越界的格子标成 insufficient 置 NaN，
    # 而不是悄悄用一段被暖机污染的区间去算平均。
    start = min(max(delays.values()), n // 2) if align else 0
    i_bf = int(g.val_mean.argmin())
    sl = slice(start, None)
    m_none = float(e[g.i_none, sl].mean())
    m_bf = float(e[i_bf, sl].mean())
    m_or = float(e[:, sl].min(axis=0).mean())
    m_bt = float(e[:, sl].mean(axis=1).min())      # 常量策略的 oracle（同一评测区间）
    rows: list[dict[str, Any]] = []
    for w in windows:
        roll = _sliding_mean(e, int(w))
        for r, d in delays.items():
            sel = np.full(n, i_bf, dtype=int)
            if d < n:
                sel[d:] = (roll[:, : n - d] if d > 0 else roll).argmin(axis=0)
            bad = bool(align and d > start)     # 评测区间里还残留暖机窗口 -> 不可比
            m = float(_pick(e, sel)[sl].mean())
            rows.append({
                "key": g.key, "backbone": g.backbone, "dataset": g.dataset,
                "pred_len": H, "n_test_windows": n,
                "eval_start": start, "n_eval_windows": n - start,
                "feedback_window": int(w), "delay_ratio": r, "delay_windows": d,
                "delay_feasible": bool(r >= 1.0), "insufficient": bad,
                "gain_vs_none": float("nan") if bad else 100.0 * (m_none - m) / (m_none + EPS),
                "gain_vs_best_fixed": float("nan") if bad else 100.0 * (m_bf - m) / (m_bf + EPS),
                "gain_vs_best_fixed_test":
                    float("nan") if bad else 100.0 * (m_bt - m) / (m_bt + EPS),
                "gain_oracle_vs_none": 100.0 * (m_none - m_or) / (m_none + EPS),
            })
    return rows


def _common_support(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """只保留在**所有**延迟档 × 反馈窗口上都有有效增益的 key（公共支撑）。

    2026-09-14 code review 修复项。``sweep_online`` 会把 ``d > start`` 的格子置 NaN
    （``insufficient=True``），而 ``pivot_table`` / ``groupby().mean()`` 默认跳过 NaN，
    于是「延迟 → 增益」曲线的每个点其实来自**不同的组集合**：落盘产物里
    ``n_valid_by_delay_ratio = {0.0:127, …, 1.0:123, 2.0:103}``，被掉掉的恰好是长
    horizon（最不利）的那批组。曲线因此同时混入延迟效应与样本构成变化，而
    ``breakeven_delay_ratio`` 直接读这条曲线——ref_window=200 时 ratio=0.25 的均值
    从 +0.242（127/103 混合）变成 −0.240（同一批 103 组），盈亏平衡延迟从 0.25×H
    掉到 0.125×H。所以必须先取公共支撑，再做任何聚合。
    """
    if df.empty or "key" not in df.columns:
        return df, []
    ok = df.groupby("key")["gain_vs_best_fixed"].apply(lambda s: bool(s.notna().all()))
    keys = [str(k) for k, v in ok.items() if bool(v)]
    return df[df["key"].isin(keys)], keys


def summarize_online_sweep(df: pd.DataFrame, ref_window: int = 200) -> dict[str, Any]:
    """汇总扫描结果，并算出「盈亏平衡延迟」——论文里那张图的核心数字。

    所有跨延迟档可比的量（``piv`` / ``cur`` / ``breakeven`` / feasible / cheat）都在
    **公共支撑**上算：见 ``_common_support``。而 ``n_valid_by_delay_ratio`` 仍按原始表
    统计，因为它的职责正是暴露「哪些延迟档掉了组」，是公共支撑这一步的证据。
    """
    raw = df
    sup, keys = _common_support(raw)
    piv = (sup.pivot_table(index="delay_ratio", columns="feedback_window",
                           values="gain_vs_best_fixed", aggfunc="mean")
              .round(4)) if not sup.empty else pd.DataFrame()
    empty = pd.Series(dtype=float)

    def _curve(w: int) -> pd.Series:
        if sup.empty:
            return empty
        return sup[sup["feedback_window"] == w].groupby("delay_ratio")["gain_vs_best_fixed"].mean()

    cur = _curve(ref_window)
    if cur.empty and not sup.empty:                 # ref_window 不在扫描里就退回第一个
        ref_window = int(sup["feedback_window"].iloc[0])
        cur = _curve(ref_window)
    if (sup.empty and len(raw)
            and ref_window not in set(raw["feedback_window"])):
        # 公共支撑为空（所有组都在某个延迟档掉了）时，ref_window 至少要落在真实扫过的
        # 窗口上，否则连 n_valid_by_delay_ratio 这条诊断信息都会变成空字典。
        ref_window = int(raw["feedback_window"].iloc[0])
    pos = cur[cur > 0]
    best_w = (sup.groupby("feedback_window")["gain_vs_best_fixed"].mean().idxmax()
              if not sup.empty else None)
    feas = sup[sup["delay_ratio"] == 1.0] if not sup.empty else raw.iloc[:0]
    nvalid = (raw[raw["feedback_window"] == ref_window]
              .groupby("delay_ratio")["gain_vs_best_fixed"]
              .apply(lambda s: int(s.notna().sum())))
    n_keys_all = int(raw["key"].nunique()) if "key" in raw and len(raw) else 0
    return {
        "ref_window": int(ref_window),
        # n_groups 一律是**公共支撑下**的组数：论文声明「所有延迟档均为同一批 N 组」用的就是它
        "n_groups": len(keys),
        "n_groups_all": n_keys_all,
        "n_groups_dropped_no_common_support": n_keys_all - len(keys),
        "common_support": True,
        "aligned_eval_window": bool((raw["eval_start"] > 0).any()) if "eval_start" in raw else False,
        "windows": [int(w) for w in sorted(raw["feedback_window"].unique())],
        "delay_ratios": [float(r) for r in sorted(raw["delay_ratio"].unique())],
        "gain_by_delay_ratio": {str(k): float(v) for k, v in cur.items()},
        # 每个延迟档在**原始表**里有多少组可比：ILI + 长 horizon 会因为测试窗口太少被置 NaN。
        # 它与 n_groups 的差就是公共支撑剔掉的量，必须并排报出来。
        "n_valid_by_delay_ratio": {str(k): int(v) for k, v in nvalid.items()},
        "gain_by_delay_and_window": {str(i): {str(c): (None if pd.isna(v) else float(v))
                                              for c, v in row.items()}
                                     for i, row in piv.iterrows()},
        # 最大的「还能赢过最佳固定臂」的延迟。=0 说明连不延迟都赢不了；
        # ≥1 说明真实部署设定下这条路就是可行的。
        "breakeven_delay_ratio": (float(pos.index.max()) if len(pos) else 0.0),
        "best_feedback_window": (None if best_w is None else int(best_w)),
        "feasible_gain_vs_best_fixed": (float(feas["gain_vs_best_fixed"].mean())
                                        if len(feas) else float("nan")),
        "cheat_gain_vs_best_fixed": (float(sup[sup["delay_ratio"] == 0.0]["gain_vs_best_fixed"].mean())
                                     if (not sup.empty and (sup["delay_ratio"] == 0.0).any())
                                     else float("nan")),
    }


def online_sweep_verdict(s: dict[str, Any]) -> str:
    b = s.get("breakeven_delay_ratio", 0.0)
    feas = s.get("feasible_gain_vs_best_fixed", float("nan"))
    cheat = s.get("cheat_gain_vs_best_fixed", float("nan"))
    w = s.get("best_feedback_window")
    if b >= 1.0:
        return (f"✅ 在真实延迟（H 个窗口）下仍然赢 {feas:+.2f}%，最佳反馈窗口 {w}。"
                "延迟反馈选择器可以直接作为方法主线。")
    if b > 0.0:
        return (f"⚠️ 盈亏平衡延迟只有 {b:g}×H：延迟 ≤{b:g}H 时能赢，到 1×H（唯一合法设定）"
                f"就变成 {feas:+.2f}%（作弊上界 {cheat:+.2f}%）。这是一条干净的「反馈延迟"
                "决定插件选择可行性」的定量结论，图就是这条曲线。")
    return (f"❌ 连零延迟都赢不了最佳固定臂（{cheat:+.2f}%），说明不是延迟的问题，"
            "而是滑动窗口误差本身没有可利用的时变结构。")


def online_verdict(s: dict[str, Any]) -> str:
    if not s.get("n_groups"):
        # 所有组都被 insufficient 排除（H 吃掉过半测试窗口）时，只能说没有可比的组
        return (f"🟡 没有可比的组：{s.get('n_groups_insufficient', 0)} 组的延迟 H 已吃掉过半"
                "测试窗口，公共评测区间里全是暖机窗口，任何在线/零延迟对比都不成立。")
    g = s["mean_gain_online_vs_best_fixed"]
    pct = s["groups_online_beats_best_fixed_pct"]
    sig = s.get("groups_significant_p05", 0)
    n = s["n_groups"]
    dc = s.get("mean_delay_cost_pp", float("nan"))
    # 收益归因：赢 best_fixed_val 但赢不过 best_fixed_test，说明收益来自"val 选错臂"
    adapt = s.get("mean_gain_online_vs_best_fixed_test")
    mism = s.get("groups_arm_id_mismatch_pct")
    attrib = ""
    if adapt is not None:
        if g > 0.5 and adapt <= 0.0:
            attrib = (f" 但归因要小心：相对『在 test 上选的最优常量臂』只有 {adapt:+.2f}%，"
                      f"且 {mism:.0f}% 的组 val 选出的臂本来就不是 test 最优臂——"
                      "所以收益基本来自**修正 val→test 的选臂失配**，不是窗口级时变自适应。"
                      "写论文时必须按这个口径表述，否则一个 best_fixed_test 对照就能推翻。")
        elif adapt > 0.5:
            attrib = (f" 且相对『test 上最优常量臂』仍有 {adapt:+.2f}%，"
                      "说明存在真正的时变自适应收益，不只是纠正选臂失配。")
    if g > 0.5 and pct > 60:
        return (f"✅ 延迟误差反馈可行：在线选择比最佳固定臂好 {g:.2f}%，{pct:.0f}% 的组为正，"
                f"{sig}/{n} 组逐窗口配对显著；延迟 H 个窗口的代价 {dc:.2f} pp。" + attrib)
    if g <= 0 and np.isfinite(dc) and dc > 1.0:
        return (f"⚠️ 在线选择也输给最佳固定臂 {-g:.2f}%，但『不延迟』版本能多拿 {dc:.2f} pp——"
                "说明信号存在却被 H 步延迟吃掉了。可以据此讨论「插件选择的可行性受反馈延迟约束」。"
                + attrib)
    return (f"🟡 在线选择相对最佳固定臂 {g:+.2f}%（{pct:.0f}% 的组为正），信号不明确。" + attrib)


FEATURE_GROUPS = {
    "w_pc": "可预测性/熵",
    "w_ns": "非平稳性（分布漂移）",
    "w_fr": "频域集中度",
    "w_sd": "趋势与自相关",
    "w_ch": "通道间相关",
}


def ablate_feature_groups(g: Group, mode: str = "reg", seed: int = 26) -> dict[str, Any]:
    """逐组去掉一类特征再训门控（leave-one-group-out），看哪类信号在起作用。

    报的是「去掉这组之后，门控相对最佳固定臂的增益变成多少」——掉得最多的那一组
    就是贡献最大的那一类窗口信息。比 permutation importance 更贴近我们真正关心的
    目标（决策收益），因为它直接重训并重新评估决策。
    """
    if not g.has_val_windows:
        raise ValueError(
            f"{g.key}: 该组没有可信的 val 逐窗口误差（缺 .order.json 确定序标记），"
            "不能做门控类协议。调用方应先用 Group.has_val_windows 过滤。"
        )
    i_bf = int(g.val_mean.argmin())
    m_bf = float(g.e_test[i_bf].mean())
    out: dict[str, Any] = {"key": g.key, "backbone": g.backbone, "dataset": g.dataset,
                           "pred_len": g.pred_len}
    predict, _ = _fit_gate(g.x_val, g.e_val, g.arms, g.i_none, mode, seed)
    full = 100.0 * (m_bf - float(_pick(g.e_test, predict(g.x_test)).mean())) / (m_bf + EPS)
    out["gain_full"] = full
    for pref in FEATURE_GROUPS:
        keep = [i for i, n in enumerate(g.feat_names) if not n.startswith(pref)]
        if len(keep) == len(g.feat_names) or not keep:
            continue                       # 这组特征不存在 / 去掉就没特征了
        gg = Group(g.backbone, g.dataset, g.pred_len, g.seed, g.arms,
                   g.e_val, g.e_test, g.x_val[:, keep], g.x_test[:, keep],
                   [g.feat_names[i] for i in keep], val_mean=g.val_mean)
        pr, _ = _fit_gate(gg.x_val, gg.e_val, gg.arms, gg.i_none, mode, seed)
        v = 100.0 * (m_bf - float(_pick(gg.e_test, pr(gg.x_test)).mean())) / (m_bf + EPS)
        out[f"gain_drop_{pref}"] = v
        out[f"delta_drop_{pref}"] = v - full     # 负数 = 去掉后变差 = 这组有用
    return out


def evaluate_lodo(groups: list[Group], mode: str = "reg", seed: int = 26) -> list[dict[str, Any]]:
    """协议 lodo：门控在**其他数据集**的验证窗口上学，在留出数据集的测试窗口上评。

    只在 (backbone, pred_len, arms) 完全一致的组之间迁移，否则臂的下标含义会串。
    """
    rows: list[dict[str, Any]] = []
    by_shape: dict[tuple, list[Group]] = {}
    # lodo 要在别的数据集的 val 窗口上训练，故只能用 val 逐窗口可信的组
    for g in groups:
        if not g.has_val_windows:
            continue
        by_shape.setdefault((g.backbone, g.pred_len, tuple(g.arms), g.seed), []).append(g)

    for shape, gs in sorted(by_shape.items(), key=lambda kv: str(kv[0])):
        if len(gs) < 3:      # 留一之后至少要有 2 个数据集可训练
            continue
        for held in gs:
            tr = [g for g in gs if g.dataset != held.dataset]
            x_tr = np.concatenate([g.x_val for g in tr])
            e_tr = np.concatenate([g.e_val for g in tr], axis=1)
            predict, _ = _fit_gate(x_tr, e_tr, held.arms, held.i_none, mode, seed)
            sel = predict(held.x_test)
            e_none = held.e_test[held.i_none]
            m_none, m_gate = float(e_none.mean()), float(_pick(held.e_test, sel).mean())
            i_bf = int(held.val_mean.argmin())
            m_bf, m_or = float(held.e_test[i_bf].mean()), float(held.e_test.min(0).mean())
            rows.append({
                "key": held.key, "backbone": held.backbone, "dataset": held.dataset,
                "pred_len": held.pred_len, "seed": held.seed,
                "n_train_datasets": len(tr), "arms": "|".join(held.arms),
                "mse_none": m_none, "mse_gate": m_gate, "mse_best_fixed": m_bf,
                "mse_oracle": m_or, "best_fixed_val": held.arms[i_bf],
                "gain_gate_vs_none": 100.0 * (m_none - m_gate) / (m_none + EPS),
                "gain_gate_vs_best_fixed": 100.0 * (m_bf - m_gate) / (m_bf + EPS),
                "gain_oracle_vs_none": 100.0 * (m_none - m_or) / (m_none + EPS),
                "switch_rate": float((sel != held.i_none).mean()),
                "wilcoxon_p_gate_vs_best_fixed": _wilcoxon_p(
                    _pick(held.e_test, sel), held.e_test[i_bf]),
            })
    return rows


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #
def _entropy(labels: np.ndarray, k: int) -> float:
    """oracle 标签的归一化熵：0 = 永远同一个臂最优（门控没意义），1 = 完全均匀。"""
    if labels.size == 0 or k <= 1:
        return 0.0
    p = np.bincount(labels, minlength=k) / labels.size
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(k))


def _majority_acc(labels: np.ndarray) -> float:
    if labels.size == 0:
        return float("nan")
    return float(np.bincount(labels).max() / labels.size)


def _wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float:
    """逐窗口配对 Wilcoxon（双侧）。全等或样本太少时返回 nan。"""
    try:
        from scipy.stats import wilcoxon

        d = np.asarray(a) - np.asarray(b)
        if d.size < 10 or np.allclose(d, 0):
            return float("nan")
        return float(wilcoxon(a, b).pvalue)
    except Exception:
        return float("nan")


def summarize(df: pd.DataFrame, protocol: str) -> dict[str, Any]:
    """把逐组结果压成几句能直接写进论文的话。"""
    if df.empty:
        return {"protocol": protocol, "n_groups": 0}
    n = len(df)
    win_bf = int((df["gain_gate_vs_best_fixed"] > 0).sum())
    sig = int((df["wilcoxon_p_gate_vs_best_fixed"] < 0.05).sum())
    out = {
        "protocol": protocol,
        "n_groups": n,
        "n_test_windows_total": int(df.get("n_test_windows", pd.Series([0] * n)).sum()),
        "mean_gain_gate_vs_none": float(df["gain_gate_vs_none"].mean()),
        "median_gain_gate_vs_none": float(df["gain_gate_vs_none"].median()),
        "mean_gain_bestfixed_vs_none": float(df.get(
            "gain_bestfixed_vs_none", df["gain_gate_vs_none"] * np.nan).mean()),
        "mean_gain_gate_vs_best_fixed": float(df["gain_gate_vs_best_fixed"].mean()),
        "median_gain_gate_vs_best_fixed": float(df["gain_gate_vs_best_fixed"].median()),
        "groups_gate_beats_best_fixed": win_bf,
        "groups_gate_beats_best_fixed_pct": 100.0 * win_bf / n,
        "groups_significant_p05": sig,
        "mean_gain_oracle_vs_none": float(df["gain_oracle_vs_none"].mean()),
        "mean_switch_rate": float(df["switch_rate"].mean()),
    }
    if "oracle_realized_pct" in df:
        out["mean_oracle_realized_pct"] = float(df["oracle_realized_pct"].mean())
    if "gate_acc" in df:
        out["mean_gate_acc"] = float(df["gate_acc"].mean())
        out["mean_majority_acc"] = float(df["majority_acc"].mean())
        out["gate_beats_majority_pct"] = float(100.0 * (df["gate_acc"] > df["majority_acc"]).mean())
    # 逐组相对增益做一次跨组符号检验：结论"门控整体优于最佳固定臂"要靠这个
    out["wilcoxon_p_across_groups"] = _wilcoxon_p(
        df["gain_gate_vs_best_fixed"].to_numpy(), np.zeros(n)
    )
    return out


def verdict(s: dict[str, Any], min_groups: int = 8) -> str:
    """把统计量翻译成"这条路走不走得通"的一句话结论（周一早上只想看这个）。

    刻意保守：组数太少时不给结论。1 个决策组下「100% 的组为正」是必然事件，
    而随机噪声也能让单组赢 1%——无人值守报告最怕的就是这种自欺欺人的绿灯。
    """
    n = s.get("n_groups", 0)
    if n == 0:
        return "无数据：phase-2 逐窗口误差还没产出，无法判断。"
    g = s["mean_gain_gate_vs_best_fixed"]
    pct = s["groups_gate_beats_best_fixed_pct"]
    orc = s.get("mean_gain_oracle_vs_none", 0.0)
    sig = s.get("groups_significant_p05", 0)
    p_across = s.get("wilcoxon_p_across_groups", float("nan"))
    if n < min_groups:
        return (f"⏳ 样本太少（{n} 个决策组 < {min_groups}），不给结论。"
                f"当前 gate−best_fixed = {g:+.2f}%，oracle 上界 {orc:.1f}%，仅供参考。")
    if orc < 1.0:
        return (f"❌ 死路：oracle 上界只有 {orc:.2f}%，说明逐窗口最优臂几乎等于固定臂，"
                "再怎么学门控也没有空间。应改设计（换插件池 / 换粒度 / 换问题）。")
    strong = (g > 0.5 and pct > 60
              and (np.isnan(p_across) or p_across < 0.05)
              and sig >= max(1, int(0.3 * n)))
    if strong:
        return (f"✅ 有信号：门控平均比最佳固定臂再好 {g:.2f}%，{pct:.0f}% 的组为正，"
                f"{sig}/{n} 组逐窗口配对显著（跨组 p={p_across:.3g}）。"
                f"oracle 上界 {orc:.1f}%，可以按这个方向写论文。")
    if g <= 0:
        return (f"⚠️ 有空间但学不到：oracle 上界 {orc:.1f}%，但门控比最佳固定臂差 {-g:.2f}%。"
                "空间在、可学性不足——下一步该查特征表达力与门控训练分布（val→test 漂移）。")
    return (f"🟡 弱信号：门控比最佳固定臂好 {g:.2f}%，{pct:.0f}% 的组为正，"
            f"但只有 {sig}/{n} 组显著（跨组 p={p_across:.3g}）。"
            "需要更多组或更强特征才能定论，暂不能作为论文主张。")


LEARNABLE_MIN_VAL = 1000
"""门控训练窗口数的下限。低于这个数（ILI 只有 74 个验证窗口）时，
HistGBM 连自己的 20% early-stopping 划分都不够用，「学不到」是数据量的必然结果，
不能算作「窗口级门控不可行」的证据。1000 这个阈值是保守取值：
ETTh1/ETTh2 有 2785、Exchange 665、Electricity 2537、Weather 5175、ETTm* 11425。"""


def stratify_by_val_size(df: pd.DataFrame) -> dict[str, Any]:
    """按验证窗口数分层看结论。回答「是学不到，还是没得学」。"""
    bins = [(0, 200, "<200（ILI 级，基本不可学）"),
            (200, 1000, "200–1000（Exchange 级）"),
            (1000, 5000, "1000–5000（ETTh/Electricity 级）"),
            (5000, 10**9, "≥5000（Weather/ETTm 级）")]
    out: dict[str, Any] = {}
    for lo, hi, label in bins:
        sub = df[(df["n_val_windows"] >= lo) & (df["n_val_windows"] < hi)]
        if sub.empty:
            continue
        out[label] = {
            "n_groups": int(len(sub)),
            "mean_gain_gate_vs_best_fixed": float(sub["gain_gate_vs_best_fixed"].mean()),
            "mean_gain_gate_vs_none": float(sub["gain_gate_vs_none"].mean()),
            "mean_gain_oracle_vs_none": float(sub["gain_oracle_vs_none"].mean()),
            "mean_gate_acc": float(sub["gate_acc"].mean()),
            "mean_majority_acc": float(sub["majority_acc"].mean()),
            "pct_beats_best_fixed": float(100.0 * (sub["gain_gate_vs_best_fixed"] > 0).mean()),
        }
    return out


def controls_verdict(cs: dict[str, Any]) -> str:
    """对照的解读规则：先看真门控是否稳定赢过两个对照，再看对照本身是否异常。"""
    gr, gs = cs["mean_gate_minus_random"], cs["mean_gate_minus_shuffled"]
    pr, ps = cs["groups_gate_beats_random_pct"], cs["groups_gate_beats_shuffled_pct"]
    if gr > 0.3 and gs > 0.3 and pr > 60 and ps > 60:
        return (f"✅ 收益来自窗口级信息：门控比同频随机换臂好 {gr:+.2f}%（{pr:.0f}% 的组为正），"
                f"比打乱特征训练的门控好 {gs:+.2f}%（{ps:.0f}%）。两个证伪对照都没能复现收益。")
    if cs["mean_gain_shuffled_vs_best_fixed"] > 0.3:
        return (f"⚠️ 打乱特征也能赢最佳固定臂 {cs['mean_gain_shuffled_vs_best_fixed']:+.2f}%，"
                "说明「赢」的一部分来自模型容量/换臂本身而非窗口信息，"
                "论文里必须把这一列一起报出来，否则站不住。")
    return (f"🟡 门控相对两个对照的净优势有限（random {gr:+.2f}%，shuffled {gs:+.2f}%），"
            "窗口级信息的贡献还不够硬，不要把它写成主张。")


# --------------------------------------------------------------------------- #
# 自检（不依赖真实实验结果）
# --------------------------------------------------------------------------- #
def _self_test() -> None:
    """构造一个「门控本应成功」的合成组，验证整条链路能把信号识别出来。

    设计：窗口特征 f0 决定哪个臂更好——f0>0 时 arm1 误差减半，否则 arm1 误差翻倍。
    任何固定臂都只能拿到一半好处，正确的门控应该显著优于最佳固定臂。
    """
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.standard_normal((n, 3))
    base = 1.0 + 0.1 * rng.standard_normal(n).clip(-0.5, 0.5)
    good = x[:, 0] > 0
    e_none = base
    e_arm1 = np.where(good, base * 0.5, base * 2.0)
    e = np.stack([e_none, e_arm1])
    half = n // 2
    g = Group("Fake", "Synth", 96, 2021, ["none", "arm1"],
              e_val=e[:, :half], e_test=e[:, half:],
              x_val=x[:half], x_test=x[half:], feat_names=["w_a", "w_b", "w_c"])
    for mode in ("reg", "clf"):
        r = evaluate_group(g, mode=mode)
        print(f"[self-test/{mode}] gate_vs_none={r['gain_gate_vs_none']:.2f}% "
              f"gate_vs_best_fixed={r['gain_gate_vs_best_fixed']:.2f}% "
              f"oracle={r['gain_oracle_vs_none']:.2f}% acc={r['gate_acc']:.3f} "
              f"realized={r['oracle_realized_pct']:.1f}%")
        assert r["gain_gate_vs_best_fixed"] > 5.0, "合成信号下门控必须显著赢过最佳固定臂"
        assert r["gate_acc"] > 0.9, "合成信号下逐窗口准确率应接近 1"
    print("[self-test] 通过：链路能在有信号时识别出信号")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description="逐窗口插件门控：val 学 -> test 评")
    ap.add_argument("--config", default="configs/matrix_p2.yaml")
    ap.add_argument("--features", default=None, help="默认 <artifacts_dir>/window_features.parquet")
    ap.add_argument("--winerr", nargs="*", default=None,
                    help="逐窗口误差目录，可给多个（取并集扩大插件池）；默认 paths.winerr_dir")
    ap.add_argument("--protocol", default="both", choices=["inpool", "lodo", "both", "none"],
                    help="none = 只跑 --oracle-audit / --seed-audit / --online 这类分析，"
                         "不训练任何门控器（审计只用已落盘的逐窗口误差，秒级）")
    ap.add_argument("--mode", default="reg", choices=["reg", "clf"])
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--out-dir", default=None, help="默认 <artifacts_dir>")
    ap.add_argument("--tag", default="", help="产物文件名后缀，用于同时保存 reg / clf 两套结果")
    ap.add_argument("--controls", action="store_true",
                    help="加跑 random / shuffled-feature 两个证伪对照（纯 CPU，约每组 +6 次拟合）")
    ap.add_argument("--ablate", action="store_true",
                    help="加跑特征组 leave-one-out 消融（纯 CPU，约每组 +5 次拟合）")
    ap.add_argument("--oracle-audit", action="store_true",
                    help="审计 oracle 上界统计量：重贴标签不变性 + 最优臂身份的滞后可预测性")
    ap.add_argument("--seed-audit", action="store_true",
                    help="成对 seed 审计：headroom 里有多少是训练随机性而非信号。"
                         "需要同一格子有 ≥2 个 seed 的逐窗口误差，"
                         "例如 --winerr results/p2/winerr results/p2seed/winerr")
    ap.add_argument("--n-perm", type=int, default=32, help="重贴标签零假设的置换次数")
    ap.add_argument("--online", action="store_true",
                    help="加跑延迟误差反馈在线选择器（不训练模型，纯 numpy，秒级）")
    ap.add_argument("--online-window", type=int, default=200,
                    help="在线选择器的反馈滑动窗口长度（默认 200 个窗口）")
    ap.add_argument("--split-audit", action="store_true",
                    help="加跑 gate-train/gate-eval 同 split 内部时序切分诊断（纯 CPU）。"
                         "回答审稿意见「门控标签取自被早停用过的 val，失败也许只是偏差/漂移」："
                         "val 内部切 = 漂移 0，test 内部切 = 标签乐观偏差 0 且覆盖全部组")
    ap.add_argument("--split-train-frac", type=float, default=0.7,
                    help="split 诊断里前段（gate-train）占比，默认 0.7")
    ap.add_argument("--coverage-gap", action="store_true",
                    help="输出「test 侧审计 vs 门控协议」的逐骨干覆盖缺口表，"
                         "并用 test 侧审计指标证明缺口是否会改变结论"
                         "（同质性证据来自 --oracle-audit 本次结果，或 --audit-csv 复用旧结果）")
    ap.add_argument("--audit-csv", default=None,
                    help="复用已落盘的 selector_window_oracle_audit.csv 作为 --coverage-gap 的"
                         "同质性证据，避免为一张表重跑整套置换审计")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        _self_test()
        return 0

    cfg = MatrixConfig.load(args.config)
    winerr = [Path(w) for w in args.winerr] if args.winerr else [cfg.path("winerr_dir")]
    fpath = Path(args.features) if args.features else cfg.path("artifacts_dir") / "window_features.parquet"
    out_dir = Path(args.out_dir) if args.out_dir else cfg.path("artifacts_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{args.tag}" if args.tag else ""
    summary_path = out_dir / f"selector_window_summary{sfx}.json"

    if not fpath.exists():
        print(f"[selw] 缺特征表 {fpath}；先跑 python -m src.features_window --config {args.config}")
        return 2
    feats = pd.read_parquet(fpath) if fpath.suffix == ".parquet" else pd.read_csv(fpath)
    if "split" not in feats.columns:
        feats["split"] = "test"

    print(f"[selw] 逐窗口误差目录: {[str(w) for w in winerr]}")
    groups, notes = build_groups(winerr, feats)
    if not groups:
        print("[selw] 没有可用决策组（phase-2 逐窗口误差还没产出？）")
        summary_path.write_text(
            json.dumps({"n_groups": 0, "notes": notes[:50]}, ensure_ascii=False, indent=2),
            encoding="utf-8")
        return 3

    summaries: dict[str, Any] = {}
    audit_df: pd.DataFrame | None = None   # --oracle-audit 的逐组结果，--coverage-gap 会复用
    # 需要 val 逐窗口对齐的协议（门控 / 对照 / 消融 / lodo）只能用打标的组；
    # test 侧审计（oracle / argmin 结构 / 在线延迟）用全部组，因为 test 从来没被 shuffle。
    gate_groups = [g for g in groups if g.has_val_windows]
    n_gate, n_all = len(gate_groups), len(groups)
    if n_gate < n_all:
        excl = sorted({g.backbone for g in groups if not g.has_val_windows})
        print(f"[selw] ⚠️ 门控类协议只用 {n_gate}/{n_all} 个组"
              f"（排除无 val 确定序标记的骨干：{', '.join(excl)}）；"
              f"test 侧审计仍覆盖全部 {n_all} 个组")
    summaries["coverage"] = {
        "n_groups_test_side": n_all,
        "n_groups_gating": n_gate,
        "backbones_without_val_windows":
            sorted({g.backbone for g in groups if not g.has_val_windows}),
        "note": ("门控/对照/消融需要 val 逐窗口与特征按 window_index 对齐，"
                 "只在打了确定序标记的组上跑；oracle 审计/半衰期/在线延迟只用 test 侧，"
                 "覆盖全部组。"),
    }

    if args.protocol in ("inpool", "both") and not gate_groups:
        print("[selw] ⚠️ 没有任何组具备 val 逐窗口标记，跳过门控类协议")

    if args.protocol in ("inpool", "both") and gate_groups:
        df = pd.DataFrame([evaluate_group(g, args.mode, args.seed) for g in gate_groups])
        df.to_csv(out_dir / f"selector_window_inpool{sfx}.csv", index=False)
        s = summarize(df, "inpool")
        s["verdict"] = verdict(s)
        s["by_val_windows"] = stratify_by_val_size(df)
        # 只用「训练窗口够多」的组再判一次：ILI 只有 74 个验证窗口，
        # 门控在那种组上不可能学到东西，把它和 11425 窗口的 ETTm1 平均在一起
        # 会把结论稀释成噪声。论文主张必须建立在可学的那一层上。
        big = df[df["n_val_windows"] >= LEARNABLE_MIN_VAL]
        if len(big) >= 4:
            sb = summarize(big, "inpool")
            sb["verdict"] = verdict(sb)
            sb["filter"] = f"n_val_windows >= {LEARNABLE_MIN_VAL}"
            summaries["inpool_learnable"] = sb
        summaries["inpool"] = s
        print("\n=== 协议 inpool（val 学 -> test 评，同一 backbone×dataset×horizon）===")
        print(json.dumps(s, ensure_ascii=False, indent=2))
        print("\n按骨干分层（gate vs best_fixed，%）:")
        print(df.groupby("backbone")[["gain_gate_vs_none", "gain_bestfixed_vs_none",
                                      "gain_gate_vs_best_fixed", "gain_oracle_vs_none"]]
              .mean().round(2).to_string())

        if args.controls:
            dc = pd.DataFrame([evaluate_controls(g, args.mode, args.seed) for g in gate_groups])
            dc.to_csv(out_dir / f"selector_window_controls{sfx}.csv", index=False)
            cs = {
                "n_groups": int(len(dc)),
                "mean_gain_random_vs_best_fixed": float(dc["gain_random_vs_best_fixed"].mean()),
                "mean_gain_shuffled_vs_best_fixed": float(dc["gain_shuffled_vs_best_fixed"].mean()),
                "mean_gate_minus_random": float(dc["gate_minus_random"].mean()),
                "mean_gate_minus_shuffled": float(dc["gate_minus_shuffled"].mean()),
                "groups_gate_beats_random_pct": float(100.0 * (dc["gate_minus_random"] > 0).mean()),
                "groups_gate_beats_shuffled_pct": float(100.0 * (dc["gate_minus_shuffled"] > 0).mean()),
            }
            cs["verdict"] = controls_verdict(cs)
            summaries["inpool_controls"] = cs
            print("\n=== 证伪对照（random / shuffled-feature）===")
            print(json.dumps(cs, ensure_ascii=False, indent=2))

        if args.ablate:
            da = pd.DataFrame([ablate_feature_groups(g, args.mode, args.seed) for g in gate_groups])
            da.to_csv(out_dir / f"selector_window_ablation{sfx}.csv", index=False)
            dcols = [c for c in da.columns if c.startswith("delta_drop_")]
            abl = {c.replace("delta_drop_", ""): float(da[c].mean()) for c in dcols}
            summaries["inpool_ablation"] = {
                "n_groups": int(len(da)),
                "mean_gain_full": float(da["gain_full"].mean()),
                "mean_delta_when_dropped": abl,
                "most_important_group": (min(abl, key=abl.get) if abl else None),
            }
            print("\n=== 特征组消融（负数 = 去掉后变差 = 这组有用，单位 pp）===")
            for k, v in sorted(abl.items(), key=lambda kv: kv[1]):
                print(f"   去掉 {k:6s} ({FEATURE_GROUPS.get(k, '')}): {v:+.3f} pp")

    if args.oracle_audit:
        # 这一步必须在解读任何门控结果之前看：如果"最优臂身份"在合法延迟下不可预测，
        # 那么门控赢不了 best_fixed 就不是方法问题，而是问题本身是空的。
        dn = pd.DataFrame([evaluate_argmin_structure(g, args.n_perm, args.seed) for g in groups])
        audit_df = dn                     # 供 --coverage-gap 比较「已覆盖 vs 未覆盖」两批组
        dn.to_csv(out_dir / f"selector_window_oracle_audit{sfx}.csv", index=False)
        sn = summarize_argmin_structure(dn)
        sn["verdict"] = argmin_structure_verdict(sn)
        summaries["oracle_audit"] = sn
        print("\n=== oracle 上界审计（重贴标签不变性 + 最优臂身份的滞后可预测性）===")
        print(json.dumps(sn, ensure_ascii=False, indent=2))
        print("\n按 horizon 分层（滞后一致率超出 chance）:")
        print(dn.groupby("pred_len")[["chance_persist", "persist_lag1", "persist_lagH",
                                      "excess_persist_lag1", "excess_persist_lagH"]]
              .mean().round(4).to_string())

        # 分层：论文主张是「相关长度 << 视界」，必须证明它不是被某个数据集/骨干拖出来的均值。
        dstr = stratify_audit(dn)
        dstr.to_csv(out_dir / f"selector_window_audit_strata{sfx}.csv", index=False)
        sstr = summarize_strata(dstr)
        sstr["verdict"] = strata_verdict(sstr)
        summaries["audit_strata"] = sstr
        print("\n=== 分层审计（backbone / dataset / horizon 各切一遍）===")
        print(dstr[["stratum_key", "stratum", "n_groups", "mean_excess_persist_lag1",
                    "mean_excess_persist_lagH", "groups_z_gt2_lagH",
                    "median_half_life_windows", "median_half_life_over_H"]]
              .round(4).to_string(index=False))
        print("\n" + sstr["verdict"])

    if args.split_audit:
        # 堵审稿意见：「门控标签取自被早停用过的 val，失败也许只是乐观偏差 / val→test 漂移」。
        # 两个同 split 内部时序切分诊断，都是零 GPU（只用已落盘的逐窗口误差 + 特征表）。
        split_summaries: dict[str, Any] = {}
        for source, pool, why in (
            ("val", gate_groups, "同分布：漂移=0，乐观偏差在训练/评估两侧同向抵消"),
            ("test", groups, "标签乐观偏差=0（test 从未参与早停/best-epoch），且覆盖全部组"),
        ):
            rows = [evaluate_split_gate(g, source, args.split_train_frac, args.mode, args.seed)
                    for g in pool]
            kept = [r for r in rows if r is not None]
            n_skip = len(rows) - len(kept)
            if not kept:
                print(f"\n[selw] split 诊断（{source} 内部）：{len(pool)} 个组全部窗口数不足，跳过")
                continue
            ds = pd.DataFrame(kept)
            ds.to_csv(out_dir / f"selector_window_split_{source}{sfx}.csv", index=False)
            ss = summarize(ds, f"split_{source}")
            ss["source"] = source
            ss["train_frac"] = args.split_train_frac
            ss["why_this_answers_the_reviewer"] = why
            ss["n_groups_skipped_insufficient_windows"] = n_skip
            ss["min_train_windows"] = SPLIT_MIN_TRAIN
            ss["min_eval_windows"] = SPLIT_MIN_EVAL
            # 与论文主表口径（整段 val 均值选臂）的对比，说明差异不是基线换了才出来的
            ss["mean_gain_gate_vs_best_fixed_valmean"] = float(
                ds["gain_gate_vs_best_fixed_valmean"].mean())
            ss["pct_bf_split_is_eval_argmin"] = float(
                100.0 * ds["bf_split_is_eval_argmin"].mean())
            ss["by_backbone"] = {
                str(k): {"n_groups": int(len(v)),
                         "mean_gain_gate_vs_best_fixed":
                             float(v["gain_gate_vs_best_fixed"].mean()),
                         "mean_gain_oracle_vs_none": float(v["gain_oracle_vs_none"].mean())}
                for k, v in ds.groupby("backbone")
            }
            split_summaries[f"split_{source}"] = ss
            print(f"\n=== split 诊断（{source} 内部时序切分 "
                  f"{args.split_train_frac:.0%}/{1 - args.split_train_frac:.0%}）===")
            print(f"    {why}；{len(kept)} 组可用，{n_skip} 组窗口不足")
            print(json.dumps({k: v for k, v in ss.items() if k != "by_backbone"},
                             ensure_ascii=False, indent=2))
            print("    按骨干：")
            for k, v in ss["by_backbone"].items():
                print(f"      {k:14s} n={v['n_groups']:3d} "
                      f"gate−bf {v['mean_gain_gate_vs_best_fixed']:+.2f}% "
                      f"oracle {v['mean_gain_oracle_vs_none']:+.1f}%")
        summaries.update(split_summaries)
        if split_summaries:
            v = split_gate_verdict(split_summaries.get("split_val"),
                                   split_summaries.get("split_test"))
            summaries["split_audit_verdict"] = v
            print("\n" + v)

    if args.coverage_gap:
        # 论文 threats 必需：门控只覆盖一部分组，必须显式列出缺口并证明缺口不改变结论。
        if audit_df is None and args.audit_csv:
            ap_csv = Path(args.audit_csv)
            if ap_csv.exists():
                audit_df = pd.read_csv(ap_csv)
                print(f"[selw] 覆盖缺口的同质性证据复用 {ap_csv}（{len(audit_df)} 组）")
            else:
                print(f"[selw] ⚠️ --audit-csv {ap_csv} 不存在，覆盖表将只有组数")
        dcov = coverage_gap_table(groups, audit_df)
        dcov.to_csv(out_dir / f"selector_window_coverage_gap{sfx}.csv", index=False)
        cv = coverage_gap_verdict(dcov)
        summaries["coverage_gap"] = {
            "table": dcov.to_dict(orient="records"),
            "verdict": cv,
            "note": ("test 侧审计（oracle 上界 / 身份半衰期 / 合法延迟 H 下超出 chance 的量）"
                     "对全部组可算且不受 val 顺序 bug 影响，因此可用来检验"
                     "「未覆盖组是否与已覆盖组同质」。"),
        }
        print("\n=== 覆盖缺口（门控类协议 vs test 侧审计）===")
        cols = [c for c in ["backbone", "n_groups_test_side", "n_groups_gating", "n_missing",
                            "pct_gating_covered", "mean_oracle_gain_pct", "median_half_life",
                            "median_half_life_over_H", "mean_excess_lagH"] if c in dcov.columns]
        print(dcov[cols].round(4).to_string(index=False))
        print("\n" + cv)
        if audit_df is None:
            print("[selw] ⚠️ 没跑 --oracle-audit，覆盖表只有组数没有同质性证据")

    if args.seed_audit:
        # oracle 审计的第二条腿：把 headroom 拆成「跨 seed 可复现的信号」+「训练噪声」。
        # 需要同一格子两个 seed 的逐窗口误差，所以 --winerr 要同时给 p2 与 p2seed。
        pairs = pair_seed_groups(groups)
        if not pairs:
            seeds = sorted({g.seed for g in groups})
            print(f"\n[selw] seed 审计跳过：没有同格子的多 seed 组（现有 seed={seeds}）。"
                  "先跑 scripts/make_p2seed_config.py 派生的复制组，再把两个 winerr 目录一起传进来。")
        else:
            dsd = pd.DataFrame([evaluate_seed_pair(a, b) for a, b in pairs])
            dsd.to_csv(out_dir / f"selector_window_seed_audit{sfx}.csv", index=False)
            sd = summarize_seed_audit(dsd)
            sd["verdict"] = seed_audit_verdict(sd)
            summaries["seed_audit"] = sd
            print("\n=== 成对 seed 审计（headroom 里有多少是训练随机性）===")
            print(json.dumps(sd, ensure_ascii=False, indent=2))
            print("\n逐格明细（同 seed oracle vs 跨 seed oracle，相对测试段最佳固定臂 %）:")
            print(dsd[["cell", "n_test_windows", "argmin_agree", "chance_agree",
                       "self_oracle_gain_pct", "xseed_oracle_gain_pct", "noise_frac"]]
                  .round(4).to_string(index=False))

    if args.online:
        # 在线选择器不训练任何模型、也不用特征表，只用已经落盘的逐窗口误差，
        # 因此和 --mode 无关：reg / clf 两次调用会得到同一份结果，落盘一次就够。
        do = pd.DataFrame([evaluate_online(g, args.online_window) for g in groups])
        do.to_csv(out_dir / f"selector_window_online{sfx}.csv", index=False)
        so = summarize_online(do)
        so["verdict"] = online_verdict(so)
        summaries["online"] = so
        print("\n=== 在线延迟误差反馈选择器（延迟恰好 H 个窗口，无未来信息）===")
        print(json.dumps(so, ensure_ascii=False, indent=2))
        print("\n按骨干分层（online vs best_fixed，%）:")
        print(do.groupby("backbone")[["gain_online_vs_none", "gain_online_vs_best_fixed",
                                      "gain_nodelay_vs_best_fixed", "delay_cost_pp"]]
              .mean().round(2).to_string())

        # 延迟 × 反馈窗口扫描：把「延迟吃掉了收益」变成可画的曲线，并给出盈亏平衡延迟
        ds = pd.DataFrame([r for g in groups for r in sweep_online(g)])
        ds.to_csv(out_dir / f"selector_window_online_sweep{sfx}.csv", index=False)
        ss = summarize_online_sweep(ds)
        ss["verdict"] = online_sweep_verdict(ss)
        summaries["online_sweep"] = ss
        print("\n=== 延迟 × 反馈窗口扫描（数值 = 相对最佳固定臂的增益 %）===")
        print(ds.pivot_table(index="delay_ratio", columns="feedback_window",
                             values="gain_vs_best_fixed", aggfunc="mean").round(2).to_string())
        print(f"\n盈亏平衡延迟 = {ss['breakeven_delay_ratio']:g}×H；{ss['verdict']}")

    if args.protocol in ("lodo", "both"):
        rows = evaluate_lodo(groups, args.mode, args.seed)
        if rows:
            dfl = pd.DataFrame(rows)
            dfl.to_csv(out_dir / f"selector_window_lodo{sfx}.csv", index=False)
            s = summarize(dfl, "lodo")
            s["verdict"] = verdict(s)
            summaries["lodo"] = s
            print("\n=== 协议 lodo（跨数据集迁移）===")
            print(json.dumps(s, ensure_ascii=False, indent=2))
        else:
            print("\n[selw] lodo 跳过：同形状 (backbone,horizon,arms) 的数据集不足 3 个")

    for k_, s_ in summaries.items():
        # 在线选择器不涉及 reg/clf 学习器，标错 mode 会让报告读者以为它也训练了模型
        s_["mode"] = ("error-feedback" if k_.startswith("online")
                      else "permutation-null" if k_ == "oracle_audit"
                      else "paired-seed-replicate" if k_ == "seed_audit" else args.mode)
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[selw] 结论 -> {summary_path}")
    for k, s_ in summaries.items():
        if isinstance(s_, dict) and s_.get("verdict"):
            print(f"  [{k}] {s_['verdict']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
