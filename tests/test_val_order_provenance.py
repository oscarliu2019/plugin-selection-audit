"""val 段逐窗口误差「顺序确定性」的回归测试（2026-09-14 code review P0）。

事故本体：TSLib 的 ``data_provider/data_factory.py`` 是
``shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True``，
因此 ``data_provider(configs, "val")`` 返回的是 **shuffle=True** 的 loader。
而 ``StreamingMetrics`` 按 loader 迭代顺序累积逐窗口误差、落盘到 ``winerr/val/``，
下游 ``selector_window.build_groups`` 又严格按下标把窗口特征与逐窗口误差对齐
（``x_val[i]`` ↔ ``e_val[:, i]``）。结果是门控器在「特征 i 配随机窗口 j 的误差」上训练，
且每个臂的置换各不相同（不同插件在训练期消耗的全局 RNG 不同），
跨臂 ``argmin_k e_val[k, t]`` 也一起错位。

因为 ``drop_last=False``，两侧长度完全一致，原有的长度校验拦不住，全程静默。

这里锁死三件事：
1. 逐窗口误差的顺序**确实**跟随 loader 顺序——即 shuffle 会真的破坏对齐（证明危害存在，
   而不是「反正 mean 一样」）；
2. 写入方会给确定序数组打上 ``.order.json`` 溯源标记；
3. 读取方拒绝未打标的旧 val 数组，而不是静默用污染数据算出一堆门控结论。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src.runner import (  # noqa: E402
    StreamingMetrics,
    mark_deterministic_order,
    save_window_mse_atomic,
)
from src.selector_window import (  # noqa: E402
    ablate_feature_groups,
    build_groups,
    evaluate_controls,
    evaluate_group,
    load_winerr,
)

CELL = "DLinear__ETTh1__h96__none__s2021"


def _feed_in_order(pred, true, order, bs=4):
    """按给定顺序喂样本，模拟一个特定 loader 的迭代顺序。"""
    m = StreamingMetrics(pred_len=pred.shape[1], per_window=True)
    idx = torch.as_tensor(order, dtype=torch.long)
    p, t = pred[idx], true[idx]
    for i in range(0, p.shape[0], bs):
        m.update(p[i : i + bs], t[i : i + bs])
    return m.window_mse()


# --------------------------------------------------------------------------- #
# 1. 危害存在性：shuffle 真的会打乱逐窗口曲线
# --------------------------------------------------------------------------- #
def test_window_order_follows_loader_order_so_shuffle_breaks_alignment() -> None:
    """若 loader 被打乱，落盘的逐窗口误差就是置换后的顺序——这正是 P0 的机理。"""
    torch.manual_seed(0)
    n = 17
    pred, true = torch.randn(n, 6, 2), torch.randn(n, 6, 2)

    natural = _feed_in_order(pred, true, np.arange(n))
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    shuffled = _feed_in_order(pred, true, perm)

    # 均值不变（所以聚合 val_mse 与 early stopping 是安全的）……
    assert np.isclose(natural.mean(), shuffled.mean(), rtol=1e-5)
    # ……但逐窗口曲线是置换关系，按下标对齐必然错位
    assert not np.allclose(natural, shuffled)
    np.testing.assert_allclose(np.sort(natural), np.sort(shuffled), rtol=1e-5)
    np.testing.assert_allclose(shuffled, natural[perm], rtol=1e-5)


# --------------------------------------------------------------------------- #
# 2. 写入方打标
# --------------------------------------------------------------------------- #
def test_mark_deterministic_order_writes_marker(tmp_path) -> None:
    d = tmp_path / "winerr" / "val"
    save_window_mse_atomic(d, CELL, np.arange(5, dtype=np.float32))
    p = mark_deterministic_order(d, CELL, 5)

    assert p.name == f"{CELL}.order.json"
    meta = json.loads(p.read_text())
    assert meta["loader_order"] == "deterministic"
    assert meta["n_windows"] == 5


# --------------------------------------------------------------------------- #
# 3. 读取方拒绝未打标的旧数组
# --------------------------------------------------------------------------- #
def test_load_winerr_accepts_marked_val_arrays(tmp_path) -> None:
    d = tmp_path / "val"
    save_window_mse_atomic(d, CELL, np.arange(5, dtype=np.float32))
    mark_deterministic_order(d, CELL, 5)

    got = load_winerr([d], require_order_marker=True)
    assert list(got) == [("DLinear", "ETTh1", 96, 2021)]


def test_load_winerr_rejects_unmarked_val_arrays(tmp_path) -> None:
    """修复前落盘的 val 数组没有标记，必须报错而不是静默载入。"""
    d = tmp_path / "val"
    save_window_mse_atomic(d, CELL, np.arange(5, dtype=np.float32))  # 故意不打标

    with pytest.raises(RuntimeError, match="缺少确定序标记"):
        load_winerr([d], require_order_marker=True)


def test_load_winerr_test_side_needs_no_marker(tmp_path) -> None:
    """test loader 本来就是 shuffle=False，不该被这道校验波及。"""
    d = tmp_path / "winerr"
    save_window_mse_atomic(d, CELL, np.arange(5, dtype=np.float32))
    got = load_winerr([d])  # require_order_marker 默认 False
    assert list(got) == [("DLinear", "ETTh1", 96, 2021)]


def test_contaminated_val_cannot_reach_any_gating_protocol(tmp_path) -> None:
    """端到端：未打标的 val 绝不能被门控类协议使用。

    断言的是**安全性质**，不是「组被整个丢掉」。设计上未打标的组仍然保留，
    因为 test 侧从来没被 shuffle，oracle 审计 / argmin 半衰期 / 在线延迟这些
    主结论理应继续覆盖它（否则会白丢掉 TimesNet 这个最贵的骨干）。
    真正必须成立的是：``e_val`` / ``x_val`` 为 None、``has_val_windows`` 为假，
    且任何门控函数拿到它都要显式报错而不是算出一个数。
    """
    n = 8
    root = tmp_path / "winerr"
    for arm in ("none", "revin"):
        cid = f"DLinear__ETTh1__h96__{arm}__s2021"
        save_window_mse_atomic(root, cid, np.random.rand(n).astype(np.float32))
        save_window_mse_atomic(root / "val", cid, np.random.rand(n).astype(np.float32))
        # val 侧故意不打标，模拟修复前的历史产物

    feats = pd.DataFrame({
        "dataset": ["ETTh1"] * (2 * n),
        "split": ["val"] * n + ["test"] * n,
        "window_index": list(range(n)) * 2,
        "w_a": np.random.rand(2 * n),
        "w_b": np.random.rand(2 * n),
    })

    groups, notes = build_groups(root, feats, verbose=False)
    assert len(groups) == 1, "test 侧仍可用，组不该被整个丢掉"
    g = groups[0]
    assert g.e_val is None and g.x_val is None
    assert not g.has_val_windows
    assert any("确定序标记" in m for m in notes), notes
    # 门控类协议必须显式拒绝，而不是在 None 上崩出难懂的 TypeError
    for fn in (evaluate_group, ablate_feature_groups):
        with pytest.raises(ValueError, match="不能做门控类协议"):
            fn(g)
    with pytest.raises(ValueError, match="不能做门控类协议"):
        evaluate_controls(g)


def test_val_mean_survives_shuffled_order(tmp_path) -> None:
    """val 每臂**均值**对窗口置换不变，所以 best_fixed_val 对未打标的组依然正确。

    这正是「只丢逐窗口、不丢整组」的依据。
    """
    n = 200
    root = tmp_path / "winerr"
    rng = np.random.default_rng(0)
    truth = {}
    for arm, scale in (("none", 1.0), ("revin", 0.5)):
        cid = f"TimesNet__ETTh1__h96__{arm}__s2021"
        v = (rng.random(n) * scale).astype(np.float32)
        truth[arm] = float(v.mean())
        save_window_mse_atomic(root, cid, rng.random(n).astype(np.float32))
        # 模拟被打乱顺序落盘：均值不变，逐窗口顺序不可信
        save_window_mse_atomic(root / "val", cid, rng.permutation(v))

    feats = pd.DataFrame({
        "dataset": ["ETTh1"] * (2 * n),
        "split": ["val"] * n + ["test"] * n,
        "window_index": list(range(n)) * 2,
        "w_a": rng.random(2 * n),
    })
    groups, _ = build_groups(root, feats, verbose=False)
    g = groups[0]
    assert not g.has_val_windows
    for i, arm in enumerate(g.arms):
        assert g.val_mean[i] == pytest.approx(truth[arm], rel=1e-6)
    # revin 的 val 均值更小 => best_fixed_val 必须选中它
    assert g.arms[int(g.val_mean.argmin())] == "revin"


def test_marked_groups_keep_full_gating_capability(tmp_path) -> None:
    """打标的组必须照常具备 val 逐窗口能力，且 val_mean 与 e_val 自洽。"""
    n = 40
    root = tmp_path / "winerr"
    rng = np.random.default_rng(1)
    for arm in ("none", "revin"):
        cid = f"DLinear__ETTh1__h96__{arm}__s2021"
        save_window_mse_atomic(root, cid, rng.random(n).astype(np.float32))
        v = rng.random(n).astype(np.float32)
        save_window_mse_atomic(root / "val", cid, v)
        mark_deterministic_order(root / "val", cid, n)
    feats = pd.DataFrame({
        "dataset": ["ETTh1"] * (2 * n),
        "split": ["val"] * n + ["test"] * n,
        "window_index": list(range(n)) * 2,
        "w_a": rng.random(2 * n),
    })
    groups, _ = build_groups(root, feats, verbose=False)
    g = groups[0]
    assert g.has_val_windows
    assert g.e_val is not None and g.e_val.shape == (2, n)
    assert np.allclose(g.val_mean, g.e_val.mean(axis=1))


def test_mixed_state_splits_coverage_by_backbone(tmp_path, capsys) -> None:
    """混合状态（部分骨干已重跑）：门控覆盖面小于 test 侧覆盖面，且差异可定位到骨干。"""
    n = 12
    root = tmp_path / "winerr"
    rng = np.random.default_rng(2)
    for bb, marked in (("DLinear", True), ("TimesNet", False)):
        for arm in ("none", "revin"):
            cid = f"{bb}__ETTh1__h96__{arm}__s2021"
            save_window_mse_atomic(root, cid, rng.random(n).astype(np.float32))
            save_window_mse_atomic(root / "val", cid, rng.random(n).astype(np.float32))
            if marked:
                mark_deterministic_order(root / "val", cid, n)
    feats = pd.DataFrame({
        "dataset": ["ETTh1"] * (2 * n),
        "split": ["val"] * n + ["test"] * n,
        "window_index": list(range(n)) * 2,
        "w_a": rng.random(2 * n),
    })
    groups, notes = build_groups(root, feats, verbose=False)
    assert len(groups) == 2, "两个骨干都要保留在 test 侧"
    gate = [g.backbone for g in groups if g.has_val_windows]
    assert gate == ["DLinear"], gate
    assert "TimesNet" in " ".join(notes)
    assert "未打标" in capsys.readouterr().err
