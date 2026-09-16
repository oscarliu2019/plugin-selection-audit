"""逐窗口测试误差记录的回归测试。

背景：phase-1 的 selector 失败的根因是特征只有 8 个互异取值（每个数据集一份，
horizon 之间完全相同），LODO 之下等价于用 7 个样本去预测第 8 个。把决策粒度下沉到
单个测试窗口是唯一能把 N 从 42 抬到 1e4 量级的办法，前提是每格都留下
「按 loader 顺序排列的逐窗口 MSE」，且不同格子的第 i 项必须指向同一个窗口。

这里锁死三件事：
1. 关掉开关时不产生任何额外内存/字段（长跑默认路径零开销）；
2. 打开开关时逐窗口 MSE 的均值等于全局 MSE（保证没算错、没漏窗口）；
3. 顺序与 batch 切分无关（不同 batch_size 得到同一条曲线），否则跨格子按下标配对会错位。
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.runner import StreamingMetrics  # noqa: E402


def _feed(metrics: StreamingMetrics, pred: torch.Tensor, true: torch.Tensor, bs: int) -> None:
    for i in range(0, pred.shape[0], bs):
        metrics.update(pred[i : i + bs], true[i : i + bs])


def test_disabled_by_default_costs_nothing() -> None:
    m = StreamingMetrics(pred_len=8)
    _feed(m, torch.randn(10, 8, 3), torch.randn(10, 8, 3), bs=4)
    assert m.window_mse().size == 0
    assert m.result()["window_mse"].size == 0


def test_window_mean_equals_global_mse() -> None:
    torch.manual_seed(0)
    pred, true = torch.randn(37, 12, 5), torch.randn(37, 12, 5)
    m = StreamingMetrics(pred_len=12, per_window=True)
    _feed(m, pred, true, bs=8)
    win = m.window_mse()

    assert win.shape == (37,)
    assert win.dtype == np.float32
    # 每个窗口的元素数相同，所以窗口均值的均值 == 全局 MSE
    assert m.result()["mse"] == pytest.approx(float(win.mean()), rel=1e-5)
    expected = ((pred - true) ** 2).mean(dim=(1, 2)).numpy()
    np.testing.assert_allclose(win, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("bs", [1, 5, 16, 64])
def test_order_is_independent_of_batch_size(bs: int) -> None:
    """跨格子按下标配对的前提：窗口顺序只由 loader 决定，与 batch 切分无关。"""
    torch.manual_seed(1)
    pred, true = torch.randn(23, 6, 2), torch.randn(23, 6, 2)

    ref = StreamingMetrics(pred_len=6, per_window=True)
    _feed(ref, pred, true, bs=23)  # 单个大 batch 作为基准

    m = StreamingMetrics(pred_len=6, per_window=True)
    _feed(m, pred, true, bs=bs)

    np.testing.assert_allclose(m.window_mse(), ref.window_mse(), rtol=1e-5, atol=1e-6)


def test_empty_loader_is_safe() -> None:
    m = StreamingMetrics(pred_len=4, per_window=True)
    r = m.result()
    assert np.isnan(r["mse"])
    # 空 loader 不能产出一个「看起来有效」的逐窗口数组
    assert r.get("window_mse") is None or np.asarray(r["window_mse"]).size == 0
    assert m.window_mse().size == 0


def test_atomic_save_roundtrip(tmp_path) -> None:
    """np.save 会给非 .npy 路径自动补后缀，直接 rename 临时名会 FileNotFoundError。"""
    from src.runner import save_window_mse_atomic

    win = np.arange(5, dtype=np.float32) / 3.0
    out = save_window_mse_atomic(tmp_path / "winerr", "DLinear__ETTh1__h96__none__s2021", win)

    assert out.name == "DLinear__ETTh1__h96__none__s2021.npy"
    assert out.is_file()
    np.testing.assert_allclose(np.load(out), win)
    # 不留任何临时文件
    assert list(out.parent.glob(".*.tmp")) == []
    assert list(out.parent.glob("*.npy.npy")) == []


def test_atomic_save_overwrites_in_place(tmp_path) -> None:
    from src.runner import save_window_mse_atomic

    d = tmp_path / "winerr"
    save_window_mse_atomic(d, "c", np.zeros(3, dtype=np.float32))
    out = save_window_mse_atomic(d, "c", np.ones(7, dtype=np.float32))
    assert np.load(out).shape == (7,)
    assert len(list(d.glob("*.npy"))) == 1
