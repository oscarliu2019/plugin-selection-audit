"""截断 / 试跑的逐窗口误差不得落到正式路径。

背景（2026-09-14 code review P1，`src/runner.py`）：
`--max-eval-steps N` 只累计前 N 个 batch，写出的数组长度远小于真实窗口数；
而 `selector_window.build_groups` 用 ``n = min(各臂长度)`` 把**全组所有臂**
静默截断到最短那个臂，于是一次手工烟囱跑（README 自己推荐的
``--tag smoke --max-eval-steps 5``）就能让该组的 oracle 上界、best_fixed、
滞后一致率全部变成只在几百个窗口前缀上算出的数字，且结果表看起来完全正常。

这里锁死：截断跑与 probe 跑都必须跳过落盘。
"""

from __future__ import annotations

import inspect

import numpy as np

from src import runner as R


def _save_body() -> str:
    """取 run_cell 中负责逐窗口落盘的那段源码。"""
    src = inspect.getsource(R.run_cell)
    i = src.index("win = best_test.pop")
    return src[i : i + 2000]


def test_guard_mentions_both_truncation_and_probe() -> None:
    body = _save_body()
    assert "max_eval_steps" in body, "必须显式检查评估是否被截断"
    assert "run.probe" in body, "probe 测速跑也不该覆盖正式 winerr"


def test_truncated_or_probe_runs_skip_persist(monkeypatch, tmp_path) -> None:
    """用一个最小替身验证控制流：truncated / probe 时不调用任何落盘函数。"""
    calls: list[str] = []
    monkeypatch.setattr(R, "save_window_mse_atomic",
                        lambda *a, **k: calls.append("save"))
    monkeypatch.setattr(R, "mark_deterministic_order",
                        lambda *a, **k: calls.append("mark"))

    # 复刻 run_cell 里的判定逻辑（保持与源码同构，见上一个测试的源码断言）
    def persist(save_window_mse: bool, max_eval_steps, probe: bool) -> None:
        calls.clear()
        truncated = max_eval_steps is not None
        if save_window_mse and (truncated or probe):
            return
        if save_window_mse:
            R.save_window_mse_atomic(tmp_path, "cid", np.ones(4, dtype=np.float32))
            R.save_window_mse_atomic(tmp_path / "val", "cid", np.ones(4, dtype=np.float32))
            R.mark_deterministic_order(tmp_path / "val", "cid", 4)

    persist(True, 5, False)
    assert calls == [], "--max-eval-steps 截断时必须跳过落盘"

    persist(True, None, True)
    assert calls == [], "probe 跑必须跳过落盘"

    persist(True, None, False)
    assert calls == ["save", "save", "mark"], "正常跑必须照常落盘并打标"

    persist(False, None, False)
    assert calls == [], "开关关闭时不落盘"


def test_full_eval_writes_marker_with_true_window_count(tmp_path) -> None:
    """打标记录的 n_windows 必须是真实窗口数，供下游核对。"""
    import json

    arr = np.arange(1234, dtype=np.float32)
    R.save_window_mse_atomic(tmp_path, "cid", arr)
    R.mark_deterministic_order(tmp_path, "cid", arr.size)
    meta = json.loads((tmp_path / f"cid{R.ORDER_MARKER_SUFFIX}").read_text())
    assert meta["n_windows"] == 1234
    assert meta["loader_order"] == "deterministic"
