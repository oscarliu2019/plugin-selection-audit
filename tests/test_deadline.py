"""硬截止（``--deadline``）的行为测试。

这套逻辑是周末无人值守的唯一保险：它决定「周一早上是一份能看的矩阵」还是
「一个跑了 5 小时被砍掉的半成品 + 一堆没跑的洞」。所以必须有测试。
"""

from __future__ import annotations

import time

import pytest

from src.config import MatrixConfig
from src.scheduler import Cell, CellPlan, parse_deadline, run_queue


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def test_parse_deadline_formats():
    a = parse_deadline("2026-09-14 12:00")
    b = parse_deadline("2026-09-14T12:00")
    c = parse_deadline("2026-09-14 12:00:00")
    assert a == b == c
    assert time.strftime("%F %H:%M", time.localtime(a)) == "2026-09-14 12:00"
    assert parse_deadline(None) is None and parse_deadline("") is None
    y = time.localtime().tm_year
    assert time.localtime(parse_deadline("09-14 08:00")).tm_year == y


def test_parse_deadline_rejects_garbage():
    with pytest.raises(ValueError):
        parse_deadline("下周一中午")


# --------------------------------------------------------------------------- #
# 投放决策
# --------------------------------------------------------------------------- #
def _plan(sec: float, name: str) -> CellPlan:
    return CellPlan(Cell("DLinear", "ETTh1", 96, name, 2021), 10, 1.0, sec, "prior", False)


def _capture_dispatch(monkeypatch) -> list[str]:
    """把子进程执行换成记录 cell_id，纯逻辑测试，不真的训练。"""
    launched: list[str] = []

    class FakeProc:
        returncode = 0

        def poll(self):
            return 0

    def fake_popen(cmd, **kw):
        launched.append(cmd[cmd.index("--plugin") + 1])
        return FakeProc()

    def fake_run(cmd, **kw):
        launched.append(cmd[cmd.index("--plugin") + 1])
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr("src.scheduler.aggregate", lambda cfg: None)
    monkeypatch.setattr("time.sleep", lambda s: None)
    return launched


@pytest.mark.parametrize("workers", [1, 3])
def test_deadline_skips_cells_that_cannot_finish(monkeypatch, workers):
    """剩 1 小时的时候，2 小时的格子必须被跳过，20 分钟的格子必须照跑。"""
    cfg = MatrixConfig.load()
    launched = _capture_dispatch(monkeypatch)
    plans = [_plan(1200, "cheap"), _plan(7200, "expensive")]   # 20min / 2h
    run_queue(cfg, plans, device="cpu", workers=workers,
              deadline_ts=time.time() + 3600)                  # 剩 1 小时
    assert "cheap" in launched
    assert "expensive" not in launched, "预估超过剩余时间的格子不该被投放"


@pytest.mark.parametrize("workers", [1, 3])
def test_past_deadline_launches_nothing(monkeypatch, workers):
    cfg = MatrixConfig.load()
    launched = _capture_dispatch(monkeypatch)
    run_queue(cfg, [_plan(60, "a"), _plan(60, "b")], device="cpu", workers=workers,
              deadline_ts=time.time() - 1)
    assert launched == []


@pytest.mark.parametrize("workers", [1, 3])
def test_no_deadline_runs_everything(monkeypatch, workers):
    cfg = MatrixConfig.load()
    launched = _capture_dispatch(monkeypatch)
    run_queue(cfg, [_plan(1e9, "huge"), _plan(60, "small")], device="cpu", workers=workers)
    assert set(launched) == {"huge", "small"}


def test_safety_margin_makes_deadline_conservative(monkeypatch):
    """安全系数 1.2：预估 55 分钟的格子在剩 60 分钟时应被判定为来不及。"""
    cfg = MatrixConfig.load()
    assert float(cfg.cost_model["safety_margin"]) > 1.0
    launched = _capture_dispatch(monkeypatch)
    run_queue(cfg, [_plan(55 * 60, "borderline")], device="cpu", workers=3,
              deadline_ts=time.time() + 60 * 60)
    assert launched == [], "55min × 1.2 = 66min > 60min，应保守跳过"


def test_cheapest_first_dispatch_order(monkeypatch):
    """并发模式必须按「便宜的先跑」投放：有截止时先把矩阵铺满比先啃硬骨头重要。"""
    cfg = MatrixConfig.load()
    launched = _capture_dispatch(monkeypatch)
    plans = sorted([_plan(300, "mid"), _plan(60, "cheap"), _plan(900, "pricey")],
                   key=lambda p: p.est_seconds)
    run_queue(cfg, plans, device="cpu", workers=1)
    assert launched == ["cheap", "mid", "pricey"]
