"""`src/horizon.py` 自检：手工构造 epoch 曲线，逐条验证每个准则的定义。

这些用例同时说明「为什么单一验证 MSE 会选错 checkpoint」——本文附加贡献的立论基础。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import horizon as HZ


def _logs(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def make_cell(cell_id: str, val_segs: list[list[float]], test: list[float] | None = None) -> pd.DataFrame:
    """按 (epoch × 段) 的验证矩阵造日志；test_mse 默认取验证均值 ×1.02。"""
    rows = []
    for e, segs in enumerate(val_segs):
        rec = {"cell_id": cell_id, "epoch": e, "val_mse": float(np.mean(segs))}
        rec["test_mse"] = float(test[e]) if test else float(np.mean(segs) * 1.02)
        rec.update({f"val_seg{i+1}_mse": float(v) for i, v in enumerate(segs)})
        rec.update({f"test_seg{i+1}_mse": float(v * 1.02) for i, v in enumerate(segs)})
        rows.append(rec)
    return _logs(rows)


# --------------------------------------------------------------------------- #
# 单个准则的定义
# --------------------------------------------------------------------------- #
def test_criteria_registry_contains_baseline_and_oracle():
    assert "val_mse" in HZ.CRITERIA and "oracle_test" in HZ.CRITERIA
    assert len(HZ.CRITERIA) == 7


def test_val_mse_and_last_and_oracle():
    g = make_cell("c", [[1.0, 1.0], [0.5, 0.5], [0.6, 0.6]], test=[1.1, 0.7, 0.4])
    assert HZ.crit_val_mse(g) == 1
    assert HZ.crit_last(g) == 2
    assert HZ.crit_oracle(g) == 2          # oracle 偷看测试集，只作为上界


def test_long_seg_prefers_the_longest_horizon_segment():
    """核心现象：epoch1 的总体验证 MSE 最优（0.80 < 0.95），但最长 horizon 段在 epoch2 才最优。"""
    g = make_cell("c", [[1.0, 2.0], [0.2, 1.4], [0.7, 1.2]])
    assert HZ.crit_val_mse(g) == 1         # 平均值（0.80）选 epoch1
    assert HZ.crit_long_seg(g) == 2        # 长 horizon 段（1.2 < 1.4）选 epoch2
    assert HZ.crit_worst_seg(g) == 2       # minimax 也选 epoch2


def test_rank_agg_gives_each_segment_one_vote():
    """段间量级差异大时，平均秩不会被大量级段主导。"""
    g = make_cell("c", [[0.10, 10.0], [0.20, 9.5], [0.08, 9.9]])
    # 段2 的量级是段1 的 ~100 倍，总体均值几乎只反映段2 → 选 epoch1
    assert HZ.crit_val_mse(g) == 1
    # 平均秩：段1 排名 (2,3,1)、段2 排名 (3,1,2) → epoch2 的平均秩最小
    assert HZ.crit_rank_agg(g) == 2


def test_slope_penalized_avoids_degrading_checkpoints():
    """两个 epoch 的总体验证 MSE 接近，但一个「越长越糟」→ 应选另一个。"""
    g = make_cell("c", [[0.5, 1.45], [0.9, 1.10]])
    assert HZ.crit_val_mse(g) == 0                # 0.975 < 1.000 → 默认准则选 epoch0
    assert HZ.crit_slope_penalized(g) == 1        # 段间退化斜率惩罚后改选平坦的 epoch1
    flat = make_cell("c", [[1.0, 1.0], [1.02, 1.02]])
    assert HZ.crit_slope_penalized(flat) == 0     # 无退化时退化为 val_mse


def test_criteria_degrade_gracefully_without_segment_columns():
    g = pd.DataFrame({"cell_id": ["c"] * 3, "epoch": [0, 1, 2],
                      "val_mse": [1.0, 0.5, 0.6], "test_mse": [1.0, 0.5, 0.6]})
    for name in ("rank_agg", "long_seg", "worst_seg", "slope_penalized"):
        assert HZ.CRITERIA[name](g) == HZ.crit_val_mse(g), name


# --------------------------------------------------------------------------- #
# 评估流程
# --------------------------------------------------------------------------- #
def test_select_epochs_and_compare(tmp_path):
    logs = pd.concat([
        make_cell("cellA", [[1.0, 2.0], [0.2, 1.4], [0.7, 1.2]], test=[1.5, 0.95, 0.80]),
        make_cell("cellB", [[1.0, 2.0], [0.2, 1.4], [0.7, 1.2]], test=[1.5, 0.90, 0.85]),
    ])
    sel = HZ.select_epochs(logs)
    assert len(sel) == 2
    assert {"val_mse_epoch", "val_mse_test_mse", "oracle_test_epoch"} <= set(sel.columns)

    cmp = HZ.compare_criteria(sel)
    assert list(cmp["mean_test_mse"]) == sorted(cmp["mean_test_mse"])   # 按测试 MSE 升序
    base = cmp[cmp.criterion == "val_mse"].iloc[0]
    assert base["mean_rel_gain_vs_baseline_pct"] == pytest.approx(0.0)
    orc = cmp[cmp.criterion == "oracle_test"].iloc[0]
    assert orc["mean_gap_to_oracle_pct"] == pytest.approx(0.0)
    assert orc["mean_test_mse"] <= cmp["mean_test_mse"].min() + 1e-12   # oracle 是下界
    # cellA 上 long_seg 抓到了更好的 checkpoint
    long_seg = cmp[cmp.criterion == "long_seg"].iloc[0]
    assert long_seg["mean_test_mse"] < base["mean_test_mse"]


def test_run_writes_artifacts_and_wilcoxon(tmp_path):
    rng = np.random.default_rng(0)
    cells = []
    for c in range(12):
        segs = []
        for e in range(6):
            short = 0.3 + 0.02 * (e - 2) ** 2
            long = 0.6 + 0.02 * (e - 4) ** 2            # 长 horizon 段更晚收敛
            segs.append([short * (1 + rng.normal(0, 0.01)), long * (1 + rng.normal(0, 0.01))])
        cells.append(make_cell(f"c{c}", segs))
    out = HZ.run(pd.concat(cells), tmp_path)
    for f in ["selected_epochs.csv", "criteria_comparison.csv", "criteria_wilcoxon.csv"]:
        assert (tmp_path / f).exists(), f
    names = {r["method"] for r in out["wilcoxon"]}
    assert "val_mse" not in names and "rank_agg" in names          # 基线不与自己比较
    cmp = pd.DataFrame(out["comparison"]).set_index("criterion")
    assert cmp.loc["last", "mean_test_mse"] > cmp.loc["val_mse", "mean_test_mse"]  # 不早停更差
    assert cmp.loc["oracle_test", "mean_epoch"] >= 0


def test_load_epoch_logs_parses_runner_jsonl(tmp_path):
    """与 runner 落盘格式对齐：runs/<cell_id>/epochs.jsonl，逐行 JSON。"""
    d = tmp_path / "DLinear__ETTh1__h96__none__s2021"
    d.mkdir(parents=True)
    lines = [
        {"epoch": 0, "val_mse": 0.5, "test_mse": 0.55,
         "val_seg_mse": [0.4, 0.6], "test_seg_mse": [0.45, 0.65]},
        {"epoch": 1, "val_mse": 0.45, "test_mse": 0.50,
         "val_seg_mse": [0.35, 0.55], "test_seg_mse": [0.40, 0.60]},
    ]
    (d / "epochs.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n\n")
    logs = HZ.load_epoch_logs(tmp_path)
    assert len(logs) == 2
    assert logs["cell_id"].nunique() == 1
    assert logs.loc[1, "val_seg2_mse"] == 0.55
    assert HZ.crit_val_mse(logs) == 1
