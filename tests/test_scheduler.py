"""`src/config.py` + `src/scheduler.py` 自检（无 GPU 也能全部跑）。

重点验证三件与「6–8 周档期」直接相关的事：
1. 矩阵展开与窗口/迭代数计算与尽调表一致（成本估计的分母不能错）；
2. 成本模型的四级回退与「实测覆盖先验」；
3. 断点续跑（已完成格子跳过）与原子聚合。

另含 2026-09-14 code review 的回归（见文件末尾一节）：speed.csv 是**四个 config 共用**的
成本状态，`--probe` 只能对它做增量 upsert，绝不能用某一阶段的 probe 目录整表覆盖/清空
（清空后 `load_speed` 抛 EmptyDataError，会把所有 config 的 status/dry-run/run 一起打挂）。
"""

from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from src.config import (
    Cell,
    MatrixConfig,
    atomic_write_json,
    atomic_write_text,
    enumerate_cells,
    iters_per_epoch,
    windows_per_epoch,
)
from src import scheduler as SCH


@pytest.fixture(scope="module")
def cfg() -> MatrixConfig:
    return MatrixConfig.load()


@pytest.fixture()
def tmp_cfg(cfg: MatrixConfig, tmp_path) -> MatrixConfig:
    """把落盘路径全部改到 tmp，避免测试污染真实 results/。"""
    c = MatrixConfig(copy.deepcopy(cfg.raw), cfg.config_path)
    c.paths.update({
        "cells_dir": str(tmp_path / "cells"),
        "results_csv": str(tmp_path / "results.csv"),
        "speed_csv": str(tmp_path / "speed.csv"),
        "runs_dir": str(tmp_path / "runs"),
    })
    return c


def _phase_cfg(cfg: MatrixConfig, tmp_path, phase: str) -> MatrixConfig:
    """造一个「阶段局部 cells/probe 目录 + 共享 speed.csv」的 config，
    复刻线上四个 yaml 的真实布局（p2/p3 各有自己的 results/<phase>/，
    但 paths.speed_csv 全都指向同一个 results/speed.csv）。"""
    c = MatrixConfig(copy.deepcopy(cfg.raw), cfg.config_path)
    c.paths.update({
        "cells_dir": str(tmp_path / phase / "cells"),
        "results_csv": str(tmp_path / phase / "results.csv"),
        "speed_csv": str(tmp_path / "speed.csv"),        # ← 四个 config 共用同一张表
        "runs_dir": str(tmp_path / phase / "runs"),
    })
    return c


def _probe_json(cfg: MatrixConfig, name: str, **kw) -> None:
    """在该 config 的**阶段局部** probe 目录里放一个测速结果。"""
    d = cfg.path("cells_dir").parent / "probe"
    atomic_write_json(d / f"{name}.json",
                      {"status": "ok", "peak_mem_mib": 800, "n_params": 1000, **kw})


# --------------------------------------------------------------------------- #
# 配置与矩阵展开
# --------------------------------------------------------------------------- #
def test_matrix_declares_the_paper_matrix(cfg: MatrixConfig):
    assert cfg.backbone_names == ["DLinear", "PatchTST", "iTransformer", "TimesNet"]
    assert cfg.plugin_names == ["none", "revin", "san_lite", "fredf"]
    assert cfg.control_plugin == "none"
    assert cfg.dataset_names() == ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Weather",
                                   "Electricity", "Exchange", "ILI"]
    assert "Traffic" in cfg.dataset_names(include_optional=True)     # Traffic 单列为可选
    assert cfg.dataset("ILI")["horizons"] == [24, 36, 48, 60]        # ILI 用短 horizon
    assert cfg.dataset("ETTh1")["horizons"] == [96, 192, 336, 720]
    assert cfg.seq_len("ILI") == 36                                  # ILI 的 look-back 也不同


def test_seed_policy_only_expands_cheap_datasets(cfg: MatrixConfig):
    assert cfg.seeds_for("Electricity") == cfg.seed_policy["base_seeds"]
    assert len(cfg.seeds_for("ETTh1")) == 3                          # 廉价数据集补到 3 seed
    assert set(cfg.seeds_for("ETTh1")) > set(cfg.seeds_for("Electricity"))


def test_windows_per_epoch_matches_feasibility_table(cfg: MatrixConfig):
    """这些数字来自尽调 §2.1，成本模型的分母必须与之一致。"""
    assert windows_per_epoch(cfg, "ETTh1", 96) == 8449
    assert windows_per_epoch(cfg, "Weather", 96) == 36696
    assert windows_per_epoch(cfg, "Electricity", 96) == 18221
    assert windows_per_epoch(cfg, "Traffic", 96) == 12089
    assert windows_per_epoch(cfg, "Exchange", 96) == 5120
    assert windows_per_epoch(cfg, "ILI", 24) == 617


def test_iters_per_epoch_uses_ceiling_because_drop_last_is_false(cfg: MatrixConfig):
    """上游已把 drop_last 统一改为 False → 迭代数向上取整（也是不能抄原论文数字的原因之一）。"""
    w = windows_per_epoch(cfg, "ETTh1", 96)
    bs = cfg.dataset("ETTh1")["batch_size"]
    assert iters_per_epoch(cfg, "ETTh1", 96) == -(-w // bs)
    assert iters_per_epoch(cfg, "ETTh1", 96) * bs >= w


def test_enumerate_cells_counts(cfg: MatrixConfig):
    cells = enumerate_cells(cfg)
    expect = sum(len(cfg.dataset(d)["horizons"]) * len(cfg.seeds_for(d))
                 for d in cfg.dataset_names()) * 4 * 4
    assert len(cells) == expect
    assert len({c.cell_id for c in cells}) == len(cells)             # cell_id 必须唯一
    sub = enumerate_cells(cfg, datasets=["ETTh1"], plugins=["revin"], horizons=[96])
    assert {c.plugin for c in sub} == {"revin"} and {c.pred_len for c in sub} == {96}


def test_cell_id_roundtrip():
    c = Cell("PatchTST", "ETTm2", 336, "san_lite", 2021)
    assert Cell.from_id(c.cell_id) == c
    assert "/" not in c.cell_id and " " not in c.cell_id


def test_atomic_write_json_leaves_no_tmp_file(tmp_path):
    p = tmp_path / "sub" / "a.json"
    atomic_write_json(p, {"a": 1})
    assert json.loads(p.read_text())["a"] == 1
    assert list(tmp_path.rglob("*.tmp.*")) == []


# --------------------------------------------------------------------------- #
# 成本模型
# --------------------------------------------------------------------------- #
def test_prior_cost_is_monotone_in_horizon_and_backbone(cfg: MatrixConfig):
    cm = SCH.CostModel(cfg)
    p96 = cm.prior_ms(Cell("PatchTST", "ETTh1", 96, "none", 2021))
    p720 = cm.prior_ms(Cell("PatchTST", "ETTh1", 720, "none", 2021))
    assert p720 > p96                                               # horizon 越长越慢
    dl = cm.prior_ms(Cell("DLinear", "ETTh1", 96, "none", 2021))
    tn = cm.prior_ms(Cell("TimesNet", "ETTh1", 96, "none", 2021))
    assert tn > 5 * dl                                              # TimesNet 是重模型


def test_amp_discount_applies_only_to_large_datasets(cfg: MatrixConfig):
    cm = SCH.CostModel(cfg)
    assert cfg.dataset("Electricity").get("use_amp") is True
    assert not cfg.dataset("ETTh1").get("use_amp")
    assert cm.amp_speedup < 1.0


def test_measured_speed_overrides_prior_with_fallback_levels(cfg: MatrixConfig, tmp_path):
    cm = SCH.CostModel(cfg)
    target = Cell("DLinear", "ETTh1", 96, "none", 2021)
    prior, src = cm.ms_per_iter(target)
    assert src == "prior"

    speed = tmp_path / "speed.csv"
    pd.DataFrame([{"backbone": "DLinear", "dataset": "ETTh1", "pred_len": 96,
                   "plugin": "none", "ms_per_iter": prior * 3}]).to_csv(speed, index=False)
    c2 = MatrixConfig(copy.deepcopy(cfg.raw), cfg.config_path)
    c2.paths["speed_csv"] = str(speed)
    cm2 = SCH.CostModel(c2)
    assert cm2.load_speed() == 1

    ms, src = cm2.ms_per_iter(target)
    assert src == "measured" and ms == pytest.approx(prior * 3)
    # 同骨干同数据集的其它 horizon → 按实测/先验比例缩放
    ms2, src2 = cm2.ms_per_iter(Cell("DLinear", "ETTh1", 720, "none", 2021))
    assert src2 == "measured_backbone_dataset"
    assert ms2 == pytest.approx(cm2.prior_ms(Cell("DLinear", "ETTh1", 720, "none", 2021)) * 3, rel=1e-6)
    # 同骨干不同数据集 / 完全不同骨干 → 更粗的回退层级
    assert cm2.ms_per_iter(Cell("DLinear", "Weather", 96, "none", 2021))[1] == "measured_backbone"
    assert cm2.ms_per_iter(Cell("TimesNet", "Weather", 96, "none", 2021))[1] == "measured_global"


def test_estimate_scales_with_epochs_and_eval_overhead(cfg: MatrixConfig):
    cm = SCH.CostModel(cfg)
    c = Cell("DLinear", "ETTh1", 96, "none", 2021)
    plan = cm.estimate(c)
    manual = (iters_per_epoch(cfg, "ETTh1", 96) * plan.ms_per_iter / 1000.0
              * cm.expected_epochs * cm.eval_overhead)
    assert plan.est_seconds == pytest.approx(manual)
    assert plan.to_row()["cell_id"] == c.cell_id


# --------------------------------------------------------------------------- #
# 剪枝、排产、断点续跑
# --------------------------------------------------------------------------- #
def test_skip_rules_prune_revin_on_internally_normalized_backbones(cfg: MatrixConfig):
    cells = enumerate_cells(cfg)
    keep, skipped = SCH.apply_skip_rules(cfg, cells)
    assert len(keep) + len(skipped) == len(cells) and skipped
    # 被剪掉的必须全是 revin × 内置归一化骨干
    for s in skipped:
        assert s.cell.plugin == "revin"
        assert cfg.backbone(s.cell.backbone)["internal_norm"] != "none"
    # DLinear（无内置归一化）上的 revin 一格都不能剪
    assert not any(s.cell.backbone == "DLinear" for s in skipped)
    # 验证子集必须被保留（否则「插件冗余」这个论断就没有实证支撑）
    kept_ids = {c.cell_id for c in keep}
    assert any(c.backbone == "PatchTST" and c.plugin == "revin" for c in keep), kept_ids


def test_make_plan_is_sorted_ascending_and_marks_done(tmp_cfg: MatrixConfig):
    plans, skipped, cost = SCH.make_plan(tmp_cfg, datasets=["ETTh1", "ILI"])
    secs = [p.est_seconds for p in plans]
    assert secs == sorted(secs)                                     # 便宜的先跑
    assert all(not p.done for p in plans)

    # 造一个「已完成」的格子 → 必须被标记 done 并从待跑预算里剔除
    first = plans[0].cell
    atomic_write_json(tmp_cfg.path("cells_dir") / f"{first.cell_id}.json",
                      {"status": "ok", **first.to_dict(), "mse": 0.4})
    plans2, _, cost2 = SCH.make_plan(tmp_cfg, datasets=["ETTh1", "ILI"])
    done = [p for p in plans2 if p.done]
    assert [p.cell.cell_id for p in done] == [first.cell_id]
    b1, b2 = SCH.budget_summary(plans, cost.safety), SCH.budget_summary(plans2, cost2.safety)
    assert b2["n_todo"] == b1["n_todo"] - 1
    assert b2["gpu_hours_todo"] < b1["gpu_hours_todo"]
    assert b2["gpu_hours_done"] > 0


def test_smoke_and_failed_files_are_not_counted_as_done(tmp_cfg: MatrixConfig):
    """`X.smoke.json` / `X.json.failed` 不得被误判为已完成（曾经踩过的坑）。"""
    d = tmp_cfg.path("cells_dir")
    atomic_write_json(d / "DLinear__ETTh1__h96__none__s2021.smoke.json", {"status": "ok"})
    atomic_write_json(d / "DLinear__ETTh1__h96__revin__s2021.json.failed", {"status": "failed"})
    assert SCH.done_cell_ids(tmp_cfg) == set()


def test_budget_summary_days_are_consistent(tmp_cfg: MatrixConfig):
    plans, _, cost = SCH.make_plan(tmp_cfg, datasets=["ETTh1"])
    b = SCH.budget_summary(plans, cost.safety)
    assert b["days_24h"] == pytest.approx(b["gpu_hours_todo"] / 24)
    assert b["days_16h"] > b["days_20h"] > b["days_24h"]
    assert cost.safety >= 1.0


def test_full_matrix_dry_run_fits_the_schedule(tmp_cfg: MatrixConfig, capsys):
    """无 GPU 环境下的核心验证：全矩阵排产可算，且主矩阵预算 ≤21 天（20h/天）。"""
    plans, skipped, cost = SCH.make_plan(tmp_cfg)
    b = SCH.budget_summary(plans, cost.safety)
    SCH.print_dry_run(tmp_cfg, plans, skipped, cost, None)
    out = capsys.readouterr().out
    assert "总预算" in out and "剪枝" in out
    assert b["n_total"] > 500
    assert b["days_20h"] < 21.0, f"排产超出档期: {b['days_20h']:.1f} 天"


def test_probe_cells_cover_backbone_dataset_and_plugins(cfg: MatrixConfig):
    cells = SCH.probe_cells(cfg)
    assert {c.plugin for c in cells} == set(cfg.plugin_names)        # 插件开销也要实测
    pairs = {(c.backbone, c.dataset) for c in cells}
    assert len(pairs) == len(cfg.backbone_names) * len(cfg.dataset_names())
    # 每个 (骨干,数据集) 至少覆盖最短与最长 horizon
    for bk in cfg.backbone_names:
        hs = {c.pred_len for c in cells if c.backbone == bk and c.dataset == "ETTm1"}
        assert {96, 720} <= hs


def test_train_cell_command_is_wellformed(cfg: MatrixConfig):
    c = Cell("PatchTST", "ETTh1", 96, "fredf", 2021)
    cmd = SCH._cmd(c, cfg.config_path, "cuda", probe=True, probe_steps=40, save_ckpt=False)
    assert cmd[1:3] == ["-m", "src.train_cell"]
    for flag, val in [("--backbone", "PatchTST"), ("--plugin", "fredf"), ("--pred-len", "96")]:
        assert cmd[cmd.index(flag) + 1] == val
    assert "--probe" in cmd and cmd[cmd.index("--max-train-steps") + 1] == "40"
    assert "--save-checkpoint" not in cmd


# --------------------------------------------------------------------------- #
# 聚合
# --------------------------------------------------------------------------- #
def test_aggregate_collects_only_successful_cells(tmp_cfg: MatrixConfig):
    d = tmp_cfg.path("cells_dir")
    ok = Cell("DLinear", "ETTh1", 96, "none", 2021)
    atomic_write_json(d / f"{ok.cell_id}.json",
                      {"status": "ok", **ok.to_dict(), "mse": 0.38, "mae": 0.40,
                       "seg_mse": [0.3, 0.4, 0.42, 0.45]})
    atomic_write_json(d / "DLinear__ETTh1__h192__none__s2021.json", {"status": "failed", "mse": 9.9})
    # 烟囱测试（带 tag）与 .failed 都不能进主表
    atomic_write_json(d / "DLinear__ETTh1__h336__none__s2021.smoke.json",
                      {"status": "ok", "backbone": "DLinear", "dataset": "ETTh1",
                       "pred_len": 336, "plugin": "none", "seed": 2021, "mse": 9.99})
    atomic_write_json(d / "DLinear__ETTh1__h720__none__s2021.json.failed", {"status": "failed"})
    out = SCH.aggregate(tmp_cfg)
    df = pd.read_csv(out)
    assert len(df) == 1 and df.iloc[0]["mse"] == 0.38
    assert {"seg1_mse", "seg4_mse"} <= set(df.columns)               # 分段指标被展开
    assert df.iloc[0]["backbone"] == "DLinear"


def test_write_speed_csv_from_probe_jsons(tmp_cfg: MatrixConfig):
    probe_dir = tmp_cfg.path("cells_dir").parent / "probe"
    atomic_write_json(probe_dir / "a.json",
                      {"status": "ok", "backbone": "DLinear", "dataset": "ETTh1", "pred_len": 96,
                       "plugin": "none", "ms_per_iter": 12.5, "peak_mem_mib": 800, "n_params": 1000})
    atomic_write_json(probe_dir / "b.json", {"status": "failed"})
    df = pd.read_csv(SCH.write_speed_csv(tmp_cfg))
    assert len(df) == 1 and df.iloc[0]["ms_per_iter"] == 12.5


# --------------------------------------------------------------------------- #
# speed.csv 是跨阶段共享的成本状态（2026-09-14 code review 回归）
# --------------------------------------------------------------------------- #
def test_empty_probe_dir_must_not_wipe_shared_speed_csv(cfg: MatrixConfig, tmp_path, capsys):
    """p2 的 probe 目录为空时，phase-1 已实测的 speed.csv 必须**原样保留**、CLI 不许崩。

    退回旧实现（无条件 `pd.DataFrame(rows).to_csv()` 整表覆盖）时：speed.csv 会被写成只含
    一个 "\\n" 的无表头空文件，下面「内容不变」与「load_speed == 2」的断言都会失败。
    """
    p1 = _phase_cfg(cfg, tmp_path, "phase1")
    p2 = _phase_cfg(cfg, tmp_path, "p2")
    shared = p1.path("speed_csv")

    _probe_json(p1, "a", backbone="DLinear", dataset="ETTh1", pred_len=96,
                plugin="none", ms_per_iter=11.0)
    _probe_json(p1, "b", backbone="TimesNet", dataset="Electricity", pred_len=720,
                plugin="fredf", ms_per_iter=203.5)
    SCH.write_speed_csv(p1)
    before = shared.read_text(encoding="utf-8")
    assert before.strip(), "phase-1 的实测值本来就该写进共享速度表"

    # ① p2 的 probe 目录根本不存在（线上最常见的形态）→ 既不许写，更不许清
    assert not (p2.path("cells_dir").parent / "probe").exists()
    SCH.write_speed_csv(p2)
    assert shared.read_text(encoding="utf-8") == before, "阶段局部的空 probe 目录清空了共享成本表"
    assert "保留原有速度表" in capsys.readouterr().out

    # ② 目录存在但只有失败的 probe（status != ok）→ 同样不许写
    _probe_json(p2, "failed_one", status="failed")
    SCH.write_speed_csv(p2)
    assert shared.read_text(encoding="utf-8") == before

    # ③ 成本表仍然完整可读，两行实测值都在，排产照旧用 measured 而不是 prior
    cm = SCH.CostModel(p2)
    assert cm.load_speed() == 2
    assert cm.measured[("DLinear", "ETTh1", 96, "none")] == pytest.approx(11.0)
    assert cm.measured[("TimesNet", "Electricity", 720, "fredf")] == pytest.approx(203.5)
    plans, _, _ = SCH.make_plan(p2, backbones=["DLinear"], datasets=["ETTh1"],
                                plugins=["none"], horizons=[96])
    assert {p.cost_source for p in plans} == {"measured"}


def test_two_phases_upsert_without_overwriting_each_other(cfg: MatrixConfig, tmp_path):
    """两个阶段各自 probe：新键追加、同键以本次实测为准、其它阶段的行一行不丢。

    退回旧实现（整表覆盖）时 phase-1 独有的 PatchTST 行会消失，本用例的 `got` 断言失败。
    """
    p1 = _phase_cfg(cfg, tmp_path, "phase1")
    p2 = _phase_cfg(cfg, tmp_path, "p2")
    shared = p1.path("speed_csv")
    assert p2.path("speed_csv") == shared                            # 前提：两阶段共用一张表

    _probe_json(p1, "a", backbone="DLinear", dataset="ETTh1", pred_len=96,
                plugin="none", ms_per_iter=11.0)
    _probe_json(p1, "b", backbone="PatchTST", dataset="ETTh1", pred_len=96,
                plugin="none", ms_per_iter=40.0)
    SCH.write_speed_csv(p1)

    # p2 只测两格：一个**新键**（新插件），一个与 phase-1 **同键**（应当被本次实测覆盖）
    _probe_json(p2, "c", backbone="DLinear", dataset="ETTh1", pred_len=96,
                plugin="san_lite", ms_per_iter=13.0)
    _probe_json(p2, "d", backbone="DLinear", dataset="ETTh1", pred_len=96,
                plugin="none", ms_per_iter=12.0)
    SCH.write_speed_csv(p2)

    df = pd.read_csv(shared)
    got = {(r.backbone, r.dataset, int(r.pred_len), r.plugin): float(r.ms_per_iter)
           for r in df.itertuples(index=False)}
    assert got == {
        ("DLinear", "ETTh1", 96, "none"): 12.0,        # 同键 → 本次实测覆盖旧值
        ("DLinear", "ETTh1", 96, "san_lite"): 13.0,    # 新键 → 追加
        ("PatchTST", "ETTh1", 96, "none"): 40.0,       # phase-1 独有 → 必须原样保留
    }
    assert len(df) == 3                                # 主键唯一，不许留重复行
    # 附属列（显存/参数量）也要跟着 upsert，不能只留下主键
    assert {"peak_mem_mib", "n_params"} <= set(df.columns)


def test_cli_survives_a_corrupt_speed_csv(cfg: MatrixConfig, tmp_path, capsys):
    """speed.csv 空/损坏时，`--status`/`--dry-run`/`--run` 共用的 make_plan 必须照样能跑。

    退回旧实现（`pd.read_csv` 裸调）时 `pandas.errors.EmptyDataError` 会一路冒到 CLI，
    所有 config 的 status/dry-run/run 以及 watchdog 的 status 轮询同时挂掉。
    """
    c = _phase_cfg(cfg, tmp_path, "phase1")
    speed = c.path("speed_csv")

    for text in ("\n", "", "backbone,dataset\nDLinear,ETTh1\n", "\x00\x01 not a csv"):
        atomic_write_text(speed, text)
        cm = SCH.CostModel(c)
        assert cm.load_speed() == 0                                  # 读不动 → 按「无实测值」
        plans, _, _ = SCH.make_plan(c, datasets=["ILI"])             # 不许抛异常
        assert plans and {p.cost_source for p in plans} == {"prior"}
        SCH.status(c, include_optional=False)                        # status 轮询也必须活着
    assert "警告" in capsys.readouterr().out


def test_load_speed_skips_dirty_rows_but_keeps_good_ones(cfg: MatrixConfig, tmp_path):
    """单行脏值（空的 ms_per_iter / 非数 / 非正）只能丢这一行，不能拖垮整张成本表。"""
    c = _phase_cfg(cfg, tmp_path, "phase1")
    atomic_write_text(c.path("speed_csv"),
                      "backbone,dataset,pred_len,plugin,ms_per_iter\n"
                      "DLinear,ETTh1,96,none,11.0\n"
                      "DLinear,ETTh1,192,none,\n"           # 落盘中断留下的半行
                      "DLinear,ETTh1,336,none,abc\n"        # 非数
                      "DLinear,ETTh1,720,none,0\n")         # 0 ms/iter 无意义
    cm = SCH.CostModel(c)
    assert cm.load_speed() == 1
    assert list(cm.measured) == [("DLinear", "ETTh1", 96, "none")]
