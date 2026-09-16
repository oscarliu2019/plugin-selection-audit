"""⭐ 排产调度器：6–8 周档期能否守住的关键。

流水线
------
1. **成本先验**：从 configs/matrix.yaml 的 cost_model 算出每格的 ms/iter 与预估墙钟；
2. **1-epoch 测速**（`--probe`）：跑少量 iter 实测 ms/iter，按
   (骨干,数据集,horizon,插件) **增量 upsert** 进 results/speed.csv
   （该表被四个 config 共用，因此只能增量更新，不能整表覆盖；见 write_speed_csv）；
3. **成本模型更新**：实测值按 (骨干,数据集,horizon,插件) → (骨干,数据集) → (骨干) → 全局
   四级回退覆盖先验；
4. **升序排产**：便宜的格子先跑，保证「任何时刻停下来都已经有一张能用的表」；
5. **断点续跑**：`results/cells/<cell_id>.json` 存在即视为完成，直接跳过；
6. **原子落盘**：每格结果单独原子写 JSON，再原子替换聚合出 results/results.csv。

目录约定（configs/matrix.yaml 的 paths 段）
    runs/<cell_id>/{configs.json, epochs.jsonl, checkpoint.pth}
    results/cells/<cell_id>.json      # 成功
    results/cells/<cell_id>.json.failed  # 失败（带 traceback）
    results/probe/<cell_id>.json      # 测速
    results/results.csv               # 聚合主表
    results/speed.csv                 # 实测速度表
    <artifacts_dir>/plan.csv          # 排产计划（每个 config 各自一份）

用法
----
    python -m src.scheduler --dry-run                    # 无 GPU 也能跑：打印排产与总预算
    python -m src.scheduler --dry-run --include-optional # 把 Traffic 也算进来
    python -m src.scheduler --probe --probe-steps 40     # 有 GPU：先测速
    python -m src.scheduler --run --budget-hours 20      # 按预算跑，超时优雅停止
    python -m src.scheduler --aggregate                  # 汇总 CSV
    python -m src.scheduler --status                     # 进度概览
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import (
    REPO_ROOT,
    Cell,
    CellPlan,
    MatrixConfig,
    atomic_write_text,
    enumerate_cells,
    iters_per_epoch,
)


# --------------------------------------------------------------------------- #
# 速度表（speed.csv）：跨 config 共享的成本状态
# --------------------------------------------------------------------------- #
# 一行实测速度的主键。speed.csv 是**四个 config 共用**的一张表（四个 yaml 的
# paths.speed_csv 都指向 results/speed.csv），所以任何写入都必须按这个主键做增量
# upsert，不能整表覆盖——见 write_speed_csv 的注释（2026-09-14 code review 修复项）。
SPEED_KEY = ["backbone", "dataset", "pred_len", "plugin"]
SPEED_COLS = SPEED_KEY + ["ms_per_iter"]


def read_speed_table(path: Path) -> pd.DataFrame:
    """容错读取 speed.csv：文件不存在/空文件/损坏/缺列时返回**空表**而不是抛异常。

    2026-09-14 code review 修复项（fail-safe 而不是崩掉整个 CLI）：
    ``make_plan`` 必调 ``CostModel.load_speed``，而 ``--status`` / ``--dry-run`` /
    ``--run`` 都走 ``make_plan``。历史上只要 speed.csv 被写成无表头空文件（写端整表
    覆盖 + 本次 probe 目录为空），``pd.read_csv`` 就抛 ``EmptyDataError``，
    于是**所有 config 的所有子命令**连带 watchdog 的 status 轮询一起挂掉。
    成本表只是「加速排产的先验」，读不动时退回纯先验完全可用，绝不该让 CLI 崩。
    """
    if not path.exists():
        return pd.DataFrame(columns=SPEED_COLS)
    try:
        df = pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError, OSError) as exc:
        print(f"[sched] 警告：速度表 {path} 无法解析（{type(exc).__name__}: {exc}），"
              f"本次按「无实测值」处理，排产退回先验")
        return pd.DataFrame(columns=SPEED_COLS)
    if df.empty or not set(SPEED_COLS) <= set(df.columns):
        print(f"[sched] 警告：速度表 {path} 为空或缺列（现有列 {list(df.columns)}），"
              f"本次按「无实测值」处理")
        return pd.DataFrame(columns=SPEED_COLS)
    return df


# --------------------------------------------------------------------------- #
# 成本模型
# --------------------------------------------------------------------------- #
class CostModel:
    """ms/iter 的先验 + 实测覆盖（四级回退）。"""

    def __init__(self, cfg: MatrixConfig) -> None:
        self.cfg = cfg
        cm = cfg.cost_model
        self.base_ms: dict[str, float] = dict(cm["dataset_base_ms"])
        self.expected_epochs = float(cm["expected_epochs"])
        self.eval_overhead = float(cm["eval_overhead"])
        self.amp_speedup = float(cm["amp_speedup"])
        self.safety = float(cm["safety_margin"])
        self.measured: dict[tuple[str, str, int, str], float] = {}
        # 逐骨干校准系数（实测/预估的中位数）。见 load_calib() 的说明。
        self.calib: dict[str, float] = {}

    # ---- 实测值载入 ----
    def load_speed(self, path: Path | None = None) -> int:
        """载入 speed.csv 的实测 ms/iter；读不动或某行不可用时**跳过该行**而不是抛异常。

        2026-09-14 code review 修复项：这里是整个 CLI 的必经之路（make_plan → load_speed），
        因此对「空文件 / 损坏文件 / 单行脏值」一律 fail-safe——最坏结果只是这一格退回先验，
        而不是让 `--status` / `--dry-run` / `--run` 全部崩掉。
        """
        p = path or self.cfg.path("speed_csv")
        df = read_speed_table(p)
        for r in df.itertuples(index=False):
            try:
                ms = float(r.ms_per_iter)
                key = (str(r.backbone), str(r.dataset), int(r.pred_len), str(r.plugin))
            except (TypeError, ValueError):
                continue                                  # 脏行（NaN/空串/非数）直接忽略
            if not math.isfinite(ms) or ms <= 0:
                continue                                  # 0 或负的 ms/iter 是无意义的实测值
            self.measured[key] = ms
        return len(self.measured)

    def load_calib(self, path: Path | None = None) -> int:
        """载入逐骨干校准系数 ``{backbone: k}``，令 est ← est × k。

        为什么还需要它：``expected_epochs`` 是**全矩阵一个标量**，但早停的实际轮数强烈
        依赖骨干——phase-1 实测 pred/actual 的中位数是 DLinear 0.60、PatchTST 1.10、
        iTransformer 1.73、TimesNet 2.28，也就是说同一组 (epochs, overhead) 标量下
        TimesNet 被高估 2.3 倍（它每轮太慢，patience 很早就触发），
        DLinear 被低估 1.7 倍。一个标量修不了这种系统性差异。

        k 是纯数据（由 tools/calibrate_speed.py 从 results.csv 反解），
        因此放在 results/cost_calib.json 而不是 yaml——配置描述实验设计，不该塞进实测值。
        """
        p = path or Path(self.cfg.paths.get("calib_json", "results/cost_calib.json"))
        p = p if p.is_absolute() else (REPO_ROOT / p)
        if not p.exists():
            return 0
        self.calib = {k: float(v) for k, v in json.loads(p.read_text("utf-8")).items()}
        return len(self.calib)

    # ---- 先验 ----
    def prior_ms(self, cell: Cell) -> float:
        ds = self.cfg.dataset(cell.dataset)
        bk = self.cfg.backbone(cell.backbone)
        # 实测表里的插件名可能不属于当前配置（例如用 phase-1 的 speed.csv 给
        # FreDF α 扫描组排产），此时按 1.0 的开销处理而不是抛 KeyError。
        try:
            pl = self.cfg.plugin(cell.plugin)
        except KeyError:
            pl = {}
        base = float(self.base_ms[cell.dataset]) * float(bk["cost_factor"])
        seq = self.cfg.seq_len(cell.dataset)
        base_pred = int(min(ds["horizons"]))
        hf = ((seq + cell.pred_len) / (seq + base_pred)) ** float(bk["horizon_cost_exponent"])
        ms = base * hf * float(pl.get("cost_overhead", 1.0))
        if ds.get("use_amp"):
            ms *= self.amp_speedup
        return ms

    # ---- 四级回退 ----
    def ms_per_iter(self, cell: Cell) -> tuple[float, str]:
        key = (cell.backbone, cell.dataset, cell.pred_len, cell.plugin)
        if key in self.measured:
            return self.measured[key], "measured"
        # 同 (骨干,数据集) 的其它 horizon/插件：按先验比例缩放
        same = [
            (k, v) for k, v in self.measured.items() if k[0] == cell.backbone and k[1] == cell.dataset
        ]
        if same:
            ratios = [v / max(self.prior_ms(Cell(k[0], k[1], k[2], k[3], cell.seed)), 1e-9) for k, v in same]
            return self.prior_ms(cell) * (sum(ratios) / len(ratios)), "measured_backbone_dataset"
        same_bk = [(k, v) for k, v in self.measured.items() if k[0] == cell.backbone]
        if same_bk:
            ratios = [v / max(self.prior_ms(Cell(k[0], k[1], k[2], k[3], cell.seed)), 1e-9)
                      for k, v in same_bk]
            return self.prior_ms(cell) * (sum(ratios) / len(ratios)), "measured_backbone"
        if self.measured:
            ratios = [
                v / max(self.prior_ms(Cell(k[0], k[1], k[2], k[3], cell.seed)), 1e-9)
                for k, v in self.measured.items()
            ]
            return self.prior_ms(cell) * (sum(ratios) / len(ratios)), "measured_global"
        return self.prior_ms(cell), "prior"

    # ---- 单格墙钟 ----
    def estimate(self, cell: Cell, done: bool = False) -> CellPlan:
        ipe = iters_per_epoch(self.cfg, cell.dataset, cell.pred_len)
        ms, src = self.ms_per_iter(cell)
        sec = (ipe * ms / 1000.0 * self.expected_epochs * self.eval_overhead
               * self.calib.get(cell.backbone, 1.0))
        return CellPlan(cell, ipe, ms, sec, src, done)


# --------------------------------------------------------------------------- #
# 剪枝规则
# --------------------------------------------------------------------------- #
@dataclass
class SkipInfo:
    cell: Cell
    rule_id: str
    reason: str


def apply_skip_rules(cfg: MatrixConfig, cells: list[Cell]) -> tuple[list[Cell], list[SkipInfo]]:
    """按 matrix.yaml 的 skip_rules 剪掉「可先验判定为 no-op」的格子，保留验证子集。"""
    rules = cfg.raw.get("skip_rules") or []
    keep: list[Cell] = []
    skipped: list[SkipInfo] = []
    for c in cells:
        hit = None
        for r in rules:
            if r.get("plugin") and r["plugin"] != c.plugin:
                continue
            want_norm = r.get("backbone_internal_norm")
            if want_norm and cfg.backbone(c.backbone).get("internal_norm") != want_norm:
                continue
            sub = r.get("keep_verification_subset") or {}
            in_sub = (
                (not sub.get("datasets") or c.dataset in sub["datasets"])
                and (not sub.get("horizons") or c.pred_len in sub["horizons"])
                and (not sub.get("seeds") or c.seed in sub["seeds"])
            )
            if in_sub:
                continue
            hit = r
            break
        if hit is None:
            keep.append(c)
        else:
            skipped.append(SkipInfo(c, hit.get("id", "?"), hit.get("reason", "")))
    return keep, skipped


# --------------------------------------------------------------------------- #
# 排产
# --------------------------------------------------------------------------- #
def done_cell_ids(cfg: MatrixConfig) -> set[str]:
    d = cfg.path("cells_dir")
    if not d.exists():
        return set()
    return {p.stem for p in d.glob("*.json") if "." not in p.stem}


def make_plan(
    cfg: MatrixConfig,
    include_optional: bool = False,
    apply_skips: bool = True,
    **filters: Any,
) -> tuple[list[CellPlan], list[SkipInfo], CostModel]:
    cost = CostModel(cfg)
    cost.load_speed()
    cost.load_calib()
    cells = enumerate_cells(cfg, include_optional=include_optional, **filters)
    skipped: list[SkipInfo] = []
    if apply_skips:
        cells, skipped = apply_skip_rules(cfg, cells)
    done = done_cell_ids(cfg)
    plans = [cost.estimate(c, done=c.cell_id in done) for c in cells]
    plans.sort(key=lambda p: (p.est_seconds, p.cell.cell_id))
    return plans, skipped, cost


def budget_summary(plans: Iterable[CellPlan], safety: float) -> dict[str, float]:
    todo = [p for p in plans if not p.done]
    total = sum(p.est_seconds for p in todo) * safety
    done_sec = sum(p.est_seconds for p in plans if p.done) * safety
    return {
        "n_total": float(len(list(plans))),
        "n_todo": float(len(todo)),
        "gpu_hours_todo": total / 3600.0,
        "gpu_hours_done": done_sec / 3600.0,
        "days_24h": total / 3600.0 / 24.0,
        "days_20h": total / 3600.0 / 20.0,
        "days_16h": total / 3600.0 / 16.0,
    }


def _plan_frame(plans: list[CellPlan]) -> pd.DataFrame:
    return pd.DataFrame([p.to_row() for p in plans])


def print_dry_run(cfg: MatrixConfig, plans: list[CellPlan], skipped: list[SkipInfo],
                  cost: CostModel, plan_out: Path | None) -> None:
    df = _plan_frame(plans)
    safety = cost.safety
    b = budget_summary(plans, safety)

    print("=" * 96)
    print(f"PlugGate 排产计划   配置: {cfg.config_path}")
    print(f"成本来源分布: {df['cost_source'].value_counts().to_dict()}")
    print("=" * 96)

    print("\n[1] 按数据集汇总（含安全系数 %.2f）" % safety)
    g = df.assign(h=df.est_hours * safety).groupby("dataset").agg(
        cells=("cell_id", "count"), gpu_hours=("h", "sum"), max_cell_h=("h", "max")
    ).sort_values("gpu_hours", ascending=False)
    print(g.round(2).to_string())

    print("\n[2] 按骨干汇总")
    g2 = df.assign(h=df.est_hours * safety).groupby("backbone").agg(
        cells=("cell_id", "count"), gpu_hours=("h", "sum")
    ).sort_values("gpu_hours", ascending=False)
    print(g2.round(2).to_string())

    print("\n[3] 按插件汇总")
    g3 = df.assign(h=df.est_hours * safety).groupby("plugin").agg(
        cells=("cell_id", "count"), gpu_hours=("h", "sum")
    ).sort_values("gpu_hours", ascending=False)
    print(g3.round(2).to_string())

    print("\n[4] 最贵的 10 个格子（排产在最后）")
    cols = ["cell_id", "iters_per_epoch", "ms_per_iter", "est_hours", "cost_source"]
    print(df.sort_values("est_hours", ascending=False).head(10)[cols].to_string(index=False))

    print("\n[5] 最先跑的 10 个格子（升序排产，保证随时可出表）")
    print(df.head(10)[cols].to_string(index=False))

    if skipped:
        print(f"\n[6] 剪枝：{len(skipped)} 个格子被 skip_rules 剪掉")
        by_rule: dict[str, int] = {}
        for s in skipped:
            by_rule[s.rule_id] = by_rule.get(s.rule_id, 0) + 1
        for rid, n in by_rule.items():
            saved = sum(cost.estimate(s.cell).est_seconds for s in skipped if s.rule_id == rid) * safety
            print(f"  - {rid}: {n} 格, 省下 {saved/3600:.1f} GPU 小时")
            print(f"    理由: {next(s.reason for s in skipped if s.rule_id == rid)}")

    print("\n[7] 总预算")
    print(f"  待跑格子      : {int(b['n_todo'])} / {int(b['n_total'])}")
    print(f"  已完成 GPU 小时: {b['gpu_hours_done']:.1f}")
    print(f"  待跑 GPU 小时  : {b['gpu_hours_todo']:.1f}")
    print(f"  墙钟天数      : {b['days_24h']:.1f} 天 (24h/天) | "
          f"{b['days_20h']:.1f} 天 (20h/天) | {b['days_16h']:.1f} 天 (16h/天)")
    print("  提示：6–8 周档期里主矩阵应控制在 ≤21 天，剩余时间留给 selector/统计/写作。")

    if plan_out:
        plan_out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(plan_out, index=False)
        print(f"\n排产表已写出: {plan_out}")


# --------------------------------------------------------------------------- #
# 执行
# --------------------------------------------------------------------------- #
def probe_cells(cfg: MatrixConfig, include_optional: bool = False) -> list[Cell]:
    """测速用的代表性格子：每 (骨干×数据集) 取最短/最长 horizon × none 插件，
    再在 ETTh1 上把每个插件都测一遍（插件开销与数据集近似无关）。"""
    out: list[Cell] = []
    seed = cfg.seed_policy["base_seeds"][0]
    for ds in cfg.dataset_names(include_optional=include_optional):
        hs = cfg.dataset(ds)["horizons"]
        for bk in cfg.backbone_names:
            for h in (min(hs), max(hs)):
                out.append(Cell(bk, ds, int(h), cfg.control_plugin, seed))
    ref = "ETTh1" if "ETTh1" in cfg.dataset_names(True) else cfg.dataset_names(True)[0]
    for bk in cfg.backbone_names:
        for pl in cfg.plugin_names:
            if pl == cfg.control_plugin:
                continue
            out.append(Cell(bk, ref, int(min(cfg.dataset(ref)["horizons"])), pl, seed))
    return out


def _cmd(cell: Cell, cfg_path: Path | None, device: str, probe: bool, probe_steps: int,
         save_ckpt: bool, save_winerr: bool = False) -> list[str]:
    cmd = [
        sys.executable, "-m", "src.train_cell",
        "--backbone", cell.backbone, "--dataset", cell.dataset,
        "--pred-len", str(cell.pred_len), "--plugin", cell.plugin,
        "--seed", str(cell.seed), "--device", device,
    ]
    if cfg_path:
        cmd += ["--config", str(cfg_path)]
    if probe:
        cmd += ["--probe", "--max-epochs", "1", "--max-train-steps", str(probe_steps),
                "--max-eval-steps", "5"]
    if save_ckpt:
        cmd += ["--save-checkpoint"]
    if save_winerr:
        cmd += ["--save-window-mse"]
    return cmd


def parse_deadline(text: str | None) -> float | None:
    """把 'YYYY-MM-DD HH:MM' / 'YYYY-MM-DDTHH:MM'（本地时区）解析成 epoch 秒。

    无人值守场景下必须用**绝对时刻**而不是 ``--budget-hours`` 相对时长：
    watchdog 脚本会在崩溃后反复重启 scheduler，相对预算每次都从 0 重新计时，
    永远不会到点；绝对截止对重启是幂等的。
    """
    if not text:
        return None
    s = text.strip().replace("T", " ")
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m-%d %H:%M"):
        try:
            tm = time.strptime(s, fmt)
        except ValueError:
            continue
        if fmt == "%m-%d %H:%M":  # 省略年份时按当前年份补齐
            tm = time.strptime(f"{time.localtime().tm_year}-{s}", "%Y-%m-%d %H:%M")
        return time.mktime(tm)
    raise ValueError(f"无法解析 --deadline={text!r}，期望 'YYYY-MM-DD HH:MM'")


def run_queue(
    cfg: MatrixConfig,
    plans: list[CellPlan],
    device: str = "cuda",
    budget_hours: float | None = None,
    probe: bool = False,
    probe_steps: int = 40,
    save_ckpt: bool = False,
    save_winerr: bool = False,
    limit: int | None = None,
    workers: int = 1,
    deadline_ts: float | None = None,
) -> None:
    """按排产顺序执行（子进程隔离），支持预算上限、断点续跑与单卡多路并发。

    并发的依据（``tools/bench_concurrency.py`` 在 V100-32GB 上实测，
    TimesNet × Electricity 真实张量形状，bs=16，AMP）::

        N=1  106 ms/iter   9.40 iter/s  1.00x  276 MiB
        N=2  138 ms/iter  14.48 iter/s  1.54x  276 MiB
        N=3  155 ms/iter  19.36 iter/s  2.06x  280 MiB   <- 甜点
        N=4  207 ms/iter  19.29 iter/s  2.05x  276 MiB   <- 饱和

    全矩阵单格显存峰值仅 3.56 GB / 32 GB，因此瓶颈是 SM 争抢而不是显存；
    N=3 拿到 2.06x 吞吐后 N=4 不再有收益，故默认上限为 3。

    注意：并发只影响墙钟，不影响任何指标——每格是独立子进程、固定 seed，
    MSE/MAE 与串行完全一致；只有 ``train_seconds`` 会被拉长，
    因此**并发模式下不更新成本模型**（成本模型只信 ``--probe`` 的串行实测）。
    """
    t0 = time.time()
    todo = [p for p in plans if not p.done]
    if limit:
        todo = todo[:limit]
    workers = max(1, int(workers))
    dl_txt = ("无" if deadline_ts is None
              else time.strftime("%m-%d %H:%M", time.localtime(deadline_ts)))
    print(f"[sched] 待执行 {len(todo)} 格，device={device}，probe={probe}，并发={workers}，"
          f"截止={dl_txt}")

    def _budget_exhausted() -> bool:
        if budget_hours is not None and (time.time() - t0) / 3600.0 > budget_hours:
            return True
        return deadline_ts is not None and time.time() >= deadline_ts

    # 判断"这一格来不来得及"时统一乘上 safety_margin：成本模型是中位数估计，
    # 一半的格子会比它慢；宁可少投一格，也不要在截止前留一个跑不完的半成品。
    safety = float(cfg.cost_model.get("safety_margin", 1.2))

    def _remaining() -> float:
        """到硬截止还剩多少秒；没有截止则视为无穷。"""
        return float("inf") if deadline_ts is None else deadline_ts - time.time()

    def _fits(p: CellPlan) -> bool:
        return p.est_seconds * safety <= _remaining()

    def _take_fitting(pool: list[CellPlan]) -> CellPlan | None:
        """从队尾取下一格；有硬截止时跳过**注定跑不完**的格子（尾部装箱）。

        无人值守长跑的核心保护：临近截止时不再投放 4 小时的 TimesNet-h720，
        而是继续往后找能在剩余时间内跑完的便宜格子。被跳过的格子留在 pending 里，
        下次（人来了、或换更长的截止）续跑即可——断点续跑保证不重复计算。
        """
        if not pool:
            return None
        if deadline_ts is None:
            return pool.pop()
        for i in range(len(pool) - 1, -1, -1):
            if _fits(pool[i]):
                return pool.pop(i)
        return None

    if workers == 1:
        for i, p in enumerate(todo, 1):
            if _budget_exhausted():
                print(f"[sched] 达到预算/截止，优雅停止（剩余 {len(todo)-i+1} 格下次续跑）")
                break
            if not _fits(p):
                print(f"[sched] 跳过 {p.cell.cell_id}（预估 {p.est_seconds*safety/3600:.1f}h > "
                      f"剩余 {_remaining()/3600:.1f}h），留到下次")
                continue
            cmd = _cmd(p.cell, cfg.config_path, device, probe, probe_steps, save_ckpt, save_winerr)
            print(f"[sched] ({i}/{len(todo)}) {p.cell.cell_id}  预估 {p.est_seconds/60:.1f} min")
            rc = subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode
            if rc != 0:
                print(f"[sched] !! 失败 rc={rc}: {p.cell.cell_id}（已记录 .failed，继续下一格）")
    else:
        # 排产表按预估耗时**升序**，这里 reversed + pop() 等价于「最便宜的先进池」（SPT）。
        # 为什么不是经典的 LPT（最长优先、makespan 最优）：本项目的产出是一张矩阵，
        # 有硬截止时「完成的格子数」比「总墙钟」重要得多——先把 600 个便宜格子跑完，
        # 能立刻验证结论是否成立；反过来先投 4 小时的 TimesNet-h720，
        # 截止到了可能一个便宜格子都没跑，矩阵全是洞。断点续跑保证中断不丢已完成的格子。
        pending = list(reversed(todo))
        running: dict[subprocess.Popen, CellPlan] = {}
        done_n, total = 0, len(todo)
        stopped = False
        while pending or running:
            while pending and len(running) < workers and not stopped:
                if _budget_exhausted():
                    stopped = True
                    print(f"[sched] 达到预算/截止，停止投放新格子（等待在跑的 {len(running)} 格收尾）")
                    break
                p = _take_fitting(pending)
                if p is None:
                    stopped = True
                    print(f"[sched] 剩余 {_remaining()/3600:.1f}h 已装不下任何待跑格子"
                          f"（剩 {len(pending)} 格留到下次），等待在跑的 {len(running)} 格收尾")
                    break
                cmd = _cmd(p.cell, cfg.config_path, device, probe, probe_steps, save_ckpt, save_winerr)
                # 并发时子进程日志会交错，统一重定向到 logs/<cell_id>.log 便于事后定位
                log_dir = REPO_ROOT / "logs"
                log_dir.mkdir(exist_ok=True)
                fh = (log_dir / f"{p.cell.cell_id}.log").open("w", encoding="utf-8")
                proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=fh, stderr=subprocess.STDOUT)
                proc._pluggate_log = fh  # type: ignore[attr-defined]
                running[proc] = p
                print(f"[sched] +投放 {p.cell.cell_id}  预估 {p.est_seconds/60:.1f} min "
                      f"（池 {len(running)}/{workers}，剩余 {len(pending)}）", flush=True)
            if not running:
                break
            time.sleep(5)
            for proc in [q for q in running if q.poll() is not None]:
                p = running.pop(proc)
                log = getattr(proc, "_pluggate_log", None)
                if log is not None:
                    log.close()
                done_n += 1
                if proc.returncode != 0:
                    print(f"[sched] !! 失败 rc={proc.returncode}: {p.cell.cell_id} "
                          f"（详见 logs/{p.cell.cell_id}.log，继续）", flush=True)
                else:
                    print(f"[sched] -完成 ({done_n}/{total}) {p.cell.cell_id}", flush=True)

    if probe:
        write_speed_csv(cfg)
    else:
        aggregate(cfg)


def write_speed_csv(cfg: MatrixConfig) -> Path:
    """把**本阶段** probe 目录里的实测 ms/iter 增量 upsert 进（跨阶段共享的）speed.csv。

    2026-09-14 code review 修复项。原实现是「读阶段局部、写全局共享」的整表覆盖：
    读端是 ``cfg.path("cells_dir").parent / "probe"``（p2 → ``results/p2/probe``、
    p3 → ``results/p3/probe``），写端却是 ``cfg.path("speed_csv")``，而四个 yaml 的
    ``speed_csv`` **全部**指向同一个 ``results/speed.csv``。后果有两层：
    ① 用非 phase-1 的 config 跑一次 ``--probe``，phase-1 辛苦测出来的 ms/iter 就被这次
       阶段局部的结果整表顶掉；
    ② 这些阶段的 probe 目录通常是空的，``rows`` 为空时 ``pd.DataFrame([]).to_csv()``
       只产出 ``"\\n"``，speed.csv 变成无表头空文件，之后 ``load_speed`` 的 ``read_csv``
       抛 ``EmptyDataError``，把所有 config 的 ``--status``/``--dry-run``/``--run``
       连带 watchdog 的 status 轮询一起打挂。
    而 ``run_weekend.sh`` 的前提正是「必须先校准，否则 ``--deadline`` 的尾部装箱会在
    周日晚上投出跑不完的格子」——成本表被清空恰好破坏这个前提。

    因此这里改成：speed.csv 是**跨 config 共享的成本状态**，
    - 本次没有任何可用 probe 结果时**直接跳过写入**（保留已有实测值，绝不清表）；
    - 有结果时按 ``(backbone, dataset, pred_len, plugin)`` 主键做**增量 upsert**
      （本次实测覆盖同键旧值，其它阶段的行原样保留）。
    """
    rows = []
    d = cfg.path("cells_dir").parent / "probe"
    for p in sorted(d.glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        rows.append({k: r[k] for k in ("backbone", "dataset", "pred_len", "plugin", "ms_per_iter")}
                    | {"peak_mem_mib": r.get("peak_mem_mib"), "n_params": r.get("n_params")})
    out = cfg.path("speed_csv")
    new = pd.DataFrame(rows)
    if new.empty:
        # 关键分支：宁可什么都不写，也不能把跨阶段共享的成本表清成空文件。
        print(f"[sched] {d} 没有可用的 probe 结果，保留原有速度表 {out} 不动")
        return out
    old = read_speed_table(out)
    if not old.empty:
        # 主键 dtype 对齐：老表从 csv 读回来 pred_len 可能是 int64/float64，
        # 新行来自 probe json（int），不统一会让 drop_duplicates 认不出同一格。
        for df_ in (old, new):
            df_["pred_len"] = pd.to_numeric(df_["pred_len"], errors="coerce").astype("Int64")
            for c in ("backbone", "dataset", "plugin"):
                df_[c] = df_[c].astype(str)
        merged = (pd.concat([old, new], ignore_index=True)
                  .drop_duplicates(subset=SPEED_KEY, keep="last")     # keep="last" = 本次实测优先
                  .sort_values(SPEED_KEY, kind="mergesort")
                  .reset_index(drop=True))
        n_new = len(merged) - len(old)
    else:
        merged, n_new = new, len(new)
    atomic_write_text(out, merged.to_csv(index=False))
    print(f"[sched] 实测速度表 -> {out}  (本次 upsert {len(new)} 行，新增 {n_new} 行，"
          f"合计 {len(merged)} 行)")
    return out


def aggregate(cfg: MatrixConfig) -> Path:
    """汇总 results/cells/*.json → results/results.csv（原子替换）。

    只收「正式格子」：带 tag 的文件（如 `<cell_id>.smoke.json`）与 `.failed` 一律跳过，
    否则烟囱测试的 20-step 结果会悄悄混进主表。
    """
    rows: list[dict[str, Any]] = []
    for p in sorted(cfg.path("cells_dir").glob("*.json")):
        if "." in p.stem:  # <cell_id>.smoke / <cell_id>.json.failed
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        if r.get("status") != "ok":
            continue
        seg = r.pop("seg_mse", []) or []
        for i, v in enumerate(seg):
            r[f"seg{i+1}_mse"] = v
        r["tag"] = ""
        rows.append(r)
    df = pd.DataFrame(rows)
    out = cfg.path("results_csv")
    atomic_write_text(out, df.to_csv(index=False))
    print(f"[sched] 聚合结果 -> {out}  ({len(df)} 行)")
    return out


def status(cfg: MatrixConfig, include_optional: bool = False) -> None:
    plans, skipped, cost = make_plan(cfg, include_optional=include_optional)
    b = budget_summary(plans, cost.safety)
    failed = list((cfg.path("cells_dir")).glob("*.failed"))
    print(f"完成 {int(b['n_total']-b['n_todo'])}/{int(b['n_total'])} 格  "
          f"（剪枝 {len(skipped)} 格，失败 {len(failed)} 格）")
    print(f"剩余预估 {b['gpu_hours_todo']:.1f} GPU 小时 ≈ {b['days_20h']:.1f} 天(20h/天)")
    for f in failed[:10]:
        print("  失败:", f.name)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description="PlugGate 实验排产调度器")
    ap.add_argument("--config", default=None)
    ap.add_argument("--dry-run", action="store_true", help="只打印排产与预算，不跑（无 GPU 可用）")
    ap.add_argument("--probe", action="store_true", help="测速模式：少量 iter 实测 ms/iter")
    ap.add_argument("--probe-steps", type=int, default=40)
    ap.add_argument("--run", action="store_true", help="按排产顺序正式执行")
    ap.add_argument("--aggregate", action="store_true", help="仅汇总 CSV")
    ap.add_argument("--status", action="store_true", help="进度概览")
    ap.add_argument("--include-optional", action="store_true", help="把 optional 数据集(Traffic)纳入")
    ap.add_argument("--no-skip-rules", action="store_true", help="关闭剪枝规则，跑满矩阵")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--deadline", default=None,
                    help="硬截止墙钟（本地时区），格式 'YYYY-MM-DD HH:MM' 或 'YYYY-MM-DDTHH:MM'。"
                         "到点后不再投放新格子；且任何**预估耗时超过剩余时间**的格子会被跳过，"
                         "改投能跑完的便宜格子（尾部装箱）。无人值守跑周末必开。")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save-checkpoint", action="store_true")
    ap.add_argument("--save-window-mse", action="store_true",
                    help="每格额外存逐窗口测试 MSE（phase-2 逐窗口门控的数据基础）")
    ap.add_argument("--backbones", nargs="*", default=None)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--plugins", nargs="*", default=None)
    ap.add_argument("--horizons", nargs="*", type=int, default=None)
    ap.add_argument("--plan-out", default=None,
                    help="排产表输出路径；默认写到该 config 自己的 artifacts_dir/plan.csv，"
                         "避免 phase-1 / phase-2 两套配置互相覆盖同一个文件")
    ap.add_argument("--workers", type=int, default=1,
                    help="单卡并发格子数；实测 V100-32GB 上 N=3 拿到 2.06x 吞吐、N=4 饱和"
                         "（见 tools/bench_concurrency.py）。probe 模式强制串行以保证测速干净。")
    args = ap.parse_args()

    deadline_ts = parse_deadline(args.deadline)
    if deadline_ts is not None:
        print(f"[sched] 硬截止 {time.strftime('%F %H:%M', time.localtime(deadline_ts))}"
              f"（剩余 {(deadline_ts - time.time())/3600:.1f} 小时）")

    cfg = MatrixConfig.load(args.config)
    if args.aggregate:
        aggregate(cfg)
        return
    if args.status:
        status(cfg, args.include_optional)
        return

    filters: dict[str, Any] = {}
    for k in ("backbones", "datasets", "plugins", "horizons"):
        v = getattr(args, k)
        if v:
            filters[k] = v

    if args.probe:
        cells = probe_cells(cfg, args.include_optional)
        cost = CostModel(cfg)
        plans = [cost.estimate(c) for c in cells]
        plans.sort(key=lambda p: p.est_seconds)
        if args.dry_run:
            print(f"[probe dry-run] {len(plans)} 个测速格子，每格 {args.probe_steps} iter")
            print(_plan_frame(plans)[["cell_id", "iters_per_epoch", "ms_per_iter", "est_hours"]]
                  .to_string(index=False))
            return
        run_queue(cfg, plans, args.device, args.budget_hours, probe=True,
                  probe_steps=args.probe_steps, limit=args.limit)
        return

    plans, skipped, cost = make_plan(
        cfg, include_optional=args.include_optional, apply_skips=not args.no_skip_rules, **filters
    )
    if args.run:
        run_queue(cfg, plans, args.device, args.budget_hours,
                  save_ckpt=args.save_checkpoint, save_winerr=args.save_window_mse, limit=args.limit,
                  workers=args.workers, deadline_ts=deadline_ts)
    else:
        print_dry_run(cfg, plans, skipped, cost,
                      (REPO_ROOT / args.plan_out) if args.plan_out
                      else cfg.path("artifacts_dir") / "plan.csv")


if __name__ == "__main__":
    main()
