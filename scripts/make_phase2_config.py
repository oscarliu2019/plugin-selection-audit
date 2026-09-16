#!/usr/bin/env python
"""由 configs/matrix.yaml 派生出 phase-2 配置 configs/matrix_p2.yaml。

phase-2 要解决的问题：phase-1 的 selector 之所以失败，是因为 `artifacts/features.csv`
里只有 8 个互异的特征向量（每个数据集一份，4 个 horizon 之间完全相同），
LODO 之下等价于「拿 7 个训练样本预测第 8 个」。把决策粒度从「数据集」下沉到
「单个测试窗口」是唯一能把 N 抬到万级的办法，代价是每格都要留下逐窗口测试误差。

phase-1 的格子已经跑完但没存逐窗口误差，且当时没留 checkpoint，无法只重跑推理，
所以必须重训一遍。为了控制成本，phase-2 只跑 1 个 seed。

之所以派生而不是手抄一份 yaml：矩阵定义（骨干/数据集/horizon/插件/剪枝规则/成本模型）
必须与 phase-1 严格一致，否则两阶段结果不可比。这里只改三件事：

1. 产物路径全部挪到 results/p2/ 与 artifacts/p2/，不污染 phase-1 已完成的 800 格；
2. seed 策略退化为单 seed（cheap 层不再补到 3 seed）；
3. 登记一条 deviation，说明 phase-2 的定位与单 seed 的后果。

用法：python scripts/make_phase2_config.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs" / "matrix.yaml"
DST = ROOT / "configs" / "matrix_p2.yaml"

# phase-1 -> phase-2 的产物路径改写
# speed_csv 刻意不改写：phase-2 跑的是同一批格子、同一块 V100，phase-1 的
# results/speed.csv 里已有 76 个实测 ms/iter，直接复用可以让升序排产基于实测值而非先验，
# 省掉一次 probe，也保证两阶段的排产顺序一致。
PATH_OVERRIDES = {
    "cells_dir": "results/p2/cells",
    "results_csv": "results/p2/results.csv",
    "winerr_dir": "results/p2/winerr",
    "features_csv": "artifacts/p2/features.csv",
    "artifacts_dir": "artifacts/p2",
    "runs_dir": "runs_p2",
}


def build(src: Path = SRC) -> dict:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

    missing = [k for k in PATH_OVERRIDES if k not in cfg["paths"]]
    if missing:  # matrix.yaml 改过名就立刻炸，别让两阶段悄悄写到同一个目录
        raise KeyError(f"matrix.yaml 的 paths 缺少这些键，无法安全派生 phase-2: {missing}")
    cfg["paths"].update(PATH_OVERRIDES)

    cfg["seed_policy"]["tiers_with_extra_seeds"] = []
    cfg["seed_policy"]["note"] = (
        "phase-2 全矩阵仅 seed=2021。逐窗口门控的 N 来自测试窗口数（万级），"
        "不再依赖多 seed 撑样本量；但块级 MSE 的 seed 方差因此不可估计，"
        "论文里所有块级数字一律引用 phase-1 的 3-seed 结果。"
    )

    cfg.setdefault("deviations", {})["phase2_window_level"] = {
        "what": "重跑 1-seed 全网格，额外落盘 best epoch 的逐窗口测试 MSE",
        "why": (
            "phase-1 的特征只有 8 个互异取值（每数据集一份，horizon 间相同），"
            "LODO selector 准确率 0.381 低于多数类基线 0.405，permutation importance 全为 0；"
            "这是决策粒度问题而非模型问题"
        ),
        "how": (
            "scheduler --run --save-window-mse，逐格写 results/p2/winerr/<cell_id>.npy；"
            "TSLib 的 test loader shuffle=False，故不同格子的第 i 项严格对应同一个测试窗口，"
            "可按下标做配对比较"
        ),
        "cost": "428 格 / 81.0 GPU 小时；升序排产下前三个骨干约 5.6 GPU 小时即可先验证链路",
    }
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 matrix_p2.yaml")
    args = ap.parse_args()

    if DST.exists() and not args.force:
        print(f"[p2] {DST} 已存在，加 --force 覆盖")
        return 1

    cfg = build()
    header = (
        "# 本文件由 scripts/make_phase2_config.py 从 configs/matrix.yaml 自动派生，请勿手改。\n"
        "# 改矩阵定义请改 matrix.yaml 后重新生成，否则 phase-1/phase-2 结果不可比。\n"
    )
    DST.write_text(header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "utf-8")
    print(f"[p2] 已写出 {DST}")
    print(f"[p2] 产物路径: {PATH_OVERRIDES['cells_dir']} / {PATH_OVERRIDES['winerr_dir']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
