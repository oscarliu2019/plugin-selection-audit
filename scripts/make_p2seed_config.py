#!/usr/bin/env python
"""由 configs/matrix_p2.yaml 派生出「seed 复制组」配置 configs/matrix_p2seed.yaml。

要回答的问题
------------
phase-2 的 oracle 审计给出 per-window oracle 相对 best_fixed 有 ~12% 的 headroom，
但我们已经证明这个统计量对「逐窗口重贴臂标签」完全不变，因此它不能作为
「插件选择可学」的证据。审计的第二条腿是：**这 12% 里有多少是可复现的信号，
多少只是训练随机性（seed）在单个窗口上的噪声？**

phase-2 全矩阵只有 seed=2021，无法回答这个问题；phase-1 有 3 个 seed 但没有落盘
逐窗口误差（这正是 phase-2 存在的理由）。所以必须为**已经跑完的一批格子**补一个
seed，构成成对样本，然后做：

* 逐窗口 argmin 臂身份的跨 seed 一致率（vs 随机 1/K）；
* **跨 seed oracle**：用 seed A 的逐窗口 argmin 去选臂、在 seed B 的误差上结算。
  它仍然用到了「未来」，但只保留跨 seed 可复现的那部分臂身份信息，
  因此是任何依赖可复现结构的选择器的**上界**。

若同 seed oracle ≈ 12% 而跨 seed oracle ≈ 0，则 headroom 主要是噪声，
oracle 是在过拟合单次训练的随机性——这比「相关长度太短」更根本。

为什么只补一小块
----------------
主线 run_weekend.sh 正在占满 GPU，这里刻意只挑**已经有 seed=2021 完整 4 臂**、
且单格成本最低的格子（DLinear 全便宜数据集 + ETTh1 上的 PatchTST/iTransformer），
总计 80 格、约 2 GPU 小时，串行 1 worker 跑，对主线吞吐影响可忽略。

配对契约
--------
除 seed 外，矩阵定义（骨干/数据集/horizon/插件/剪枝/协议/成本模型）与 phase-2
严格一致——因为本文件是从 matrix_p2.yaml 派生的，只改 seed 与产物路径。
只有这样，两个 seed 的第 i 个窗口才严格对应同一段时间。

用法：python scripts/make_p2seed_config.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs" / "matrix_p2.yaml"
DST = ROOT / "configs" / "matrix_p2seed.yaml"

# 复制 seed。刻意不用 seed_policy.extra_seeds（那会按 tier 铺开成 192 格），
# 而是把 base_seeds 换掉 + 用 CLI 的 --backbones/--datasets/--horizons 精确框定。
REPLICATE_SEED = 2022

# 产物路径全部隔离：主线 run_weekend.sh 正在并发写 results/p2/results.csv，
# 共用会有 append 竞争。winerr 分开放，分析时把两个目录一起传给
# selector_window --winerr（load_winerr 支持多目录取并集，且 group key 含 seed，
# 所以 s2021 与 s2022 天然是不同的组，不会互相覆盖）。
PATH_OVERRIDES = {
    "cells_dir": "results/p2seed/cells",
    "results_csv": "results/p2seed/results.csv",
    "winerr_dir": "results/p2seed/winerr",
    "features_csv": "artifacts/p2seed/features.csv",
    "artifacts_dir": "artifacts/p2seed",
    "runs_dir": "runs_p2seed",
}


def build(src: Path = SRC) -> dict:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))

    missing = [k for k in PATH_OVERRIDES if k not in cfg["paths"]]
    if missing:  # p2 改过键名就立刻炸，别让两个 seed 悄悄写到同一个目录
        raise KeyError(f"matrix_p2.yaml 的 paths 缺少这些键，无法安全派生 seed 复制组: {missing}")
    cfg["paths"].update(PATH_OVERRIDES)

    cfg["seed_policy"] = {
        "base_seeds": [REPLICATE_SEED],
        "extra_seeds": [],
        "tiers_with_extra_seeds": [],
        "note": (
            f"seed 复制组只跑 seed={REPLICATE_SEED}，与 phase-2 的 seed=2021 构成成对样本。"
            "用途仅限「逐窗口 argmin 臂身份有多少是训练噪声」这一个问题，"
            "不参与任何块级 MSE 汇报（块级数字一律引用 phase-1 的 3-seed 结果）。"
        ),
    }

    cfg.setdefault("deviations", {})["p2seed_replicate"] = {
        "what": f"对 phase-2 已完成的 80 个便宜格子补跑 seed={REPLICATE_SEED}，落盘逐窗口误差",
        "why": (
            "per-window oracle 的 headroom 对逐窗口重贴臂标签不变，故不能证明可学性；"
            "需要区分 headroom 里「跨 seed 可复现的信号」与「单次训练的随机噪声」，"
            "而 phase-2 单 seed、phase-1 无逐窗口误差，都无法回答"
        ),
        "how": (
            "scheduler --run --save-window-mse，配置除 seed 与产物路径外与 matrix_p2.yaml 完全一致；"
            "分析用 selector_window --seed-audit，把 results/p2/winerr 与 results/p2seed/winerr 一起传入，"
            "按 (backbone,dataset,pred_len) 配对、逐窗口下标对齐"
        ),
        "cost": "80 格 / 约 2 GPU 小时（1 worker，与主线并发，nice 降优先级）",
        "scope": (
            "只覆盖 DLinear×{ETTh1,ETTh2,Exchange,ILI} 全 horizon 与 ETTh1 上的 "
            "PatchTST/iTransformer×{h96,h720}——即 seed=2021 已有完整 4 臂的那批格子"
        ),
    }
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="覆盖已存在的 matrix_p2seed.yaml")
    args = ap.parse_args()

    if DST.exists() and not args.force:
        print(f"[p2seed] {DST} 已存在，加 --force 覆盖")
        return 1

    cfg = build()
    header = (
        "# 本文件由 scripts/make_p2seed_config.py 从 configs/matrix_p2.yaml 自动派生，请勿手改。\n"
        f"# 唯一差异：seed={REPLICATE_SEED} + 产物路径隔离到 results/p2seed/。\n"
        "# 它的存在只为回答「逐窗口 argmin 臂身份有多少是训练噪声」。\n"
    )
    DST.write_text(header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "utf-8")
    print(f"[p2seed] 已写出 {DST}")
    print(f"[p2seed] seed={REPLICATE_SEED}，产物路径: {PATH_OVERRIDES['winerr_dir']}")
    print("[p2seed] 跑法（两条，串行）：")
    print("  python -m src.scheduler --config configs/matrix_p2seed.yaml --run --device cuda \\")
    print("      --workers 1 --save-window-mse --backbones DLinear \\")
    print("      --datasets ETTh1 ETTh2 Exchange ILI")
    print("  python -m src.scheduler --config configs/matrix_p2seed.yaml --run --device cuda \\")
    print("      --workers 1 --save-window-mse --backbones PatchTST iTransformer \\")
    print("      --datasets ETTh1 --horizons 96 720")
    return 0


if __name__ == "__main__":
    sys.exit(main())
