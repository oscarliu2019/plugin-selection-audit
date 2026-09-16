#!/usr/bin/env python
"""由 configs/matrix.yaml 派生 FreDF 公平性配置 configs/matrix_p3_fredf.yaml。

要回答的三个问题（internal/docs/fredf_tuning_evidence.md 的三个待办）
--------------------------------------------------------
phase-1 的结论「固定 α=0.5 的 FreDF 是 60 胜 55 负、平均 −3.01%」有一个致命的
可反驳点：**FreDF 官方脚本从来不用 α=0.5**，而是逐 (数据集, 骨干) 在 0.0–0.9 之间配置。
不做下面这组实验，审稿人会直接判定为 straw man。

1. **调参差距（tuning gap）**：把 α 扫成一格网格，看「每格用验证集选最优 α」
   能把 FreDF 从 −3% 挽回到多少。这比复现作者那几个具体 α 更强——它给出的是
   *调参的上界*，作者配置只是其中一个点。
2. **√H 归一化**：频域项相对 MSE 的量级按 √H 增长（实测 237.7/86.5 = 2.75 ≈ √7.5），
   所以固定 α 跨 horizon 在数学上就不成立。把频域项除以 √(H//2+1) 后再固定 α，
   看单一 α 能否跨 horizon 可用。能，就是一个小而干净的方法贡献。
3. **给门控器扩池**：这些臂同样落盘逐窗口误差，于是逐窗口门控的候选池从 4 个臂
   扩到 12 个（含不同 α），门控可学的空间更大，也更贴近真实使用场景
   （"挂不挂 FreDF" 其实是 "α 取多少"）。

为什么只跑 3 个骨干
------------------
TimesNet 一格 20–80 分钟，而这 8 个臂 × 8 数据集 × 4 horizon 已经是 768 格；
非 TimesNet 单格实测中位 0.3–7.7 分钟，整组约 16 进程小时（3 并发 ≈ 5 小时），
加上 TimesNet 会变成 150+ 小时。FreDF 是**损失函数**插件，其 α–H 量纲问题
与骨干无关，3 个骨干足以支撑结论，且 phase-1 已有 TimesNet × α=0.5 的对照。

用法：python scripts/make_fredf_config.py [--force]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "configs" / "matrix.yaml"
DST = ROOT / "configs" / "matrix_p3_fredf.yaml"

PATH_OVERRIDES = {
    "cells_dir": "results/p3/cells",
    "results_csv": "results/p3/results.csv",
    "winerr_dir": "results/p3/winerr",
    "features_csv": "artifacts/p3/features.csv",
    "artifacts_dir": "artifacts/p3",
    "runs_dir": "runs_p3",
}

KEEP_BACKBONES = ["DLinear", "PatchTST", "iTransformer"]

# α 网格。plain 取 {0.1,0.3,0.7,0.9}（0.5 已在 phase-1/2 跑过，不重复烧算力），
# √H 归一化取 {0.1,0.5,0.9}——与 plain 在 0.1 / 0.9 上重叠，构成配对对照。
PLAIN_ALPHAS = [0.1, 0.3, 0.7, 0.9]
SQRTH_ALPHAS = [0.1, 0.5, 0.9]


def _arm(name: str, impl: str, alpha: float, desc: str) -> dict:
    return {
        "name": name,
        "impl": impl,                     # 多个标签共用一个实现，见 config.plugin_impl()
        "kind": "loss",
        "cost_overhead": 1.05,
        "params": {"alpha": alpha},
        "desc": desc,
    }


def build(src: Path = SRC) -> dict:
    cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
    missing = [k for k in PATH_OVERRIDES if k not in cfg["paths"]]
    if missing:
        raise KeyError(f"matrix.yaml 的 paths 缺少 {missing}，无法安全派生 FreDF 配置")
    cfg["paths"].update(PATH_OVERRIDES)

    cfg["backbones"] = [b for b in cfg["backbones"] if b["name"] in KEEP_BACKBONES]
    assert len(cfg["backbones"]) == len(KEEP_BACKBONES), "骨干名对不上，检查 matrix.yaml"

    none_arm = next(p for p in cfg["plugins"] if p["name"] == "none")
    plugins = [none_arm]      # 同一份 none 在 p2/p3 各跑一次，顺便当确定性自检（应完全相同）
    for a in PLAIN_ALPHAS:
        tag = f"fredf_a{int(round(a*10)):02d}"
        plugins.append(_arm(tag, "fredf", a, f"FreDF 原始尺度，固定 α={a}（α 扫描臂）"))
    for a in SQRTH_ALPHAS:
        tag = f"fredf_sqrth_a{int(round(a*10)):02d}"
        plugins.append(_arm(tag, "fredf_sqrth", a,
                            f"FreDF 频域项除以 √(H//2+1) 后固定 α={a}（尺度归一化臂）"))
    cfg["plugins"] = plugins

    # revin 相关剪枝规则在这里无意义（本组没有 revin 臂），留着也不会命中，直接清掉更干净
    cfg["skip_rules"] = []
    cfg["seed_policy"]["base_seeds"] = [2021]
    cfg["seed_policy"]["extra_seeds"] = []
    cfg["seed_policy"]["tiers_with_extra_seeds"] = []
    cfg["seed_policy"]["note"] = "FreDF 公平性组只跑 seed=2021；结论靠 α 网格与 horizon 配对，不靠多 seed。"

    cfg.setdefault("deviations", {})["fredf_fairness"] = {
        "what": f"α 扫描（plain {PLAIN_ALPHAS} / √H 归一化 {SQRTH_ALPHAS}）× 3 个骨干 × 8 数据集 × 4 horizon",
        "why": (
            "FreDF 官方脚本逐 (数据集,骨干) 配置 α ∈ [0.0, 0.9]，其中 ECL×PatchTST 配成 0（等于关闭）。"
            "若只报告固定 α=0.5 的结果，会被判定为 straw man。本组给出『调参能挽回多少』的上界，"
            "并检验 √H 归一化能否让单一 α 跨 horizon 可用。"
        ),
        "how": "同 phase-2 协议（1 seed，落盘逐窗口 val/test 误差），产物写 results/p3/",
        "note": "不含 TimesNet：FreDF 是损失插件，α–H 量纲问题与骨干无关；TimesNet×α=0.5 已在 phase-1 有对照。",
    }
    return cfg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if DST.exists() and not args.force:
        print(f"[p3] {DST} 已存在，加 --force 覆盖")
        return 1
    cfg = build()
    header = (
        "# 本文件由 scripts/make_fredf_config.py 从 configs/matrix.yaml 自动派生，请勿手改。\n"
        "# 用途：FreDF 公平性组（α 扫描 + √H 归一化），产物在 results/p3/ 与 artifacts/p3/。\n"
    )
    DST.write_text(header + yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), "utf-8")
    n_arms = len(cfg["plugins"])
    print(f"[p3] 已写出 {DST}")
    print(f"[p3] {n_arms} 个臂 × {len(cfg['backbones'])} 骨干："
          f"{', '.join(p['name'] for p in cfg['plugins'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
