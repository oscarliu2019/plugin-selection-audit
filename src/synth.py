"""合成结果生成器：在**没有 GPU** 的环境里端到端验证 selector / stats / report 流水线。

它做两件事：
1. `make_results()`：以**真实计算出的复杂度特征**为自变量，按一组显式的
   「机理假设」合成每格的 MSE。这样 selector 面对的信号结构与真实实验同构
   （特征 → 增益符号），可以真实地验证 leave-one-dataset-out 是否有效；
2. `make_epoch_logs()`：合成每 epoch 的验证/测试曲线，用于验证 horizon-aware
   早停准则模块。

⚠️ 生成的 CSV 一律带 `source=synthetic` 列，并写在 `artifacts/synthetic/` 下，
   **绝不能混进 results/results.csv**。论文里不会出现任何合成数字。

机理假设（也就是本文要用真实实验去检验的科学假设）
- 基线误差随「窗口级模式复杂度」上升（Accuracy Law 的误差下界视角）；
- RevIN 只在**没有内置实例归一化的骨干**上有大增益，且随非平稳度/窗口漂移增大；
- SAN-lite 在「窗口内非平稳（方差漂移大）+ 长 horizon」时占优，平稳数据上有害；
- FreDF 在「频谱能量集中 / 季节强度高」时占优，接近白噪声时无效。
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import MatrixConfig


def _gains(row: pd.Series, backbone: str, internal_norm: str, rng: np.random.Generator) -> dict[str, float]:
    """返回各插件相对基线的**相对**增益（正 = 误差下降的比例）。"""
    drift = float(row.get("ns_mean_drift", 0.5))
    shift = float(row.get("hz_mean_shift", 0.3))
    vdrift = float(row.get("ns_var_drift", 0.3))
    topk = float(row.get("fr_topk_energy", 0.5))
    seas = float(row.get("sd_seasonal_strength", 0.5))
    spec = float(row.get("pc_spec_entropy", 0.5))
    h_ratio = float(row.get("hz_ratio_seq", 1.0))

    revin = (0.10 * drift + 0.15 * shift - 0.02) if internal_norm == "none" else rng.normal(0.0, 0.002)
    san = 0.06 * vdrift + 0.08 * shift * min(h_ratio, 4.0) / 4.0 - 0.02
    fredf = 0.07 * topk + 0.05 * seas - 0.06 * spec - 0.005
    if backbone == "TimesNet":  # 重模型自带周期建模，频域损失边际收益更小
        fredf *= 0.5
    # 统一的「挂插件的代价」：插件改变了优化景观，平均带来一点额外误差。
    # 有了它，弱信号格子上最优决策才会是「不挂」——这正是本文要检出的决策类型。
    penalty = 0.012
    return {"none": 0.0, "revin": revin - penalty, "san_lite": san - penalty,
            "fredf": fredf - penalty}


def make_results(
    features: pd.DataFrame,
    cfg: MatrixConfig | None = None,
    seeds: list[int] | None = None,
    noise: float = 0.004,
    seed: int = 26,
) -> pd.DataFrame:
    """按机理假设合成 (骨干 × 数据集 × horizon × 插件 × seed) 的结果长表。"""
    cfg = cfg or MatrixConfig.load()
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for _, row in features.iterrows():
        ds, h = str(row["dataset"]), int(row["pred_len"])
        pc = float(row.get("pc_perm_entropy", 0.9))
        # Accuracy Law 式的基线：误差随模式复杂度指数上升，随 horizon 增长
        base_ds = 0.18 * np.exp(1.6 * (pc - 0.8)) * (1 + 0.22 * np.log2(max(h, 1) / 96 + 1))
        for bk in cfg.backbone_names:
            inorm = str(cfg.backbone(bk).get("internal_norm", "none"))
            bk_factor = {"DLinear": 1.06, "PatchTST": 0.98,
                         "iTransformer": 0.97, "TimesNet": 1.02}.get(bk, 1.0)
            base = base_ds * bk_factor
            g = _gains(row, bk, inorm, rng)
            for pl in cfg.plugin_names:
                for sd in (seeds or cfg.seeds_for(ds)):
                    mse = base * (1.0 - g[pl]) * (1.0 + rng.normal(0.0, noise))
                    rows.append({
                        "backbone": bk, "dataset": ds, "pred_len": h, "plugin": pl, "seed": sd,
                        "mse": float(mse), "mae": float(mse * 0.78 + rng.normal(0, noise)),
                        "val_mse": float(mse * 1.05), "best_epoch": int(rng.integers(2, 9)),
                        "epochs_run": 10, "source": "synthetic",
                    })
    return pd.DataFrame(rows)


def make_epoch_logs(n_cells: int = 40, n_epochs: int = 10, n_segments: int = 4,
                    seed: int = 7) -> pd.DataFrame:
    """合成每 epoch 的验证/测试曲线（长表：一行 = 一个 cell 的一个 epoch）。

    刻意注入本文要研究的现象：**验证集整体 MSE 最优的 epoch 往往在长 horizon 段上并不最优**
    （短 horizon 段先收敛、长 horizon 段还在改善），因此单一验证 MSE 会选错 checkpoint。
    """
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        cell_id = f"synthCell{c:03d}"
        short_opt = rng.integers(2, 5)
        long_opt = short_opt + rng.integers(1, 4)
        for e in range(n_epochs):
            segs_val, segs_test = [], []
            for s in range(n_segments):
                opt = short_opt + (long_opt - short_opt) * s / max(n_segments - 1, 1)
                curve = 0.3 * (1 + 0.25 * s) + 0.02 * (e - opt) ** 2 / 4.0
                segs_val.append(float(curve * (1 + rng.normal(0, 0.01))))
                segs_test.append(float(curve * 1.02 * (1 + rng.normal(0, 0.01))))
            rows.append({
                "cell_id": cell_id, "epoch": int(e),
                "val_mse": float(np.mean(segs_val)), "test_mse": float(np.mean(segs_test)),
                **{f"val_seg{i+1}_mse": v for i, v in enumerate(segs_val)},
                **{f"test_seg{i+1}_mse": v for i, v in enumerate(segs_test)},
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="生成合成结果（仅用于无 GPU 环境验证流水线）")
    ap.add_argument("--features", default="artifacts/features_real_subset.csv")
    ap.add_argument("--out-dir", default="artifacts/synthetic")
    ap.add_argument("--seeds", nargs="*", type=int, default=[2021])
    args = ap.parse_args()

    cfg = MatrixConfig.load()
    feats = pd.read_csv(args.features)
    res = make_results(feats, cfg, seeds=args.seeds)
    logs = make_epoch_logs()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res.to_csv(out / "results_synthetic.csv", index=False)
    logs.to_csv(out / "epoch_logs_synthetic.csv", index=False)
    print(f"[synth] results {res.shape} -> {out/'results_synthetic.csv'}")
    print(f"[synth] epoch logs {logs.shape} -> {out/'epoch_logs_synthetic.csv'}")


if __name__ == "__main__":
    main()
