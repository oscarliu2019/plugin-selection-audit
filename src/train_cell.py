"""单格实验的命令行入口：由 scheduler 以子进程方式调用，也可手工单跑。

    # CPU 烟囱测试（本机验证插件/数据管线是否通）
    python -m src.train_cell --backbone DLinear --dataset ETTh1 --pred-len 96 \
        --plugin revin --seed 2021 --device cpu --max-epochs 1 --max-train-steps 20 \
        --max-eval-steps 10 --tag smoke

    # V100 正式跑（默认）
    python -m src.train_cell --backbone PatchTST --dataset ETTh1 --pred-len 96 --plugin fredf

子进程隔离的好处：单格 OOM / 崩溃不会带崩整条排产链，显存也一定被释放。
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

from .config import Cell, MatrixConfig, atomic_write_json
from .runner import RunConfig, run_cell


def main() -> int:
    ap = argparse.ArgumentParser(description="跑一个矩阵格子")
    ap.add_argument("--config", default=None)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pred-len", type=int, required=True)
    ap.add_argument("--plugin", required=True)
    ap.add_argument("--seed", type=int, default=2021)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--max-train-steps", type=int, default=None)
    ap.add_argument("--max-eval-steps", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--save-checkpoint", action="store_true")
    ap.add_argument("--save-window-mse", action="store_true",
                    help="存 best epoch 的逐窗口测试 MSE 到 results/winerr/<cell_id>.npy")
    ap.add_argument("--probe", action="store_true", help="仅测速，结果写到 probe 目录不进主表")
    ap.add_argument("--tag", default=None, help="结果文件后缀（如 smoke），避免污染正式结果")
    args = ap.parse_args()

    cfg = MatrixConfig.load(args.config)
    cell = Cell(args.backbone, args.dataset, args.pred_len, args.plugin, args.seed)
    run = RunConfig(
        device=args.device,
        max_epochs=args.max_epochs,
        max_train_steps=args.max_train_steps,
        max_eval_steps=args.max_eval_steps,
        num_workers=args.num_workers,
        save_checkpoint=args.save_checkpoint,
        save_window_mse=args.save_window_mse,
        probe=args.probe,
    )

    suffix = f".{args.tag}" if args.tag else ""
    if args.probe:
        out = cfg.path("cells_dir").parent / "probe" / f"{cell.cell_id}{suffix}.json"
    else:
        out = cfg.path("cells_dir") / f"{cell.cell_id}{suffix}.json"

    try:
        result = run_cell(cfg, cell, run)
    except Exception as exc:  # 失败也落盘，方便调度器区分「没跑」与「跑失败」
        err = {
            **cell.to_dict(),
            "cell_id": cell.cell_id,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
        atomic_write_json(Path(str(out) + ".failed"), err)
        print(f"[train_cell] FAILED {cell.cell_id}: {err['error']}", file=sys.stderr)
        return 1

    result["status"] = "ok"
    atomic_write_json(out, result)
    print(f"[train_cell] OK {cell.cell_id} mse={result['mse']:.5f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
