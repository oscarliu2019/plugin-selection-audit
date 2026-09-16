"""单格实验执行器：零侵入地驱动 TSLib 骨干 + PlugGate 插件。

为什么不直接用 TSLib 的 `run.py` / `Exp_Long_Term_Forecast`
--------------------------------------------------------
1. `run.py` 的 argparse 不认识 `--plugin` / `--backbone`，加参数就得改它的源码；
2. `Exp_Basic._scan_models_directory()` 依赖 **当前工作目录**是 tslib 根目录，
   在调度器里 chdir 是隐式副作用，容易踩坑；
3. TSLib 的 `test()` 会把**全部** preds/trues 堆进内存再算指标：
   Traffic(862 通道) × pred_len=720 × 数千个测试窗口 ≈ 数 GB，V100 机器上极易 OOM。

所以本模块自己写训练/评估循环，但**只从 TSLib import**（data_provider / models / utils），
一行都不改它的源码：
- `models.<Backbone>.Model(configs)` —— 骨干（TSLib 统一签名）
- `data_provider.data_factory.data_provider(configs, flag)` —— 数据与切分（drop_last=False）
- `utils.tools.adjust_learning_rate` —— 与官方一致的学习率调度

同时修掉尽调 §1.5 提到的 CPU 瓶颈：损失累计全部用 tensor/标量原地累加，
指标计算改为**流式**（只维护平方误差和/绝对误差和 + 逐时间步误差和），内存 O(pred_len)。
"""

from __future__ import annotations

import importlib
import json
import os
import random
import sys
import time
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Cell, MatrixConfig, atomic_write_json
from .fft_compat import apply_fft_fp32_patch
from .plugins import kind_of, make_criterion, wrap

# fp16 autocast 下 cuFFT 只支持 2 的幂长度，TimesNet/FredF 会炸；见 fft_compat 模块注释
apply_fft_fp32_patch()


# --------------------------------------------------------------------------- #
# 运行配置
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    device: str = "cuda"                 # cuda | cpu
    max_epochs: int | None = None         # 覆盖 protocol.train_epochs（测速时设 1）
    max_train_steps: int | None = None    # 每 epoch 最多多少 iter（烟囱测试/测速用）
    max_eval_steps: int | None = None
    save_checkpoint: bool = False         # 只有做 horizon-aware 准则消融时才需要存
    save_window_mse: bool = False         # 存 best epoch 的逐窗口测试 MSE（phase-2 逐窗口门控用）
    num_workers: int | None = None
    probe: bool = False                   # 只测速：1 epoch + 少量 iter，不写正式结果
    seed_torch_deterministic: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# TSLib 接入
# --------------------------------------------------------------------------- #
def ensure_tslib(cfg: MatrixConfig) -> Path:
    """把 tslib 根目录挂到 sys.path（不 chdir、不改其源码）。"""
    root = cfg.path("tslib_root")
    if not (root / "models").is_dir():
        raise FileNotFoundError(
            f"未找到 TSLib：{root}\n"
            "请先执行：git clone --depth 1 https://github.com/thuml/Time-Series-Library "
            f"{root}"
        )
    p = str(root)
    if p not in sys.path:
        sys.path.insert(0, p)
    return root


def build_configs(cfg: MatrixConfig, cell: Cell, run: RunConfig) -> Namespace:
    """构造 TSLib 骨干 / data_provider 需要的 configs 命名空间（等价于 run.py 的 args）。"""
    from .config import REPO_ROOT

    ds = cfg.dataset(cell.dataset)
    bk = cfg.backbone(cell.backbone)
    proto = cfg.protocol
    hp = dict(bk.get("hparams") or {})

    ns = Namespace(
        # 任务
        task_name=proto["task_name"],
        is_training=1,
        model_id=cell.cell_id,
        model=cell.backbone,
        # 数据
        data=ds["tslib_data"],
        root_path=str((REPO_ROOT / ds["root_path"]).resolve()) + os.sep,
        data_path=ds["data_path"],
        features=proto["features"],
        target="OT",
        freq=ds["freq"],
        checkpoints=str(cfg.path("runs_dir") / cell.cell_id) + os.sep,
        seq_len=cfg.seq_len(cell.dataset),
        label_len=cfg.label_len(cell.dataset),
        pred_len=int(cell.pred_len),
        seasonal_patterns="Monthly",
        inverse=bool(proto.get("inverse", False)),
        mask_rate=0.25,
        anomaly_ratio=0.25,
        # 模型
        expand=2, d_conv=4, tv_dt=0, tv_B=0, tv_C=0, use_D=0,
        top_k=int(hp.get("top_k", 5)), num_kernels=6,
        enc_in=int(ds["enc_in"]), dec_in=int(ds["enc_in"]), c_out=int(ds["enc_in"]),
        d_model=int(hp.get("d_model", 512)), n_heads=int(hp.get("n_heads", 8)),
        e_layers=int(hp.get("e_layers", 2)), d_layers=int(hp.get("d_layers", 1)),
        d_ff=int(hp.get("d_ff", 2048)), moving_avg=int(hp.get("moving_avg", 25)),
        factor=int(hp.get("factor", 3)), distil=True,
        dropout=float(hp.get("dropout", 0.1)), embed="timeF", activation="gelu",
        channel_independence=int(hp.get("channel_independence", 1)),
        decomp_method="moving_avg", use_norm=int(hp.get("use_norm", 1)),
        down_sampling_layers=0, down_sampling_window=1, down_sampling_method=None,
        seg_len=96, patch_len=int(hp.get("patch_len", 16)),
        node_dim=10, gcn_depth=2, gcn_dropout=0.3, propalpha=0.3,
        conv_channel=32, skip_channel=32, individual=False,
        alpha=0.1, top_p=0.5, pos=1,
        num_class=0,
        # 优化
        num_workers=int(run.num_workers if run.num_workers is not None else proto["num_workers"]),
        itr=1,
        train_epochs=int(run.max_epochs or proto["train_epochs"]),
        batch_size=int(ds["batch_size"]),
        patience=int(proto["patience"]),
        learning_rate=float(ds.get("learning_rate", proto["learning_rate"])),
        des=cfg.meta["project"],
        loss=proto["loss"],
        lradj=proto["lradj"],
        use_amp=bool(ds.get("use_amp", False)) and run.device == "cuda",
        # 设备
        use_gpu=run.device == "cuda",
        gpu=0, gpu_type="cuda", use_multi_gpu=False, devices="0",
        p_hidden_dims=[128, 128], p_hidden_layers=2,
        use_dtw=False, augmentation_ratio=0, seed=int(cell.seed),
    )
    return ns


def set_seed(seed: int, deterministic: bool = True) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def build_model(cfg: MatrixConfig, cell: Cell, configs: Namespace):
    """实例化骨干并套上插件。骨干通过 importlib 直接取 `models/<name>.py`。"""
    ensure_tslib(cfg)
    mod = importlib.import_module(f"models.{cell.backbone}")
    backbone = mod.Model(configs).float()
    params = dict(cfg.plugin(cell.plugin).get("params") or {})
    return wrap(
        backbone,
        cfg.plugin_impl(cell.plugin),
        n_channels=configs.enc_in,
        seq_len=configs.seq_len,
        pred_len=configs.pred_len,
        **params,
    )


# --------------------------------------------------------------------------- #
# 流式指标
# --------------------------------------------------------------------------- #
class StreamingMetrics:
    """流式累计 MSE / MAE 与逐时间步 MSE（内存 O(pred_len)，不堆 preds/trues）。"""

    def __init__(self, pred_len: int, n_segments: int = 4, per_window: bool = False) -> None:
        self.pred_len = pred_len
        self.n_segments = n_segments
        self.se = 0.0
        self.ae = 0.0
        self.count = 0
        self.ts_se = np.zeros(pred_len, dtype=np.float64)
        self.ts_count = np.zeros(pred_len, dtype=np.float64)
        # 逐窗口 MSE：selector 的决策粒度从「数据集」下沉到「单个测试窗口」所必需。
        # test_loader 在 TSLib 里 shuffle=False，所以不同格子的第 i 项严格对齐同一个窗口，
        # 可以直接按下标做配对比较。存储成本 = n_windows × 4B（ETTh1-96 约 11 KB）。
        self.per_window = bool(per_window)
        self._win: list[np.ndarray] = []

    def update(self, pred, true) -> None:
        import torch

        with torch.no_grad():
            d = (pred - true).float()
            self.se += float((d**2).sum())
            self.ae += float(d.abs().sum())
            self.count += int(d.numel())
            per_t = (d**2).sum(dim=(0, 2)).detach().cpu().numpy()
            n_t = d.shape[0] * d.shape[2]
            self.ts_se[: per_t.size] += per_t
            self.ts_count[: per_t.size] += n_t
            if self.per_window:
                self._win.append((d**2).mean(dim=(1, 2)).detach().cpu().numpy().astype(np.float32))

    def window_mse(self) -> np.ndarray:
        """按 loader 顺序排列的逐窗口 MSE；未开启 per_window 时返回空数组。"""
        if not self._win:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(self._win).astype(np.float32)

    def result(self) -> dict[str, Any]:
        if self.count == 0:
            return {"mse": float("nan"), "mae": float("nan"), "seg_mse": []}
        ts = np.divide(self.ts_se, np.maximum(self.ts_count, 1e-12))
        segs = np.array_split(ts[self.ts_count > 0], self.n_segments)
        return {
            "mse": self.se / self.count,
            "mae": self.ae / self.count,
            "rmse": float(np.sqrt(self.se / self.count)),
            "seg_mse": [float(np.mean(s)) for s in segs if s.size],
            "window_mse": self.window_mse(),
        }


def save_window_mse_atomic(win_dir: Path, cell_id: str, win: np.ndarray) -> Path:
    """把逐窗口 MSE 原子写到 ``win_dir/<cell_id>.npy``，返回最终路径。

    坑：``np.save`` 会给不以 ``.npy`` 结尾的路径**自动补** ``.npy``，所以不能直接
    ``np.save(tmp)`` 再 rename——临时名必须交给已打开的文件对象，否则 rename 会
    因为找不到源文件而 FileNotFoundError（2026-09-10 实测踩到）。
    """
    win_dir.mkdir(parents=True, exist_ok=True)
    final = win_dir / f"{cell_id}.npy"
    tmp = win_dir / f".{cell_id}.npy.tmp"
    with open(tmp, "wb") as fh:  # 传文件对象，np.save 就不会再改名
        np.save(fh, np.asarray(win, dtype=np.float32))
    tmp.replace(final)  # 同目录 rename，长跑中断不会留半截文件
    return final


ORDER_MARKER_SUFFIX = ".order.json"


def mark_deterministic_order(win_dir: Path, cell_id: str, n: int) -> Path:
    """给逐窗口误差数组打上「loader 顺序确定」的溯源标记。

    2026-09-14 的 code review 发现：TSLib 的 ``data_factory`` 只对 test 关 shuffle，
    因此此前落盘的 ``winerr/val/*.npy`` 是**随机置换**顺序，与按 ``window_index``
    排序的窗口特征按下标对齐时整体错位。修复后 val 改用确定序 loader，但磁盘上
    仍可能残留旧的污染数组，且二者无法从文件内容上区分。
    所以由写入方显式打标，读取方（``selector_window.load_winerr``）只接受带标记的
    val 数组——避免污染数据在下一轮分析里被静默复用。
    """
    win_dir.mkdir(parents=True, exist_ok=True)
    p = win_dir / f"{cell_id}{ORDER_MARKER_SUFFIX}"
    atomic_write_json(p, {"loader_order": "deterministic", "n_windows": int(n)})
    return p


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run_cell(cfg: MatrixConfig, cell: Cell, run: RunConfig) -> dict[str, Any]:
    """训练 + 每 epoch 验证/测试；返回该格的结果字典（同时写 runs/<cell_id>/epochs.jsonl）。"""
    import torch
    import torch.nn as nn

    ensure_tslib(cfg)
    from data_provider.data_factory import data_provider  # type: ignore
    from utils.tools import adjust_learning_rate  # type: ignore

    configs = build_configs(cfg, cell, run)
    set_seed(cell.seed, run.seed_torch_deterministic)
    device = torch.device("cuda:0" if (run.device == "cuda" and torch.cuda.is_available()) else "cpu")

    run_dir = cfg.path("runs_dir") / cell.cell_id
    run_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(run_dir / "configs.json", vars(configs))
    epoch_log = run_dir / "epochs.jsonl"
    if epoch_log.exists():
        epoch_log.unlink()

    train_set, train_loader = data_provider(configs, "train")
    val_set, val_loader = data_provider(configs, "val")
    _, test_loader = data_provider(configs, "test")

    # ⚠️ TSLib 的 data_factory 只对 test 关 shuffle：
    #     shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    # 所以 data_provider(configs, "val") 返回的是 shuffle=True 的 loader。逐窗口误差是按
    # loader 迭代顺序累积的（StreamingMetrics._win），一旦 loader 打乱，落盘的
    # winerr/val/<cell_id>.npy 第 i 项就不再对应第 i 个验证窗口，而下游
    # selector_window.build_groups 是严格按下标把窗口特征与逐窗口误差对齐的
    # （且每个臂的置换还各不相同，跨臂 argmin 也会错位）。
    # 因此逐窗口落盘必须用一个确定序的 val loader；聚合 val["mse"] 与 early stopping
    # 不受顺序影响，仍沿用原 loader 即可。
    val_eval_loader = val_loader
    if run.save_window_mse:
        val_eval_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size=configs.batch_size,
            shuffle=False,
            num_workers=configs.num_workers,
            drop_last=False,
        )

    model = build_model(cfg, cell, configs).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    extra_params = model.extra_param_count()
    optim = torch.optim.Adam(model.parameters(), lr=configs.learning_rate)
    criterion = make_criterion(
        cfg.plugin_impl(cell.plugin),
        default=nn.MSELoss(),
        **dict(cfg.plugin(cell.plugin).get("params") or {}),
    )
    scaler = torch.amp.GradScaler("cuda") if configs.use_amp else None

    best_val = float("inf")
    best_epoch = -1
    best_test: dict[str, Any] = {}
    best_val_full: dict[str, Any] = {}
    bad_epochs = 0
    iter_times: list[float] = []
    t0 = time.time()

    for epoch in range(configs.train_epochs):
        model.train()
        loss_sum, n_batches = 0.0, 0
        ep_start = time.time()
        for i, (bx, by, bxm, bym) in enumerate(train_loader):
            if run.max_train_steps is not None and i >= run.max_train_steps:
                break
            it0 = time.time()
            optim.zero_grad(set_to_none=True)
            loss = _forward_loss(model, criterion, configs, bx, by, bxm, bym, device)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()
            loss_sum += float(loss.detach())  # 标量累加，避免 issue #324 的 numpy 列表瓶颈
            n_batches += 1
            if i >= 3:  # 跳过预热 iter 再计时
                iter_times.append(time.time() - it0)

        val = _evaluate(
            model, criterion, configs, val_eval_loader, device, run.max_eval_steps,
            per_window=run.save_window_mse,
        )
        test = _evaluate(
            model, criterion, configs, test_loader, device, run.max_eval_steps,
            per_window=run.save_window_mse,
        )
        rec = {
            "epoch": epoch,
            "train_loss": loss_sum / max(n_batches, 1),
            "lr": optim.param_groups[0]["lr"],
            "epoch_seconds": round(time.time() - ep_start, 2),
            "val_mse": val["mse"], "val_mae": val["mae"], "val_seg_mse": val["seg_mse"],
            "test_mse": test["mse"], "test_mae": test["mae"], "test_seg_mse": test["seg_mse"],
        }
        with open(epoch_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
        print(f"[run] {cell.cell_id} ep{epoch} train={rec['train_loss']:.5f} "
              f"val={val['mse']:.5f} test={test['mse']:.5f} ({rec['epoch_seconds']}s)", flush=True)

        if val["mse"] < best_val - 1e-8:
            best_val, best_epoch, best_test, bad_epochs = val["mse"], epoch, test, 0
            best_val_full = val
            if run.save_checkpoint:
                torch.save(model.state_dict(), run_dir / "checkpoint.pth")
        else:
            bad_epochs += 1
            if bad_epochs >= configs.patience:
                print(f"[run] early stop at epoch {epoch}", flush=True)
                break
        adjust_learning_rate(optim, epoch + 1, configs)

    ms_per_iter = float(np.median(iter_times) * 1000) if iter_times else float("nan")
    peak_mem = (
        float(torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else float("nan")
    )
    # best epoch 的逐窗口误差单独落盘：ndarray 不能进 JSON，且按下标与其他格子对齐。
    #
    # ⭐ 为什么**验证段**也必须存：门控器（selector）如果在测试窗口上训练、又在测试窗口上
    # 评估，等价于用测试标签选插件，审稿人一眼就否。正确做法是「在验证窗口上学门控、
    # 在测试窗口上评估」——这既无泄漏，也正好是真实部署时能拿到的信息。
    # 验证段前向本来每个 epoch 都要跑，多存一个 float32 数组是零额外算力。
    win = best_test.pop("window_mse", None)
    win_val = best_val_full.pop("window_mse", None)
    n_test_windows = int(win.size) if win is not None and win.size else 0
    n_val_windows = int(win_val.size) if win_val is not None and win_val.size else 0
    # ⚠️ 截断/试跑的逐窗口数组绝不能落到正式路径。
    # `--max-eval-steps N` 只累计前 N 个 batch，数组长度远小于真实窗口数；
    # 而下游 selector_window 用 `n = min(各臂长度)` 把**全组所有臂**静默截断到最短那个，
    # 于是一次手工烟囱跑就能让该组的 oracle 上界 / best_fixed / 滞后一致率
    # 全部变成只在前几百个窗口上算出的数字，且结果表看起来完全正常。
    # 同理 probe 只为测速，不该覆盖正式误差文件。
    truncated = run.max_eval_steps is not None
    if run.save_window_mse and (truncated or run.probe):
        why = "--max-eval-steps 截断了评估" if truncated else "probe 测速跑"
        print(
            f"[run] ⚠️ {cell.cell_id}: {why}，跳过逐窗口误差落盘"
            f"（避免覆盖正式 winerr 并静默截断整组）",
            flush=True,
        )
    elif run.save_window_mse:
        if n_test_windows:
            save_window_mse_atomic(cfg.path("winerr_dir"), cell.cell_id, win)
        if n_val_windows:
            save_window_mse_atomic(cfg.path("winerr_dir") / "val", cell.cell_id, win_val)
            # 只有走了确定序 val loader 才打标，见 mark_deterministic_order 的说明
            mark_deterministic_order(cfg.path("winerr_dir") / "val", cell.cell_id, n_val_windows)
    result = {
        **cell.to_dict(),
        "cell_id": cell.cell_id,
        "mse": best_test.get("mse", float("nan")),
        "mae": best_test.get("mae", float("nan")),
        "seg_mse": best_test.get("seg_mse", []),
        "n_test_windows": n_test_windows,
        "n_val_windows": n_val_windows,
        "val_mse": best_val,
        "best_epoch": best_epoch,
        "epochs_run": epoch + 1,
        "ms_per_iter": ms_per_iter,
        "train_seconds": round(time.time() - t0, 1),
        "n_params": n_params,
        "extra_params": extra_params,
        "peak_mem_mib": peak_mem,
        "batch_size": configs.batch_size,
        "seq_len": configs.seq_len,
        "use_amp": bool(configs.use_amp),
        "device": str(device),
        "plugin_kind": kind_of(cfg.plugin_impl(cell.plugin)),
        "n_train_windows": len(train_set),
        "probe": bool(run.probe),
        "source": "measured",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    return result


def _forward_loss(model, criterion, configs, bx, by, bxm, bym, device):
    import torch

    bx = bx.float().to(device)
    by = by.float().to(device)
    bxm = bxm.float().to(device)
    bym = bym.float().to(device)
    dec_inp = torch.cat(
        [by[:, : configs.label_len, :], torch.zeros_like(by[:, -configs.pred_len :, :])], dim=1
    ).float()
    f_dim = -1 if configs.features == "MS" else 0
    if configs.use_amp:
        with torch.amp.autocast("cuda"):
            out = model(bx, bxm, dec_inp, bym)
            out = out[:, -configs.pred_len :, f_dim:]
            loss = criterion(out, by[:, -configs.pred_len :, f_dim:])
    else:
        out = model(bx, bxm, dec_inp, bym)
        out = out[:, -configs.pred_len :, f_dim:]
        loss = criterion(out, by[:, -configs.pred_len :, f_dim:])
    aux = model.pop_aux_loss() if hasattr(model, "pop_aux_loss") else 0.0
    return loss + aux if torch.is_tensor(aux) else loss


def _evaluate(
    model, criterion, configs, loader, device, max_steps: int | None, per_window: bool = False
) -> dict[str, Any]:
    import torch

    model.eval()
    m = StreamingMetrics(configs.pred_len, per_window=per_window)
    with torch.no_grad():
        for i, (bx, by, bxm, bym) in enumerate(loader):
            if max_steps is not None and i >= max_steps:
                break
            bx = bx.float().to(device)
            by = by.float().to(device)
            bxm = bxm.float().to(device)
            bym = bym.float().to(device)
            dec_inp = torch.cat(
                [by[:, : configs.label_len, :], torch.zeros_like(by[:, -configs.pred_len :, :])],
                dim=1,
            ).float()
            f_dim = -1 if configs.features == "MS" else 0
            if configs.use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(bx, bxm, dec_inp, bym)
            else:
                out = model(bx, bxm, dec_inp, bym)
            m.update(out[:, -configs.pred_len :, f_dim:], by[:, -configs.pred_len :, f_dim:])
    model.train()
    return m.result()
