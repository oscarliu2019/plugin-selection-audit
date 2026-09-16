"""实验矩阵配置载入与「单格实验」(Cell) 的定义、枚举、成本估计。

设计原则：所有可变的东西都在 configs/matrix.yaml 里，代码只负责解释它。
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "matrix.yaml"


# --------------------------------------------------------------------------- #
# 配置对象
# --------------------------------------------------------------------------- #
class MatrixConfig:
    """configs/matrix.yaml 的薄封装（只读）。"""

    def __init__(self, raw: dict[str, Any], path: Path | None = None) -> None:
        self.raw = raw
        self.config_path = path
        self.meta: dict[str, Any] = raw["meta"]
        self.protocol: dict[str, Any] = raw["meta"]["protocol"]
        self.backbones: list[dict[str, Any]] = raw["backbones"]
        self.datasets: list[dict[str, Any]] = raw["datasets"]
        self.plugins: list[dict[str, Any]] = raw["plugins"]
        self.seed_policy: dict[str, Any] = raw["seed_policy"]
        self.cost_model: dict[str, Any] = raw["cost_model"]
        self.paths: dict[str, str] = raw["paths"]

    # ---- 载入 ----
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "MatrixConfig":
        p = Path(path) if path is not None else DEFAULT_CONFIG
        with open(p, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        return cls(raw, p)

    # ---- 查表 ----
    def backbone(self, name: str) -> dict[str, Any]:
        return _find(self.backbones, name, "backbone")

    def dataset(self, name: str) -> dict[str, Any]:
        return _find(self.datasets, name, "dataset")

    def plugin(self, name: str) -> dict[str, Any]:
        return _find(self.plugins, name, "plugin")

    def plugin_impl(self, name: str) -> str:
        """插件**标签名** -> **实现名**。

        用于「同一实现、不同超参」的对照臂：例如 FreDF 公平性实验里需要
        `fredf_a09` / `fredf_auth` / `fredf_sqrth` 等多个标签共用 `fredf` 实现，
        彼此只差 `params`。yaml 里写 `impl: fredf` 即可，缺省时标签名就是实现名。
        """
        return str(self.plugin(name).get("impl") or name)

    def path(self, key: str) -> Path:
        """把 paths.<key> 解析成绝对路径（相对 pluggate/ 根目录）。"""
        return (REPO_ROOT / self.paths[key]).resolve()

    # ---- 名称列表 ----
    @property
    def backbone_names(self) -> list[str]:
        return [b["name"] for b in self.backbones]

    @property
    def plugin_names(self) -> list[str]:
        return [p["name"] for p in self.plugins]

    def dataset_names(self, include_optional: bool = False) -> list[str]:
        return [d["name"] for d in self.datasets if include_optional or not d.get("optional", False)]

    @property
    def control_plugin(self) -> str:
        for p in self.plugins:
            if p.get("is_control"):
                return str(p["name"])
        return "none"

    # ---- 每个数据集的 seed 列表 ----
    def seeds_for(self, dataset: str) -> list[int]:
        ds = self.dataset(dataset)
        seeds = list(self.seed_policy["base_seeds"])
        if ds.get("tier") in self.seed_policy.get("tiers_with_extra_seeds", []):
            seeds += list(self.seed_policy.get("extra_seeds", []))
        return seeds

    def seq_len(self, dataset: str) -> int:
        return int(self.dataset(dataset).get("seq_len", self.protocol["seq_len"]))

    def label_len(self, dataset: str) -> int:
        return int(self.dataset(dataset).get("label_len", self.protocol["label_len"]))


def _find(items: list[dict[str, Any]], name: str, what: str) -> dict[str, Any]:
    for it in items:
        if it["name"] == name:
            return it
    raise KeyError(f"unknown {what}: {name!r} (known: {[i['name'] for i in items]})")


# --------------------------------------------------------------------------- #
# 单格实验
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Cell:
    """矩阵中的一个格子：(骨干, 数据集, horizon, 插件, seed)。"""

    backbone: str
    dataset: str
    pred_len: int
    plugin: str
    seed: int

    @property
    def cell_id(self) -> str:
        """文件系统安全、可逆的唯一标识（用于断点续跑与目录名）。"""
        return f"{self.backbone}__{self.dataset}__h{self.pred_len}__{self.plugin}__s{self.seed}"

    @staticmethod
    def from_id(cell_id: str) -> "Cell":
        bk, ds, h, pl, sd = cell_id.split("__")
        return Cell(bk, ds, int(h[1:]), pl, int(sd[1:]))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CellPlan:
    """一个格子的排产信息（成本估计 + 状态）。"""

    cell: Cell
    iters_per_epoch: int
    ms_per_iter: float
    est_seconds: float
    cost_source: str = "prior"  # prior | measured | measured_backbone | measured_global
    done: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> dict[str, Any]:
        r = self.cell.to_dict()
        r.update(
            cell_id=self.cell.cell_id,
            iters_per_epoch=self.iters_per_epoch,
            ms_per_iter=round(self.ms_per_iter, 3),
            est_seconds=round(self.est_seconds, 1),
            est_hours=round(self.est_seconds / 3600.0, 3),
            cost_source=self.cost_source,
            done=self.done,
        )
        r.update(self.extra)
        return r


def windows_per_epoch(cfg: MatrixConfig, dataset: str, pred_len: int) -> int:
    """训练窗口数 = train_len - seq_len - pred_len + 1（TSLib 滑窗步长 1）。

    已核对：ETTh1/pred96 → 8449、Weather → 36696、Electricity → 18221、
    Traffic → 12089、Exchange → 5120、ILI(36/24) → 617，与尽调 §2.1 表完全一致。
    """
    ds = cfg.dataset(dataset)
    n = int(ds["train_len"]) - cfg.seq_len(dataset) - int(pred_len) + 1
    return max(n, 1)


def iters_per_epoch(cfg: MatrixConfig, dataset: str, pred_len: int) -> int:
    """迭代数 = ceil(窗口数 / batch_size)（drop_last=False）。"""
    bs = int(cfg.dataset(dataset)["batch_size"])
    w = windows_per_epoch(cfg, dataset, pred_len)
    return -(-w // bs)


def enumerate_cells(
    cfg: MatrixConfig,
    include_optional: bool = False,
    backbones: list[str] | None = None,
    datasets: list[str] | None = None,
    plugins: list[str] | None = None,
    horizons: list[int] | None = None,
    seeds: list[int] | None = None,
) -> list[Cell]:
    """按配置（及可选过滤器）展开全部格子。"""
    out: list[Cell] = []
    bks = backbones or cfg.backbone_names
    pls = plugins or cfg.plugin_names
    dss = datasets or cfg.dataset_names(include_optional=include_optional)
    for ds in dss:
        ds_cfg = cfg.dataset(ds)
        hs = horizons or list(ds_cfg["horizons"])
        for h in hs:
            if h not in ds_cfg["horizons"]:
                continue
            sds = seeds if seeds is not None else cfg.seeds_for(ds)
            for bk in bks:
                for pl in pls:
                    for sd in sds:
                        out.append(Cell(bk, ds, int(h), pl, int(sd)))
    return out


# --------------------------------------------------------------------------- #
# 原子落盘小工具（scheduler / runner 共用）
# --------------------------------------------------------------------------- #
def atomic_write_text(path: str | os.PathLike[str], text: str) -> None:
    """同目录 tmp 文件 + os.replace，保证并发/崩溃下不产生半截文件。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".tmp.{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def atomic_write_json(path: str | os.PathLike[str], obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def iter_config_cells(cfg: MatrixConfig) -> Iterator[Cell]:
    yield from enumerate_cells(cfg)
