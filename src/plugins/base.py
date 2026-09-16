"""插件包装层的基类：零侵入地包住任意 TSLib 骨干。

零侵入的三条硬约束（对应尽调 §5.2 第 10 条 / xCPD 的门槛定义）：
1. **架构无关**：只依赖 TSLib 统一的顶层签名
   ``forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None) -> (B, L_out, C_out)``，
   不改任何骨干文件、不读骨干内部结构；
2. **训练解耦**：插件不引入需要与骨干联合优化的额外阶段；
3. **零新增需逐数据集调的超参**：所有插件超参在 configs/matrix.yaml 里全矩阵固定。
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

Tensor = torch.Tensor


class PluginWrapper(nn.Module):
    """所有插件包装器的基类；默认行为等价于 ``none`` 对照组。"""

    plugin_name: str = "none"
    kind: str = "io"  # io | loss

    def __init__(self, model: nn.Module, **params: Any) -> None:
        super().__init__()
        self.model = model
        self.params = dict(params)
        self._aux_loss: Tensor | None = None

    # ---- 供 Exp 层使用的钩子 ----
    def pop_aux_loss(self) -> Tensor | float:
        """取出并清空本次 forward 产生的辅助损失（本仓库的插件均为 0，保留给未来扩展）。"""
        aux = self._aux_loss
        self._aux_loss = None
        return 0.0 if aux is None else aux

    def extra_param_count(self) -> int:
        """插件自身新增的可训练参数量（用于论文的效率表）。"""
        inner = sum(p.numel() for p in self.model.parameters())
        total = sum(p.numel() for p in self.parameters())
        return total - inner

    # ---- 前向 ----
    def forward(
        self,
        x_enc: Tensor,
        x_mark_enc: Tensor | None = None,
        x_dec: Tensor | None = None,
        x_mark_dec: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)

    def __repr__(self) -> str:  # pragma: no cover - 仅日志可读性
        return f"{self.__class__.__name__}(plugin={self.plugin_name}, inner={type(self.model).__name__})"


class NonePlugin(PluginWrapper):
    """对照组：完全透传，唯一作用是让「有/无插件」两条代码路径完全对称。"""

    plugin_name = "none"


def align_channels(stat: Tensor, out: Tensor) -> Tensor:
    """把 (B,1,C_in) 的统计量对齐到输出的通道数（TSLib 的 MS 模式下 C_out < C_in）。"""
    c_out = out.shape[-1]
    if stat.shape[-1] == c_out:
        return stat
    if stat.shape[-1] > c_out:
        return stat[..., -c_out:]
    return stat.expand(*stat.shape[:-1], c_out)
