"""插件注册表与统一入口。

用法（Exp 层只需要这两个函数，完全不必知道插件内部）：

    from src.plugins import wrap, make_criterion
    model = wrap(backbone_model, "revin", n_channels=7)      # io 型插件
    criterion = make_criterion("fredf", alpha=0.5)           # loss 型插件

设计：io 型插件包住 forward，loss 型插件替换 criterion；两者互不干扰，
因此未来要做「revin+fredf」这种组合插件时只需在这里组合，无需改任何骨干代码。
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from .base import NonePlugin, PluginWrapper
from .fredf import FreDFLoss
from .fredf_sqrth import FreDFSqrtHLoss
from .revin import RevINPlugin
from .san_lite import SANLitePlugin

# name -> (wrapper 构造器 或 None, kind)
_IO_PLUGINS: dict[str, type[PluginWrapper]] = {
    "none": NonePlugin,
    "revin": RevINPlugin,
    "san_lite": SANLitePlugin,
    "fredf": NonePlugin,  # fredf 只改损失，模型侧透传
    "fredf_sqrth": NonePlugin,  # 同上：√H 归一化变体也只改损失
}

_LOSS_PLUGINS: dict[str, Callable[..., nn.Module]] = {
    "fredf": FreDFLoss,
    "fredf_sqrth": FreDFSqrtHLoss,
}

PLUGIN_KINDS: dict[str, str] = {
    "none": "io",
    "revin": "io",
    "san_lite": "io",
    "fredf": "loss",
    "fredf_sqrth": "loss",
}


def available() -> list[str]:
    """当前可用插件名（含对照组 none）。"""
    return list(_IO_PLUGINS)


def kind_of(plugin: str) -> str:
    if plugin not in PLUGIN_KINDS:
        raise KeyError(f"未知插件 {plugin!r}，可用: {available()}")
    return PLUGIN_KINDS[plugin]


def wrap(model: nn.Module, plugin: str, **params: Any) -> PluginWrapper:
    """把骨干模型包成带插件的模型；`plugin='none'` 返回透传包装（保持代码路径对称）。

    未识别的 params 会被忽略（各插件只取自己认识的键），这样 matrix.yaml 里
    的插件参数可以直接整块传进来。
    """
    if plugin not in _IO_PLUGINS:
        raise KeyError(f"未知插件 {plugin!r}，可用: {available()}")
    cls = _IO_PLUGINS[plugin]
    accepted = _accepted_kwargs(cls)
    kwargs = {k: v for k, v in params.items() if k in accepted}
    return cls(model, **kwargs)


def make_criterion(plugin: str, default: nn.Module | None = None, **params: Any) -> nn.Module:
    """返回该插件对应的训练损失；非 loss 型插件返回默认的 MSELoss。"""
    if plugin in _LOSS_PLUGINS:
        cls = _LOSS_PLUGINS[plugin]
        accepted = _accepted_kwargs(cls)
        return cls(**{k: v for k, v in params.items() if k in accepted})
    return default if default is not None else nn.MSELoss()


def _accepted_kwargs(fn: Any) -> set[str]:
    import inspect

    sig = inspect.signature(fn.__init__ if isinstance(fn, type) else fn)
    return {p for p in sig.parameters if p not in ("self", "model")}


__all__ = [
    "available",
    "kind_of",
    "wrap",
    "make_criterion",
    "PluginWrapper",
    "NonePlugin",
    "RevINPlugin",
    "SANLitePlugin",
    "FreDFLoss",
    "PLUGIN_KINDS",
]
