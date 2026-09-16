"""cuFFT + fp16 AMP 兼容补丁。

问题
----
V100 (sm_70) 有 fp16 Tensor Core 但无 bf16，所以 Electricity / Traffic 这两个
重数据集开 `torch.amp.autocast("cuda")`（fp16）来换速度。但 TimesNet 的
`FFT_for_Period` 以及 FredF 插件的频域损失都会在 autocast 区域内调用
`torch.fft.rfft / irfft`，此时张量已是 half，cuFFT 抛：

    RuntimeError: cuFFT only supports dimensions whose sizes are powers of two
    when computing in half precision, but got a signal size of [192]

即 fp16 cuFFT 只支持 2 的幂长度，而 seq_len+pred_len=192/816 都不是。

方案
----
不改 TSLib 源码（本项目的零侵入原则），改为在进程启动时把
`torch.fft.{rfft,irfft,fft,ifft}` 包一层：进入时关闭 autocast 并把实数输入
提升到 fp32 做变换，出来后再转回调用方 dtype。

- FFT 只占 TimesNet 极小一部分计算量（主体是 Inception 2D 卷积），
  单独用 fp32 做变换对整体吞吐几乎无影响；
- fp32 FFT 精度严格优于 fp16，不存在“为了跑通牺牲数值”的问题；
- 补丁幂等，重复 import 不会叠加多层包装。
"""

from __future__ import annotations

from typing import Any, Callable

import torch

_PATCHED_FLAG = "_pluggate_fft_fp32_patched"
_WRAPPED_FUNCS = ("rfft", "irfft", "fft", "ifft", "rfft2", "irfft2")


def _wrap(fn: Callable[..., Any]) -> Callable[..., Any]:
    def wrapper(input: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:  # noqa: A002
        if not (isinstance(input, torch.Tensor) and input.is_cuda):
            return fn(input, *args, **kwargs)
        in_dtype = input.dtype
        # half / bfloat16 实数 与 complex32 都走 fp32 通道
        needs_cast = in_dtype in (torch.float16, torch.bfloat16, torch.complex32)
        with torch.amp.autocast("cuda", enabled=False):
            x = input
            if needs_cast:
                x = input.to(torch.complex64 if input.is_complex() else torch.float32)
            out = fn(x, *args, **kwargs)
        if needs_cast and isinstance(out, torch.Tensor):
            # irfft 出来是实数、rfft 出来是复数，分别映射回原精度族
            if out.is_complex():
                out = out.to(torch.complex32) if in_dtype == torch.complex32 else out
            else:
                out = out.to(in_dtype if not torch.is_complex(input) else torch.float16)
        return out

    setattr(wrapper, _PATCHED_FLAG, True)
    wrapper.__name__ = getattr(fn, "__name__", "fft_fn")
    wrapper.__doc__ = f"[PlugGate fp32-FFT 包装] {getattr(fn, '__doc__', '') or ''}"
    return wrapper


def apply_fft_fp32_patch() -> list[str]:
    """把 torch.fft 的常用变换包成 fp32 执行。返回本次实际打补丁的函数名列表。"""
    patched: list[str] = []
    for name in _WRAPPED_FUNCS:
        fn = getattr(torch.fft, name, None)
        if fn is None or getattr(fn, _PATCHED_FLAG, False):
            continue
        setattr(torch.fft, name, _wrap(fn))
        patched.append(name)
    return patched
