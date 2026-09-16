"""fp32-FFT 兼容补丁的单元测试。

真正的故障只在 CUDA + fp16 autocast 下复现（cuFFT 要求 2 的幂长度），
所以这里分两层：
- CPU 上验证补丁的幂等性、数值一致性（不改变结果）与签名透明；
- 有 GPU 时才跑真实的 fp16 autocast 非 2 的幂长度回归用例。
"""

from __future__ import annotations

import pytest
import torch

from src.fft_compat import apply_fft_fp32_patch


def test_patch_is_idempotent():
    first = set(apply_fft_fp32_patch())
    second = set(apply_fft_fp32_patch())
    # 第二次不应重复包装（可能第一次已被 runner import 时打过，故只断言第二次为空）
    assert second == set()
    assert isinstance(first, set)


def test_cpu_results_unchanged():
    apply_fft_fp32_patch()
    x = torch.randn(4, 192, 7)
    freq = torch.fft.rfft(x, dim=1)
    back = torch.fft.irfft(freq, n=192, dim=1)
    assert freq.shape == (4, 97, 7)
    assert torch.allclose(back, x, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU 才能复现 cuFFT fp16 限制")
def test_cuda_fp16_autocast_non_power_of_two():
    apply_fft_fp32_patch()
    x = torch.randn(8, 192, 21, device="cuda")  # 192 不是 2 的幂
    with torch.amp.autocast("cuda", dtype=torch.float16):
        h = torch.nn.functional.linear(x, torch.eye(21, device="cuda"))
        assert h.dtype == torch.float16
        freq = torch.fft.rfft(h, dim=1)
        amp = abs(freq).mean(dim=(0, 2))
        back = torch.fft.irfft(freq, n=192, dim=1)
    assert amp.shape == (97,)
    assert back.shape == (8, 192, 21)
    assert torch.isfinite(back).all()
