"""测试用的合成信号工厂。

设计原则：所有断言都基于**性质**（谁比谁大、是否在合法区间、是否可逆），
不硬编码任何浮点数，这样重构实现细节时测试不会假失败。
"""

from __future__ import annotations

import numpy as np
import pytest

T = 4000
PERIOD = 24


def _z(x: np.ndarray) -> np.ndarray:
    return (x - x.mean(0)) / (x.std(0) + 1e-8)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(26)


@pytest.fixture(scope="session")
def white_noise() -> np.ndarray:
    """白噪声：模式复杂度上界、无季节、平稳、通道独立。"""
    return _z(np.random.default_rng(0).standard_normal((T, 6)))


@pytest.fixture(scope="session")
def seasonal() -> np.ndarray:
    """强季节 + 弱噪声：低排列熵/低谱熵、季节强度高、频域能量集中。"""
    t = np.arange(T)[:, None]
    g = np.random.default_rng(1)
    base = np.sin(2 * np.pi * t / PERIOD) + 0.3 * np.sin(2 * np.pi * t / (PERIOD * 7))
    return _z(base + 0.1 * g.standard_normal((T, 6)))


@pytest.fixture(scope="session")
def random_walk() -> np.ndarray:
    """随机游走：非平稳（ADF 不显著）、趋势强度高、ACF 衰减极慢。"""
    return _z(np.cumsum(np.random.default_rng(2).standard_normal((T, 6)), axis=0))


@pytest.fixture(scope="session")
def rank_one() -> np.ndarray:
    """秩 1 多通道：一个公共因子 + 微噪声，通道有效秩应接近 1/C。"""
    g = np.random.default_rng(3)
    f = np.sin(2 * np.pi * np.arange(T) / PERIOD) + np.cumsum(g.standard_normal(T)) * 0.02
    X = f[:, None] * g.uniform(0.8, 1.2, size=(1, 8)) + 0.01 * g.standard_normal((T, 8))
    return _z(X)
