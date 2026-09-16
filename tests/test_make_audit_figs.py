"""`tools/make_audit_figs.py` 的测试。

图本身没法做像素级断言，所以这里只钉三件真正会出错的事：
1. 抖动量必须可复现（曾经用过 `hash(str)`，受 PYTHONHASHSEED 影响，图每次都在动）；
2. 两张图 + 分层 CSV 必须真的落盘，且 png/pdf 都有；
3. 返回的统计量要与输入数据一致（防止图与正文数字脱节）。
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from make_audit_figs import (  # noqa: E402
    fig_decay_curves,
    fig_halflife_vs_horizon,
    jitter_factor,
)
from src.selector_window import LAGS_ABS  # noqa: E402


def _df(n_bb: int = 2, n_ds: int = 2) -> pd.DataFrame:
    rows = []
    for bi in range(n_bb):
        for di in range(n_ds):
            for H in (96, 336, 720):
                hl = 6.0 + 2 * bi + 3 * di
                r = {"backbone": f"BB{bi}", "dataset": f"DS{di}", "pred_len": H,
                     "persist_half_life_windows": hl, "half_life_over_H": hl / H,
                     "excess_persist_lag1": 0.6, "excess_persist_lagH": -0.01,
                     "z_lag1": 60.0, "z_lagH": -1.0, "chance_persist": 0.28,
                     "oracle_gain_real_pct": 10.0}
                for L in LAGS_ABS:
                    r[f"excess_abs{L}"] = 0.6 * 0.5 ** (L / hl)
                rows.append(r)
    return pd.DataFrame(rows)


def test_jitter_is_deterministic_and_bounded():
    a = [jitter_factor(b, d, 8, 4) for b in range(4) for d in range(8)]
    b = [jitter_factor(b, d, 8, 4) for b in range(4) for d in range(8)]
    assert a == b                              # 同一进程内稳定
    assert all(0.9 <= x <= 1.1 for x in a)     # 抖动不能大到改变读图结论
    # 每个 (backbone,dataset) 组合必须落在不同的 x 上：早期的 `% 7` 让 32 个组合
    # 只占 7 个位置，H=96 那一列上有 4 个点精确重叠，读图时数不出有几个格子。
    assert len(set(a)) == 32
    # 对称：抖动不能把整片点云系统性地推向某一侧
    assert abs(sum(x - 1.0 for x in a)) < 1e-9


def test_jitter_survives_a_fresh_interpreter_with_hash_randomisation():
    """真正的回归点：换一个 PYTHONHASHSEED 的新进程，抖动序列必须完全一样。"""
    code = ("import sys; sys.path.insert(0, %r);"
            "from make_audit_figs import jitter_factor;"
            "print([round(jitter_factor(b, d, 8, 4), 6) for b in range(3) for d in range(8)])"
            % str(ROOT / "tools"))
    outs = []
    for seed in ("0", "1", "12345"):
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin",
                                "PYTHONPATH": str(ROOT)})
        assert p.returncode == 0, p.stderr
        outs.append(p.stdout.strip())
    assert len(set(outs)) == 1, f"抖动随 PYTHONHASHSEED 变了: {outs}"


def test_halflife_figure_writes_png_and_pdf_and_reports_consistent_stats(tmp_path):
    df = _df()
    m = fig_halflife_vs_horizon(df, tmp_path / "fig1", dpi=60)
    for ext in ("png", "pdf"):
        f = tmp_path / f"fig1.{ext}"
        assert f.exists() and f.stat().st_size > 1000, ext
    assert m["n_points"] == len(df)
    # 返回的比例必须与数据一致，不能是写死的
    assert abs(m["frac_below_H_over_10"] - float((df["half_life_over_H"] < 0.1).mean())) < 1e-12
    assert set(m["median_ratio_by_H"]) == {"96", "336", "720"}


def test_decay_curve_figure_covers_three_cuts(tmp_path):
    info = fig_decay_curves(_df(), tmp_path / "fig2", dpi=60)
    assert set(info) == {"backbone", "dataset", "pred_len"}
    assert (tmp_path / "fig2.png").exists() and (tmp_path / "fig2.pdf").exists()
    assert set(info["pred_len"]) == {"96", "336", "720"}


def test_decay_curve_panel_c_labels_its_vertical_delay_lines(tmp_path, monkeypatch):
    """(c) 面板给每个 H 画了一条同色竖虚线（= 该层唯一合法的决策延迟 H）。
    图例必须说明它是什么，否则读者只会把本图最关键的元素当成装饰。"""
    import matplotlib.pyplot as plt
    seen: list[list[str]] = []
    orig = plt.Axes.legend

    def spy(self, *a, **kw):                    # noqa: ANN001
        seen.append([t.get_label() for t in self.get_lines()])
        return orig(self, *a, **kw)

    monkeypatch.setattr(plt.Axes, "legend", spy)
    fig_decay_curves(_df(), tmp_path / "fig2c", dpi=60)
    # 第三个面板（by horizon H）的图例里必须有那一条解释竖线的条目
    assert any("legal delay H" in lab for labs in seen for lab in labs), seen
    # 前两个面板不该出现这条说明（它们没有画竖线）
    assert not any("legal delay H" in lab for lab in seen[0])


def test_cli_exits_nonzero_with_a_clear_hint_when_audit_csv_missing(tmp_path):
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "make_audit_figs.py"),
                        "--audit-csv", str(tmp_path / "nope.csv"),
                        "--out-dir", str(tmp_path / "out")],
                       capture_output=True, text=True)
    assert p.returncode == 1
    assert "oracle-audit" in p.stdout


def test_cli_end_to_end_on_synthetic_audit_csv(tmp_path):
    csv = tmp_path / "audit.csv"
    _df(2, 3).to_csv(csv, index=False)
    out = tmp_path / "paper"
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "make_audit_figs.py"),
                        "--audit-csv", str(csv), "--out-dir", str(out), "--dpi", "60"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    for name in ("fig_halflife_vs_horizon.png", "fig_halflife_vs_horizon.pdf",
                 "fig_decay_curves.png", "fig_decay_curves.pdf", "audit_strata.csv"):
        assert (out / name).exists(), name
    strata = pd.read_csv(out / "audit_strata.csv")
    assert set(strata["stratum_key"]) == {"backbone", "dataset", "pred_len"}
    assert "半衰期" in p.stdout


# --------------------------------------------------------------------------- #
# 2026-09-14 code review 的 P1 回归：缺失值污染（NaN 计入分母 / 各滞后点换了组集合）
# 期望值全部手算：把每个格子的数字直接写在构造函数里，再在断言里手写平均。
# --------------------------------------------------------------------------- #
def _df_with_undefined_half_life() -> pd.DataFrame:
    """6 个组，其中 2 个的半衰期是 NaN（`evaluate_argmin_structure` 在 excess_abs1<=0
    或衰减曲线被截断时就会置 NaN）。有定义的 4 个里，恰好 2 个的 半衰期/H < 0.1。

    手算：正确比例 = 2/4 = 0.50；把 NaN 当 False 计入分母的旧写法 = 2/6 ≈ 0.333。
    """
    ratios = [0.05, 0.08, 0.20, 0.30, np.nan, np.nan]     # 前两个 < 0.1
    rows = []
    for i, r in enumerate(ratios):
        H = (96, 336)[i % 2]
        hl = np.nan if r != r else r * H
        row = {"backbone": f"BB{i % 2}", "dataset": f"DS{i % 3}", "pred_len": H,
               "persist_half_life_windows": hl, "half_life_over_H": r,
               "excess_persist_lag1": 0.6, "excess_persist_lagH": -0.01,
               "z_lag1": 60.0, "z_lagH": -1.0, "chance_persist": 0.28,
               "oracle_gain_real_pct": 10.0}
        for L in LAGS_ABS:
            row[f"excess_abs{L}"] = 0.6 * 0.5 ** (L / 8.0)
        rows.append(row)
    return pd.DataFrame(rows)


def test_frac_below_H_over_10_excludes_groups_without_a_defined_half_life(tmp_path):
    """修复项 #5：比例只能在 half_life_over_H **非 NaN** 的组上算。

    旧写法 `(df["half_life_over_H"] < 0.1).mean()` 把 NaN 当 False 计入分母，
    于是 main() 打出的「X% 的组半衰期 < H/10」被未定义组稀释（0.50 -> 0.33）。
    """
    df = _df_with_undefined_half_life()
    m = fig_halflife_vs_horizon(df, tmp_path / "fig1", dpi=60)
    assert m["frac_below_H_over_10"] == pytest.approx(0.5)      # 2/4，手算
    assert m["frac_below_H_over_10"] != pytest.approx(2 / 6)    # 旧口径必须被排除
    # 「有定义 / 未定义」组数要一起返回，供报告如实交代样本口径
    assert m["n_groups_total"] == 6
    assert m["n_groups_half_life_defined"] == 4
    assert m["n_groups_half_life_undefined"] == 2
    # n_points 是**实际画出的点数**，NaN 组画不出点
    assert m["n_points"] == 4


def test_halflife_stats_are_all_nan_safe_when_nothing_is_defined(tmp_path):
    """极端情况：所有组的半衰期都未定义。比例只能是 NaN，绝不能是 0%（那是个假结论）。"""
    df = _df_with_undefined_half_life()
    df = df.assign(half_life_over_H=np.nan, persist_half_life_windows=np.nan)
    m = fig_halflife_vs_horizon(df, tmp_path / "fig1b", dpi=60)
    assert m["n_points"] == 0 and m["n_groups_half_life_undefined"] == 6
    assert m["frac_below_H_over_10"] != m["frac_below_H_over_10"]   # NaN
    assert (tmp_path / "fig1b.png").exists()


def _df_with_truncated_long_lags() -> tuple[pd.DataFrame, dict]:
    """同一层里混入「测试窗口很短」的组：它们在长滞后处是 NaN（L >= n_test_windows）。

    构造：4 个组同属 backbone=BB0，excess_abs{L} 全部写成常数，便于手算均值。
      * 2 个长序列组：所有滞后都有值，值 = 0.40 / 0.20；
      * 2 个短序列组：lag >= 256 处为 NaN，值 = 0.10 / 0.10。
    于是 lag<=128 的均值 = (0.40+0.20+0.10+0.10)/4 = 0.20（全组支撑），
    而 lag>=256 若按列跳过 NaN 就变成 (0.40+0.20)/2 = 0.30——不同滞后落在不同组子集上，
    「尾部压到 0」可能纯粹是换了组集合造成的。
    """
    vals = [0.40, 0.20, 0.10, 0.10]
    short = [False, False, True, True]
    rows = []
    for i, (v, is_short) in enumerate(zip(vals, short)):
        row = {"backbone": "BB0", "dataset": f"DS{i}", "pred_len": 96,
               "persist_half_life_windows": 8.0, "half_life_over_H": 8.0 / 96,
               "excess_persist_lag1": v, "excess_persist_lagH": -0.01,
               "z_lag1": 60.0, "z_lagH": -1.0, "chance_persist": 0.28,
               "oracle_gain_real_pct": 10.0}
        for L in LAGS_ABS:
            row[f"excess_abs{L}"] = np.nan if (is_short and L >= 256) else v
        rows.append(row)
    return pd.DataFrame(rows), {"full_mean": 0.20, "diluted_mean": 0.30,
                                "last_full_lag": 128}


def test_decay_curve_drops_lags_without_full_group_support(tmp_path, monkeypatch):
    """修复项 #6：曲线上每个滞后点必须在**同一批组**上取均值。

    支撑不全的滞后要置 NaN（plot 自然断线），不能拿「只剩长序列组」的均值接在后面。
    """
    import matplotlib.pyplot as plt
    df, exp = _df_with_truncated_long_lags()
    curves: list[np.ndarray] = []
    orig = plt.Axes.plot

    def spy(self, *a, **kw):                     # noqa: ANN001
        if len(a) >= 2 and np.ndim(a[1]) == 1 and len(a[1]) == len(LAGS_ABS):
            curves.append(np.asarray(a[1], dtype=float))
        return orig(self, *a, **kw)

    monkeypatch.setattr(plt.Axes, "plot", spy)
    info = fig_decay_curves(df, tmp_path / "fig2d", dpi=60)

    # (a) by backbone 那条曲线：BB0 只有一层，4 个组
    bb = info["backbone"]["BB0"]
    assert bb["n_groups"] == 4
    assert bb["last_full_support_lag"] == exp["last_full_lag"]
    assert bb["excess_at_last_full_support_lag"] == pytest.approx(exp["full_mean"])
    assert bb["excess_at_last_full_support_lag"] != pytest.approx(exp["diluted_mean"])
    # 每个滞后的有效组数必须一并给出，图例/正文才能交代口径
    assert bb["n_valid_by_lag"]["128"] == 4
    assert bb["n_valid_by_lag"]["512"] == 2

    # 真正画出去的 y：全支撑的滞后有值、支撑不全的滞后是 NaN（断线）
    idx = {L: i for i, L in enumerate(LAGS_ABS)}
    bb_curves = [y for y in curves if np.isfinite(y[idx[1]])
                 and y[idx[1]] == pytest.approx(exp["full_mean"])]
    assert bb_curves, curves
    y = bb_curves[0]
    assert y[idx[128]] == pytest.approx(exp["full_mean"])
    for L in (256, 512, 1024):
        assert not np.isfinite(y[idx[L]]), f"lag {L} 支撑不全却仍被画出：{y}"


def test_decay_curve_legend_discloses_the_last_fully_supported_lag(tmp_path, monkeypatch):
    """图例只写 `n=组数` 会骗人：读者会以为整条曲线都建立在这 n 个组上。
    必须把「到哪个滞后为止是全组支撑」标进图例。"""
    import matplotlib.pyplot as plt
    df, exp = _df_with_truncated_long_lags()
    labels: list[str] = []
    orig = plt.Axes.legend

    def spy(self, *a, **kw):                     # noqa: ANN001
        labels.extend(t.get_label() for t in self.get_lines())
        return orig(self, *a, **kw)

    monkeypatch.setattr(plt.Axes, "legend", spy)
    fig_decay_curves(df, tmp_path / "fig2e", dpi=60)
    assert any(f"full support ≤{exp['last_full_lag']}" in lab for lab in labels), labels


def test_cli_reports_the_sample_scope_of_both_figures(tmp_path):
    """终端输出（会被 weekend_report 抄进执行日志）必须交代两件事：
    半衰期未定义的组数、以及衰减曲线的全支撑滞后——不然还是会被当成全组结论引用。"""
    df, _ = _df_with_truncated_long_lags()
    df.loc[0, ["persist_half_life_windows", "half_life_over_H"]] = np.nan
    csv = tmp_path / "audit.csv"
    df.to_csv(csv, index=False)
    p = subprocess.run([sys.executable, str(ROOT / "tools" / "make_audit_figs.py"),
                        "--audit-csv", str(csv), "--out-dir", str(tmp_path / "out"),
                        "--dpi", "60"], capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "未定义 1 组" in p.stdout
    assert "有定义的组 3/4" in p.stdout
    assert "全支撑" in p.stdout and "lag<=128" in p.stdout
