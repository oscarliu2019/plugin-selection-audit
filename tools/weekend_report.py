"""周一早上要看的那一份东西：把三组实验自动汇总成一份带结论的报告。

设计原则
--------
1. **无人值守要能写出结论，而不是堆数据。** 每一节都以一句「所以呢」开头。
2. **对残缺数据免疫。** 周末的任何一段都可能被截止时间砍掉，报告必须在
   「只有 phase-1」「有 phase-2 但没跑完」「三组都齐」这三种情况下都能生成。
3. **不重复实现已有分析。** phase-1 的统计检验/横向表格直接调 ``src.stats`` /
   ``src.report`` / ``src.horizon``；本模块只负责编排 + 新增的 FreDF α 分析 + 结论。

产物
----
* ``results/weekend_report.md``   —— 人读的主报告（周一直接看这个）
* ``results/weekend_report.json`` —— 机读摘要（后续做飞书文档/图表的数据源）
* ``artifacts/p3/fredf_alpha.csv``—— FreDF α 扫描的逐格明细

    python -m tools.weekend_report            # 全量
    python -m tools.weekend_report --fast     # 跳过耗时的图表生成
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import MatrixConfig  # noqa: E402
from src.scheduler import make_plan  # noqa: E402

PY = sys.executable


# --------------------------------------------------------------------------- #
# 子步骤执行：任何一步失败都不能拖垮整份报告
# --------------------------------------------------------------------------- #
def sh(args: list[str], timeout: int = 3600) -> tuple[bool, str]:
    t0 = time.time()
    try:
        p = subprocess.run(args, cwd=str(ROOT), capture_output=True, text=True, timeout=timeout)
        ok = p.returncode == 0
        out = (p.stdout or "") + (("\n[stderr]\n" + p.stderr[-2000:]) if not ok else "")
    except subprocess.TimeoutExpired:
        ok, out = False, f"超时 {timeout}s"
    except Exception as exc:                                    # noqa: BLE001
        ok, out = False, f"{type(exc).__name__}: {exc}"
    print(f"[report] {'✔' if ok else '✘'} {' '.join(args[-4:])}  ({time.time()-t0:.0f}s)", flush=True)
    return ok, out


def tail(text: str, n: int = 40) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


# --------------------------------------------------------------------------- #
# 覆盖度
# --------------------------------------------------------------------------- #
def coverage(cfg_path: str) -> dict[str, Any]:
    """某个配置跑了多少格、还剩多少、失败几格、剩余预算。"""
    try:
        cfg = MatrixConfig.load(cfg_path)
    except Exception as exc:                                    # noqa: BLE001
        return {"config": cfg_path, "error": str(exc)}
    plans, skipped, cost = make_plan(cfg)
    done = [p for p in plans if p.done]
    todo = [p for p in plans if not p.done]
    cells_dir = cfg.path("cells_dir")
    failed = list(cells_dir.glob("*.failed")) if cells_dir.exists() else []
    winerr = cfg.path("winerr_dir") if "winerr_dir" in cfg.paths else None
    out = {
        "config": cfg_path,
        "n_total": len(plans),
        "n_done": len(done),
        "n_todo": len(todo),
        "n_pruned": len(skipped),
        "n_failed": len(failed),
        "pct_done": round(100.0 * len(done) / max(len(plans), 1), 1),
        "todo_gpu_hours": round(sum(p.est_seconds for p in todo) * cost.safety / 3600.0, 1),
        "failed_examples": [f.name for f in failed[:5]],
    }
    if winerr and winerr.exists():
        out["n_winerr_test"] = len(list(winerr.glob("*.npy")))
        out["n_winerr_val"] = len(list((winerr / "val").glob("*.npy"))) if (winerr / "val").exists() else 0
    # 哪些格子没跑完（按骨干×horizon 聚合，方便一眼看出洞在哪）
    if todo:
        t = pd.DataFrame([{"backbone": p.cell.backbone, "pred_len": p.cell.pred_len} for p in todo])
        out["todo_by_backbone_horizon"] = (
            t.value_counts().unstack(fill_value=0).to_dict()
        )
    return out


# --------------------------------------------------------------------------- #
# FreDF α 分析（本周末的新科学内容）
# --------------------------------------------------------------------------- #
def _alpha_of(plugin: str) -> tuple[str, float] | None:
    """``fredf_a07`` -> ('plain', 0.7)；``fredf_sqrth_a05`` -> ('sqrth', 0.5)。"""
    if plugin == "fredf":
        return "plain", 0.5
    if plugin.startswith("fredf_sqrth_a"):
        return "sqrth", int(plugin.split("_a")[-1]) / 10.0
    if plugin.startswith("fredf_a"):
        return "plain", int(plugin.split("_a")[-1]) / 10.0
    return None


def fredf_analysis(res_p3: pd.DataFrame, res_ref: pd.DataFrame | None) -> dict[str, Any]:
    """回答三件事：调参能挽回多少？√H 归一化有没有用？最优 α 稳不稳？

    协议：α 一律**按验证集 MSE** 选（``val_mse``），再报测试集增益。
    绝不用测试集选 α——那就是我们批评 FreDF 的那种做法。逐格调 α 与「一个 α 通吃所有
    horizon」两处都按这个协议来（后者见 :func:`_best_single_alpha`）；按测试集选出的
    数字**只**以 ``*_oracle_alpha`` / ``*_test_selected`` 这类字段名保留为作弊上界，
    报告里必须标成作弊口径。

    口径纪律（2026-09-14 code review）：任何两个均值相减之前，必须先取**同时非 NaN**
    的那批块（``paired_means``）。未跑完的格子系统性集中在最贵的角落，跨不同块集合
    做减法会带方向性偏差。
    """
    frames = [res_p3]
    if res_ref is not None and len(res_ref):
        # p3 的 plain 臂只有 α∈{0.1,0.3,0.7,0.9}，α=0.5 那一格是 phase-1/2 的 `fredf`。
        # 只取单 seed（优先 2021，与 p3 一致），否则多 seed 会让「同格重复标准差」自检误报。
        ref = res_ref[res_ref["plugin"].isin(["fredf", "none"])].copy()
        if "seed" in ref and (ref["seed"] == 2021).any():
            ref = ref[ref["seed"] == 2021]
        frames.append(ref)
    d = pd.concat(frames, ignore_index=True)
    d = d[d["backbone"] != "TimesNet"] if "backbone" in d else d
    parsed = d["plugin"].map(_alpha_of)
    d = d.assign(
        variant=[p[0] if p else ("none" if pl == "none" else None)
                 for p, pl in zip(parsed, d["plugin"])],
        alpha=[p[1] if p else np.nan for p in parsed],
    ).dropna(subset=["variant"])
    if d.empty:
        return {"n_cells": 0, "note": "还没有 FreDF α 扫描结果"}

    key = ["backbone", "dataset", "pred_len"]
    # 同一格可能在 p2/p3 各跑过一次（none 是刻意重复的确定性自检），取均值并记录偏差
    dup = d.groupby(key + ["plugin"])["mse"].agg(["mean", "std", "count"])
    repeat_max_std = float(dup["std"].max(skipna=True)) if (dup["count"] > 1).any() else float("nan")
    g = d.groupby(key + ["variant", "alpha"], dropna=False).agg(
        mse=("mse", "mean"), val=("val_mse", "mean")).reset_index()

    base = g[g["variant"] == "none"].set_index(key)[["mse", "val"]].rename(
        columns={"mse": "mse_none", "val": "val_none"})
    arms = g[g["variant"] != "none"].join(base, on=key, how="inner")
    if arms.empty:
        return {"n_cells": int(len(d)), "note": "缺 none 对照，无法算增益"}
    arms["gain"] = 100.0 * (arms["mse_none"] - arms["mse"]) / arms["mse_none"]
    # 2026-09-14 code review 修复项 #1 的前置：以前只算了测试集增益，于是「最优单一 α」
    # 只能靠 max(测试增益) 来挑，那就是按测试表现选 α（本函数 docstring 明令禁止的做法）。
    # 这里把**验证集**增益也算出来，让选 α 的那一步有合法依据可用。
    arms["val_gain"] = 100.0 * (arms["val_none"] - arms["val"]) / arms["val_none"]

    rows: list[dict[str, Any]] = []
    for (bb, ds, pl), grp in arms.groupby(key):
        r: dict[str, Any] = {"backbone": bb, "dataset": ds, "pred_len": pl,
                             "mse_none": float(grp["mse_none"].iloc[0])}
        for var in ("plain", "sqrth"):
            sub = grp[grp["variant"] == var]
            if sub.empty:
                continue
            # 用验证集选 α（诚实协议），报测试增益
            pick = sub.loc[sub["val"].idxmin()]
            r[f"{var}_best_alpha_by_val"] = float(pick["alpha"])
            r[f"{var}_gain_at_best_alpha"] = float(pick["gain"])
            r[f"{var}_gain_oracle_alpha"] = float(sub["gain"].max())
            r[f"{var}_gain_worst_alpha"] = float(sub["gain"].min())
            r[f"{var}_n_alphas"] = int(len(sub))
            half = sub[np.isclose(sub["alpha"], 0.5)]
            if len(half):
                r[f"{var}_gain_at_alpha05"] = float(half["gain"].iloc[0])
        rows.append(r)
    det = pd.DataFrame(rows)

    def m(col: str) -> float:
        return float(det[col].mean()) if col in det else float("nan")

    def paired_means(cols: list[str]) -> tuple[dict[str, float], int]:
        """在**所有指定列同时非 NaN** 的那批块上分别求均值。

        2026-09-14 code review 修复项 #2：``det[col].mean()`` 会逐列跳过 NaN，而
        ``plain_gain_at_alpha05`` 只在 α=0.5 那一行存在时才写入（α=0.5 只来自
        p2/phase-1 的 ``fredf`` 臂）。于是「调参后」在全部块上取均值、「固定 α=0.5」
        只在有参考行的块上取均值，两者相减不是「同一批块上调参挽回了多少」。
        未跑完的格子系统性集中在最贵的角落（TimesNet × 长 horizon × 大数据集），
        所以那个差值带方向性偏差，而它又驱动 verdict 与 next_steps 的阈值分支。
        """
        if any(c not in det for c in cols):
            return {}, 0
        p = det[cols].dropna()
        if p.empty:
            return {}, 0
        return {c: float(p[c].mean()) for c in cols}, int(len(p))

    summary: dict[str, Any] = {
        "n_cells": int(len(d)),
        "n_blocks": int(len(det)),
        "repeat_max_std_mse": repeat_max_std,      # none 重复跑的 MSE 标准差，应 ≈ 0
    }
    # 1) 调参差距：固定 α=0.5 -> 按验证集调 α。三个口径必须在**同一批配对块**上算，
    #    否则差值混入了「哪些块跑完了」的偏差。全块口径另存 *_all_blocks，不参与减法。
    pm, n_pair = paired_means(["plain_gain_at_alpha05", "plain_gain_at_best_alpha",
                               "plain_gain_oracle_alpha"])
    nan = float("nan")
    summary["mean_gain_fixed_alpha05"] = pm.get("plain_gain_at_alpha05", nan)
    summary["mean_gain_tuned_alpha"] = (pm.get("plain_gain_at_best_alpha", nan) if n_pair
                                        else m("plain_gain_at_best_alpha"))
    summary["mean_gain_oracle_alpha"] = (pm.get("plain_gain_oracle_alpha", nan) if n_pair
                                         else m("plain_gain_oracle_alpha"))
    summary["tuning_gap"] = (pm["plain_gain_at_best_alpha"] - pm["plain_gain_at_alpha05"]
                             if n_pair else nan)
    summary["n_blocks_paired_alpha05"] = n_pair          # 报告要标出这个配对块数
    summary["mean_gain_tuned_alpha_all_blocks"] = m("plain_gain_at_best_alpha")
    summary["mean_gain_oracle_alpha_all_blocks"] = m("plain_gain_oracle_alpha")
    # 2) √H 归一化：同样必须配对（sqrth 的 α=0.5 也可能在某些块上缺失）
    pms, n_pair_s = paired_means(["sqrth_gain_at_alpha05", "sqrth_gain_at_best_alpha"])
    summary["mean_gain_sqrth_tuned"] = (pms.get("sqrth_gain_at_best_alpha", nan) if n_pair_s
                                        else m("sqrth_gain_at_best_alpha"))
    summary["mean_gain_sqrth_at_alpha05"] = pms.get("sqrth_gain_at_alpha05", nan)
    summary["n_blocks_paired_sqrth_alpha05"] = n_pair_s
    summary["mean_gain_sqrth_tuned_all_blocks"] = m("sqrth_gain_at_best_alpha")
    if "plain_gain_at_best_alpha" in det:
        summary["blocks_tuned_alpha_positive_pct"] = float(
            100.0 * (det["plain_gain_at_best_alpha"] > 0).mean())
    # 3) 最优 α 跨 horizon 稳不稳：同一 (backbone,dataset) 下最优 α 的取值个数
    for var in ("plain", "sqrth"):
        col = f"{var}_best_alpha_by_val"
        if col in det:
            n_uni = det.groupby(["backbone", "dataset"])[col].nunique()
            summary[f"{var}_mean_distinct_best_alpha_across_horizons"] = float(n_uni.mean())
        summary.update(_best_single_alpha(arms, key, var))
    return {"summary": summary, "detail": det}


def _best_single_alpha(arms: pd.DataFrame, key: list[str], var: str) -> dict[str, Any]:
    """「一个 α 通吃所有 horizon」到底能拿到多少测试增益——**按验证集**挑那个 α。

    2026-09-14 code review 修复项 #1（最严重，协议违规）：旧实现是
    ``arms[...].groupby("alpha")["gain"].mean().max()``，其中 ``gain`` 是**测试集**增益，
    取 max 等于按测试表现选 α——正是本模块 docstring 明令禁止、也是我们批评 FreDF 的
    那种做法。而 ``fredf_verdict`` 又拿它下「归一化让『一个 α 通吃』变得可行，可以作为
    独立方法贡献」这种方法论结论，报告 §3 的标签还只写「最优单一 α 的平均增益」，
    读者根本看不出它与同表的「按测试集选 α（作弊上界）」同属作弊口径。

    正确口径：
    1. 用验证集增益的块均值挑 α（诚实协议），再报**该 α** 的测试集平均增益；
    2. 只在「各 α 覆盖块数一致」的那批块上比较——某个 α 缺了几个块时，各 α 的均值
       落在不同块集合上，比较本身就没有意义（与修复项 #2 同源的错误）；
    3. 测试集上界保留，但另起字段名 ``*_mean_gain_test_selected``，报告里标成作弊口径。
    """
    sub = arms[arms["variant"] == var]
    if sub.empty:
        return {}
    piv_val = sub.pivot_table(index=key, columns="alpha", values="val_gain")
    piv_test = sub.pivot_table(index=key, columns="alpha", values="gain")
    # 「覆盖块数一致」= 只留下所有 α 都跑过（且验证/测试都有数）的块
    common = piv_val.dropna(how="any").index.intersection(piv_test.dropna(how="any").index)
    if not len(common) or not piv_val.shape[1]:
        return {}
    pv, pt = piv_val.loc[common], piv_test.loc[common]
    a_val = pv.mean().idxmax()               # 诚实：按验证集均值挑
    a_test = pt.mean().idxmax()              # 作弊：按测试集均值挑（仅作上界参考）
    return {
        f"{var}_best_single_alpha_by_val": float(a_val),
        f"{var}_best_single_alpha_mean_gain": float(pt[a_val].mean()),
        f"{var}_best_single_alpha_n_blocks": int(len(common)),
        f"{var}_best_single_alpha_n_alphas": int(pv.shape[1]),
        f"{var}_best_single_alpha_test_selected": float(a_test),
        f"{var}_best_single_alpha_mean_gain_test_selected": float(pt[a_test].mean()),
    }


def fredf_verdict(s: dict[str, Any]) -> list[str]:
    out: list[str] = []
    if not s or s.get("n_cells", 0) == 0:
        return ["FreDF 公平性组还没有数据。"]
    g05, gt = s.get("mean_gain_fixed_alpha05"), s.get("mean_gain_tuned_alpha")
    npair = int(s.get("n_blocks_paired_alpha05", 0) or 0)
    if (g05 is not None and gt is not None and np.isfinite(g05) and np.isfinite(gt)
            and npair > 0):
        # 2026-09-14 修复项 #2：两个均值现在来自**同一批配对块**，差值才是「调参挽回了多少」。
        # 配对块数必须一起说出来，否则读者无法判断这个差值有多少支撑。
        out.append(
            f"调参差距（在 {npair} 个同时有两种口径的配对块上）：固定 α=0.5 平均增益 "
            f"{g05:+.2f}%，按验证集逐格调 α 后 {gt:+.2f}%，差距 {gt-g05:+.2f} 个百分点。"
            + ("这说明 FreDF 的收益主要来自调参，『即插即用』的说法需要限定条件——"
               "这是论文最有力的一句话。" if gt - g05 > 1.0 else
               "调参挽回有限，说明固定 α 的批评还不足以成立，需要谨慎表述。")
        )
    ps, pp = (s.get("sqrth_best_single_alpha_mean_gain"),
              s.get("plain_best_single_alpha_mean_gain"))
    if ps is not None and pp is not None and np.isfinite(ps) and np.isfinite(pp):
        # 2026-09-14 修复项 #1：这两个数现在是「按**验证集**挑出的单一 α」在测试集上的
        # 平均增益（旧实现按测试集挑 α，这条方法论结论是建立在作弊口径上的）。
        na, nb = (s.get("plain_best_single_alpha_by_val"), s.get("sqrth_best_single_alpha_by_val"))
        pick = ""
        if na is not None and nb is not None:
            pick = f"（选中的 α：原始 {na:g}、归一化后 {nb:g}）"
        out.append(
            f"√H 归一化：**按验证集**挑出的最优单一 α 的测试平均增益 原始 {pp:+.2f}% "
            f"vs 归一化后 {ps:+.2f}%{pick}。"
            + ("归一化让『一个 α 通吃所有 horizon』变得可行，可以作为一个独立的小方法贡献。"
               if ps - pp > 0.5 else
               "归一化没有带来可观改善，应把它降级为消融实验里的一行，不要当卖点。")
        )
    a, b = (s.get("plain_mean_distinct_best_alpha_across_horizons"),
            s.get("sqrth_mean_distinct_best_alpha_across_horizons"))
    if a and np.isfinite(a):
        out.append(f"最优 α 的 horizon 依赖：原始尺度下同一 (骨干,数据集) across horizon 平均有 "
                   f"{a:.2f} 个不同的最优 α" + (f"，√H 归一化后降到 {b:.2f}" if b and np.isfinite(b) else "")
                   + "（1.0 = 完全不依赖 horizon）。")
    rs = s.get("repeat_max_std_mse")
    if rs is not None and np.isfinite(rs):
        out.append(f"确定性自检：同一格在 p2/p3 各跑一次，MSE 最大标准差 {rs:.2e}"
                   + ("（≈0，复现性正常）。" if rs < 1e-6 else "（**偏大，需要排查随机性来源**）。"))
    return out


# --------------------------------------------------------------------------- #
# 报告拼装
# --------------------------------------------------------------------------- #
def _step_summary(ok: bool, path: Path, log_key: str) -> dict[str, Any]:
    """读一个子步骤的 summary json——**只在这一步本轮成功时**才读。

    2026-09-14 code review 修复项 #3：旧实现把 ``sh()`` 返回的 ``ok`` 丢掉，只用
    ``path.exists()`` 决定是否读 json。而 ``src.selector_window`` 只在成功时覆盖写
    json，所以这一步失败（或触发 7200s 超时）时，``artifacts/p2/`` 里仍留着**上一轮**
    的文件，于是 §0/§2 的 headline 会把过期数字当成本轮结论呈现，失败信息只出现在报告
    末尾折叠的执行日志里。对「作者直接抄数字进论文」的唯一交付物来说，这是静默的数据
    串轮，比报告缺一节严重得多。

    因此：失败就返回显式错误标记，绝不读旧 json；若磁盘上确有旧文件，把它的路径与
    **mtime** 一起记下来（渲染时会提示「数据来自上一轮，本报告未采用」），
    让「要不要引用上一轮」变成读者的显式选择而不是默认行为。
    """
    if ok and path.exists():
        return json.loads(path.read_text("utf-8"))
    why = (f"子进程失败或超时（见执行日志 {log_key}）" if not ok
           else f"子进程成功但没有写出 {path.name}")
    err: dict[str, Any] = {"error": f"本轮没有产出：{why}", "failed_step": log_key}
    if path.exists():
        err["stale_json"] = str(path)
        err["stale_json_mtime"] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime))
        err["error"] += (f"；磁盘上仍有上一轮的 {path.name}"
                         f"（写于 {err['stale_json_mtime']}），本报告**未采用**它")
    return err


def _fig_ref_path(p: Path) -> str:
    """把图片的绝对路径转成**相对 `results/` 的引用路径**。

    报告落在 `results/weekend_report.md`，图片写在 `artifacts/paper/` 下；如果 md 里嵌
    绝对路径，这份报告一旦离开这台机器（rsync 到本地、贴进论文仓库）图就全断了。
    抽成函数是为了能对「路径是相对的」直接做行为断言——原来这条约束只由测试里
    `assert 'os.path.relpath' in src` 这种源码字符串断言把着，改个写法就误报。
    """
    try:
        return os.path.relpath(p, ROOT / "results")
    except ValueError:                       # 跨盘符（Windows）时退回绝对路径
        return str(p)


def _pct_or_dash(v: Any, spec: str = ".1f") -> str:
    """把一个可能缺失的百分数格式化成 `12.3%`，缺失时给 `—`。

    抽出来是因为原来这条逻辑是内联的 `x != x` NaN 判断加三元表达式，一行 187 字符，
    既没法单测也很容易在改格式时把 NaN 分支写漏，让裸 `nan` 冒充成一个数值出现在
    正文里。
    """
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "—"
    if f != f:                               # NaN
        return "—"
    return format(f, spec) + "%"


def build_report(args: argparse.Namespace) -> Path:
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    logs: dict[str, str] = {}
    data: dict[str, Any] = {"generated_at": started}

    # ---------- 1. 覆盖度 ----------
    cov = {name: coverage(p) for name, p in
           [("phase1", args.config_p1), ("phase2", args.config_p2), ("fredf", args.config_p3)]}
    data["coverage"] = cov

    # ---------- 2. phase-1 全量统计 ----------
    p1 = MatrixConfig.load(args.config_p1)
    ok, out = sh([PY, "-m", "src.scheduler", "--config", args.config_p1, "--aggregate"])
    logs["p1_aggregate"] = tail(out, 5)
    ok, out = sh([PY, "-m", "src.stats", "--results", str(p1.path("results_csv")),
                  "--out", str(p1.path("artifacts_dir") / "stats")])
    logs["p1_stats"] = tail(out, 60)
    ok, out = sh([PY, "-m", "src.horizon", "--runs-dir", str(p1.path("runs_dir")),
                  "--out", str(p1.path("artifacts_dir") / "horizon")], timeout=1800)
    logs["p1_horizon"] = tail(out, 30)
    ok, out = sh([PY, "-m", "src.selector",
                  "--results", str(p1.path("results_csv")),
                  "--features", str(p1.path("features_csv")),
                  "--out", str(p1.path("artifacts_dir") / "selector")], timeout=1800)
    logs["p1_selector"] = tail(out, 30)

    # ---------- 3. 逐窗口门控（本周末的主结论）----------
    p2 = MatrixConfig.load(args.config_p2)
    ok, out = sh([PY, "-m", "src.scheduler", "--config", args.config_p2, "--aggregate"])
    logs["p2_aggregate"] = tail(out, 5)
    feat = p2.path("artifacts_dir") / "window_features.parquet"
    if not feat.exists():
        ok, out = sh([PY, "-m", "src.features_window", "--config", args.config_p2], timeout=14400)
        logs["window_features"] = tail(out, 20)
    p3cfg = MatrixConfig.load(args.config_p3)
    p2_dir, p3_dir = str(p2.path("winerr_dir")), str(p3cfg.path("winerr_dir"))

    # 三次门控评估，各自回答一个不同的问题：
    #   1) reg  + 仅 p2 臂池（none/revin/san_lite/fredf）—— 论文主表。四个骨干共用
    #      同一个 4 臂池，数字才可比；混进 p3 的 8 个 FreDF 变体会让 TimesNet（p3 里
    #      没有）和其他骨干的臂池不一样，headline 数字失去可比性。
    #   2) clf + 仅 p2 臂池 —— 区分「真没信号」与「reg 的『预测增益全 ≤0 就回退 none』
    #      把信号压没了」。只跑一种的话，周一看到 switch_rate=0 分不清是哪种。
    #   3) reg + p2 ∪ p3 臂池 —— 臂池从 4 个扩到 11 个，看 oracle 上界能涨多少：
    #      这是「插件选择这个问题值不值得做」的直接证据（3 个非 TimesNet 骨干）。
    runs: list[tuple[str, str, str, list[str]]] = [
        ("gating", "reg", "", [p2_dir]),
        ("gating_clf", "clf", "clf", [p2_dir]),
    ]
    if Path(p3_dir).exists() and any(Path(p3_dir).glob("*.npy")):
        runs.append(("gating_ext", "reg", "ext", [p2_dir, p3_dir]))
    for key, mode, tag, dirs in runs:
        # lodo（跨数据集迁移）要在 ~80k 行的池化数据上逐折训练，是整个报告最慢的一步
        # （实测 >25 min）。静态特征门控这条线已被 oracle 审计降级为负面对照，
        # 因此只在主 run 上跑一次 lodo，clf / 扩展臂池只跑 inpool，保住周一的截止时间。
        proto = "both" if key == "gating" else "inpool"
        cmd = [PY, "-m", "src.selector_window", "--config", args.config_p2,
               "--winerr", *dirs, "--protocol", proto, "--mode", mode]
        if tag:
            cmd += ["--tag", tag]
        if key == "gating":
            # 证伪对照很便宜（每组 6 次拟合），任何时候都值得跑：没有它，
            # 「门控赢了 best_fixed」这句话可以被一句「换臂本身就有用」推翻。
            cmd += ["--controls"]
            if not args.fast:      # 消融每组多 5 次拟合，只在最终报告里跑
                cmd += ["--ablate"]
        if key == "gating":
            # oracle 审计必须在 headline 之前拿到：它决定"门控赢不了"该怎么解释
            cmd += ["--oracle-audit"]
        if key in ("gating", "gating_ext"):
            # 在线延迟反馈选择器是纯 numpy、秒级，且是静态特征门控失败时的备用主线：
            # 挂在 reg 的两个 run 上就够（它与 --mode 无关，clf run 上跑只会重复算一遍）。
            cmd += ["--online"]
        ok, out = sh(cmd, timeout=7200)
        logs[f"selector_window_{key}"] = tail(out, 80)
        sfx = f"_{tag}" if tag else ""
        sw_path = p2.path("artifacts_dir") / f"selector_window_summary{sfx}.json"
        data[key] = _step_summary(ok, sw_path, log_key=f"selector_window_{key}")

    # ---------- 3.5 成对 seed 审计（headroom 是信号还是训练噪声）----------
    # 单独一个 run：--protocol none 表示不训练任何门控器，只吃已落盘的逐窗口误差，
    # 秒级完成，不会威胁周一的截止时间。前提是 seed 复制组已经产出（p2seed）。
    # 整段包在 try 里：周一的这份报告是**唯一**的交付物，一个可选章节没有资格弄挂它。
    try:
        # 注意用 ROOT 拼绝对路径：报告可能从任意 CWD 被调用，相对路径会让这一节静默跳过
        seed_cfg = ROOT / "configs" / "matrix_p2seed.yaml"
        seed_dir_p = (Path(MatrixConfig.load(str(seed_cfg)).path("winerr_dir"))
                      if seed_cfg.exists() else ROOT / "results" / "p2seed" / "winerr")
        seed_dir = str(seed_dir_p)
        if seed_dir_p.exists() and any(seed_dir_p.glob("*.npy")):
            ok, out = sh([PY, "-m", "src.selector_window", "--config", args.config_p2,
                          "--winerr", p2_dir, seed_dir, "--protocol", "none",
                          "--seed-audit", "--tag", "seed"], timeout=1800)
            logs["selector_window_seed_audit"] = tail(out, 60)
            sp = p2.path("artifacts_dir") / "selector_window_summary_seed.json"
            # 同修复项 #3：ok 必须纳入判断，否则失败时会把上一轮的 seed 审计当本轮结论
            got = _step_summary(ok, sp, log_key="selector_window_seed_audit")
            data["seed_audit"] = (got.get("seed_audit", {}) if "error" not in got
                                  else {k: v for k, v in got.items() if k != "failed_step"})
        else:
            logs["selector_window_seed_audit"] = f"跳过：{seed_dir} 还没有逐窗口误差"
    except Exception as e:                                   # noqa: BLE001
        logs["selector_window_seed_audit"] = f"seed 审计异常（不影响报告其余部分）：{e!r}"

    # ---------- 3.6 论文主图（半衰期 vs H、分层衰减曲线）----------
    # 纯 numpy/matplotlib，只读 oracle 审计的 CSV，不碰 GPU。同样包在 try 里：
    # 画图失败没有资格弄挂整份报告。
    try:
        aud_csv = p2.path("artifacts_dir") / "selector_window_oracle_audit.csv"
        fig_dir = p1.path("artifacts_dir") / "paper"
        if aud_csv.exists():
            ok, out = sh([PY, str(ROOT / "tools" / "make_audit_figs.py"),
                          "--audit-csv", str(aud_csv), "--out-dir", str(fig_dir)], timeout=600)
            logs["audit_figs"] = tail(out, 30)
            # 报告落在 results/ 下，图片必须写相对路径，否则这份 md 一离开这台机器
            # （比如 rsync 到本地看）图就全断了。
            figs = {}
            for n in ("fig_halflife_vs_horizon", "fig_decay_curves"):
                p = fig_dir / f"{n}.png"
                if p.exists():
                    figs[n] = _fig_ref_path(p)
            data["audit_figs"] = figs
        else:
            logs["audit_figs"] = f"跳过：{aud_csv} 不存在（需要先跑 --oracle-audit）"
    except Exception as e:                                   # noqa: BLE001
        logs["audit_figs"] = f"画图异常（不影响报告其余部分）：{e!r}"

    # ---------- 4. FreDF 公平性 ----------
    ok, out = sh([PY, "-m", "src.scheduler", "--config", args.config_p3, "--aggregate"])
    logs["p3_aggregate"] = tail(out, 5)
    fre: dict[str, Any] = {}
    p3res, p1res = p3cfg.path("results_csv"), p1.path("results_csv")
    if p3res.exists():
        try:
            r3 = pd.read_csv(p3res)
            # α=0.5 参考优先用 p2（同为单 seed 2021），缺失才退回 phase-1
            p2res = p2.path("results_csv")
            ref_frames = [pd.read_csv(x) for x in (p2res, p1res) if x.exists()]
            ref = pd.concat(ref_frames, ignore_index=True) if ref_frames else None
            if ref is not None and "plugin" in ref:
                ref = ref.drop_duplicates(
                    subset=[c for c in ("backbone", "dataset", "pred_len", "plugin", "seed")
                            if c in ref], keep="first")
            fre = fredf_analysis(r3, ref)
            if isinstance(fre.get("detail"), pd.DataFrame):
                dpath = p3cfg.path("artifacts_dir") / "fredf_alpha.csv"
                dpath.parent.mkdir(parents=True, exist_ok=True)
                fre["detail"].to_csv(dpath, index=False)
                fre["detail_path"] = str(dpath)
                fre["detail"] = None
        except Exception as exc:                                # noqa: BLE001
            fre = {"error": f"{type(exc).__name__}: {exc}"}
    data["fredf"] = {k: v for k, v in fre.items() if k != "detail"}

    # ---------- 5. 论文主表 ----------
    if not args.fast:
        ok, out = sh([PY, "-m", "src.report", "--results", str(p1.path("results_csv")),
                      "--features", str(p1.path("features_csv")),
                      "--selector-dir", str(p1.path("artifacts_dir") / "selector"),
                      "--stats-dir", str(p1.path("artifacts_dir") / "stats"),
                      "--horizon-dir", str(p1.path("artifacts_dir") / "horizon"),
                      "--out", str(p1.path("artifacts_dir") / "paper")], timeout=1800)
        logs["p1_paper"] = tail(out, 20)

    # ---------- 6. 写报告 ----------
    md = _render_markdown(data, logs, started)
    out_md = ROOT / "results" / "weekend_report.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    (ROOT / "results" / "weekend_report.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"[report] -> {out_md}")
    return out_md


def _render_markdown(data: dict[str, Any], logs: dict[str, str], started: str) -> str:
    L: list[str] = []
    A = L.append
    A(f"# PlugGate 周末无人值守报告\n\n生成时间：{started}\n")

    # --- 一句话结论 ---
    A("## 0. 先看这里：三个结论\n")
    gat = data.get("gating") or {}
    # 2026-09-14 code review 修复项 #3 的渲染端：某一步失败时，build_report 会写下
    # {"error": ...} 而不是去读上一轮的 json。这里必须把它顶到 headline 最前面，
    # 让读者一眼看到「这一节本轮没有产出」，绝不能只留在末尾折叠的执行日志里。
    _FAILED_LABEL = {"gating": "逐窗口门控主 run（§2）",
                     "gating_clf": "clf 头门控（§2 对比表）",
                     "gating_ext": "扩展臂池门控（§2 扩大臂池）",
                     "seed_audit": "成对 seed 审计（§2.0b）"}
    step_errors = [(lab, (data.get(k) or {}).get("error"))
                   for k, lab in _FAILED_LABEL.items() if (data.get(k) or {}).get("error")]
    if step_errors:
        A("> ⛔ **本轮有步骤失败，下列章节没有本轮产出，其数字不得引用**：")
        for lab, msg in step_errors:
            A(f">   - {lab}：{msg}")
        A("")
    inp = (gat.get("inpool_learnable") or gat.get("inpool")) or {}
    scope = "（仅训练窗口 ≥1000 的组）" if gat.get("inpool_learnable") else ""
    aud0 = gat.get("oracle_audit") or {}
    if aud0.get("verdict"):
        A(f"0. **先读这一条（oracle 上界审计）**：{aud0['verdict']}")
    if (gat.get("audit_strata") or {}).get("verdict"):
        A(f"   - 分层普遍性（是否某个骨干/数据集/短 horizon 例外）：{gat['audit_strata']['verdict']}")
    if (data.get("seed_audit") or {}).get("verdict"):
        A(f"   - 成对 seed 复制（headroom 是信号还是训练噪声）：{data['seed_audit']['verdict']}")
    A(f"1. **逐窗口门控**{scope}：{inp.get('verdict', '无数据（phase-2 逐窗口误差没跑出来）')}")
    if (gat.get("inpool_controls") or {}).get("verdict"):
        A(f"   - 证伪对照：{gat['inpool_controls']['verdict']}")
    if (gat.get("online") or {}).get("verdict"):
        A(f"   - 延迟误差反馈（不用静态特征，改用过去窗口真实误差）：{gat['online']['verdict']}")
    fs = (data.get("fredf") or {}).get("summary") or {}
    for i, line in enumerate(fredf_verdict(fs), start=2):
        A(f"{i}. **FreDF 公平性**：{line}" if i == 2 else f"   - {line}")
    A("")

    # --- 覆盖度 ---
    A("## 1. 这个周末实际跑了什么\n")
    A("| 组 | 完成/总数 | 完成率 | 失败 | 剩余预估 | 逐窗口误差文件(test/val) |")
    A("|---|---|---|---|---|---|")
    for name, c in (data.get("coverage") or {}).items():
        if "error" in c:
            A(f"| {name} | 读取失败 | | | | {c['error']} |")
            continue
        we = f"{c.get('n_winerr_test', 0)} / {c.get('n_winerr_val', 0)}"
        A(f"| {name} | {c['n_done']}/{c['n_total']} | {c['pct_done']}% | {c['n_failed']} | "
          f"{c['todo_gpu_hours']} GPU·h | {we} |")
    A("")
    for name, c in (data.get("coverage") or {}).items():
        if c.get("n_failed"):
            A(f"- {name} 失败样例：{', '.join(c.get('failed_examples', []))}")
        if c.get("todo_by_backbone_horizon"):
            A(f"- {name} 未跑完的格子分布（骨干 × horizon）：`{c['todo_by_backbone_horizon']}`")
    A("")

    # --- 门控 ---
    A("## 2. 逐窗口门控（核心实验）\n")
    if gat.get("error"):
        # 修复项 #3：本轮失败就把这件事写在本节最前面，并且下面不会有任何本轮数字。
        A(f"> ⛔ **本节本轮没有产出**：{gat['error']}\n")
        if gat.get("stale_json_mtime"):
            A(f"> 若要引用上一轮结果，必须自己去读 `{gat.get('stale_json')}`"
              f"（写于 {gat['stale_json_mtime']}）并在论文里标注「数据来自上一轮」。\n")
    aud = gat.get("oracle_audit") or {}
    if aud.get("n_groups"):
        A("### 2.0 先审计 oracle 上界这个统计量本身\n")
        A("**命题**：per-window oracle `mean_t min_k e[k,t]` 只依赖每个窗口内误差的多重集，"
          "与「这些误差属于哪个臂」无关。对每个窗口独立地随机重贴臂标签，oracle 一分不变，"
          "但「哪个窗口该用哪个臂」的可预测性会被彻底摧毁。"
          f"本报告在 {aud['n_groups']} 组上逐位验证了这一不变性："
          f"`{aud.get('oracle_invariant_under_relabel_all')}`。\n")
        A(f"因此「oracle 比 best_fixed 高 {aud.get('mean_oracle_gain_real_pct', float('nan')):.2f}%」"
          "**不能**用来论证插件选择可学——只要各臂误差在窗口尺度上有离散度，这个数就必然很大。"
          "真正决定可学性的是最优臂身份 argmin_k e[k,t] 的**滞后可预测性**，"
          "而且必须在合法延迟（≥H 个窗口）下看。\n")
        A("| 滞后 | 一致率 | 超出 chance | z>2 的组 |")
        A("|---|---|---|---|")
        ch = aud.get("mean_chance_persist", float("nan"))
        for name, label in (("lag1", "1 个窗口"), ("lagH_4", "H/4"), ("lagH_2", "H/2"),
                            ("lagH", "**H（唯一合法设定）**"), ("lag2H", "2H")):
            A(f"| {label} | {aud.get(f'mean_persist_{name}', float('nan')):.3f} | "
              f"{aud.get(f'mean_excess_persist_{name}', float('nan')):+.3f} | "
              f"{aud.get(f'groups_z_gt2_{name}', 0)}/{aud['n_groups']} |")
        A(f"\nchance = Σ p_k² = {ch:.3f}（逐窗口重贴标签后的实测一致率 "
          f"{aud.get('mean_persist_lag1_relabelled', float('nan')):.3f}，与 chance 吻合）。\n")
        curve = aud.get("excess_decay_curve") or {}
        if curve:
            A("**衰减曲线（超出 chance 的一致率 vs 绝对滞后，单位 = 窗口数）**\n")
            A("| 滞后 | " + " | ".join(curve.keys()) + " |")
            A("|---" * (len(curve) + 1) + "|")
            A("| 超出量 | " + " | ".join(f"{v:+.3f}" for v in curve.values()) + " |")
            A("")
        A(f"- **相关半衰期：中位数 {aud.get('median_persist_half_life_windows', float('nan')):.1f} 个窗口**"
          f"（均值 {aud.get('mean_persist_half_life_windows', float('nan')):.1f}），"
          f"约为预测视界 H 的 {aud.get('mean_half_life_over_H', float('nan')):.1%}")
        A(f"- 最常获胜的臂平均占 {aud.get('mean_win_rate_top', float('nan')):.1%} 的窗口，"
          f"胜率分布的归一化熵 {aud.get('mean_win_rate_entropy', float('nan')):.2f}")
        A(f"- 结论：{aud.get('verdict', '')}\n")

    st = gat.get("audit_strata") or {}
    if st.get("n_strata"):
        A("### 2.0a 这个结论是平均出来的，还是每一层都成立？（分层普遍性）\n")
        # 2026-09-14 code review 修复项 #4：这里原来写死「127 个决策组」，而上方 §2.0
        # 同一份审计用的是实测值 aud['n_groups']。报告本身声明要在不同覆盖度下都能生成，
        # 组数一变，同一份 md 里就会出现两个自相矛盾的组数（作者会照抄进论文）。
        # 因此一律从审计 summary 现读；实测值缺失时宁可不写数字，也不许写死。
        n_aud = aud.get("n_groups") or (gat.get("inpool") or {}).get("n_groups")
        scope_txt = f"{n_aud} 个决策组" if n_aud else "全部决策组"
        A(f"上面的数字是 {scope_txt}的平均。审稿人一定会问：会不会某个骨干 / 某个数据集 / "
          "某个短 horizon 上其实**有**可用信号，只是被平均抹掉了？"
          "这里把同一份审计按 backbone、dataset、pred_len **三种切法**各自重算一遍。\n")
        A("| 判据 | 结果 |")
        A("|---|---|")
        A(f"| 可用层数（去掉样本不足的层） | {st['strata_total_used']} / {st['n_strata']}"
          f"（欠功效 {st.get('n_strata_underpowered', 0)} 层） |")
        A(f"| 半衰期 < 25% H 的层 | **{st['strata_with_half_life_over_H_below_25pct']}"
          f"/{st['strata_total_used']}** |")
        A(f"| 延迟 H 下超出 chance > 0.05 的层 | **{st['strata_with_excess_lagH_above_005']}"
          f"/{st['strata_total_used']}** |")
        A(f"| 最像「有信号」的层 | `{st['max_excess_persist_lagH_stratum']}`，"
          f"超出量仅 {st['max_excess_persist_lagH']:+.3f} |")
        A(f"| 相关长度相对 H 最长的层 | `{st['max_median_half_life_over_H_stratum']}`，"
          f"{st['max_median_half_life_over_H']:.1%} |")
        lo, hi = st["median_half_life_windows_range"]
        A(f"| 半衰期中位数的层间范围 | {lo:.1f} ~ {hi:.1f} 个窗口 |")
        A("")
        if st.get("horizon_hl_spearman") is not None:
            A(f"**半衰期的 horizon 依赖**：绝对半衰期随 H 单调性 Spearman "
              f"{st['horizon_hl_spearman']:+.2f}（H={st['H_min']:.0f} 时 {st['hl_at_min_H']:.1f} 个窗口，"
              f"H={st['H_max']:.0f} 时 {st['hl_at_max_H']:.1f} 个窗口），"
              f"而**半衰期/H** 随 H 的 Spearman 是 {st['horizon_ratio_spearman']:+.2f}。"
              "也就是说：相关长度确实会随 H 略微变长，但**远远追不上 H 本身的增长**——"
              "H 越长，插件门控越不可能做。这正是「反馈延迟约束」这条主线的定量形式。\n")
        figs = data.get("audit_figs") or {}
        if figs.get("fig_halflife_vs_horizon"):
            A(f"![半衰期 vs 预测视界]({figs['fig_halflife_vs_horizon']})\n")
        if figs.get("fig_decay_curves"):
            A(f"![分层衰减曲线]({figs['fig_decay_curves']})\n")
        A(f"- 结论：{st.get('verdict', '')}\n")

    sa = data.get("seed_audit") or {}
    if sa.get("n_pairs"):
        A("### 2.0b 这个 headroom 是真实结构，还是训练随机性？（成对 seed 复制）\n")
        A("上面的「重贴标签不变性」是**构造性**论证，怀疑者可以说「现实里标签不是随机的」。"
          "这里给出完全经验性的第二条腿：把同一个格子**只换训练 seed 重跑一遍**，"
          "再问逐窗口最优臂身份还剩多少。\n")
        A("关键指标是 **跨 seed oracle**：用 seed A 的逐窗口 argmin 去选臂、在 seed B 的误差上结算。"
          "它仍然用了未来信息，但只保留了跨 seed **可复现**的那部分臂身份，"
          "因此是任何「靠可复现结构」的选择器（含用了完美特征的静态门控）的上界。\n")
        A("| 指标 | 数值 |")
        A("|---|---|")
        A(f"| 成对格子数 / seed | {sa['n_pairs']} 对 / {sa.get('seeds')} |")
        A(f"| 同 seed oracle（相对测试段最佳固定臂） | {sa.get('mean_self_oracle_gain_pct', float('nan')):+.2f}% |")
        A(f"| **跨 seed oracle（可复现上界）** | **{sa.get('mean_xseed_oracle_gain_pct', float('nan')):+.2f}%** |")
        A(f"| headroom 中可复现的比例 | {sa.get('reproducible_frac_of_headroom', float('nan')):.1%} |")
        A(f"| headroom 中属于训练噪声的比例 | {sa.get('noise_frac_of_headroom', float('nan')):.1%} |")
        A(f"| 最优臂身份的跨 seed 一致率 | {sa.get('mean_argmin_agree', float('nan')):.3f}"
          f"（独立重抽 chance={sa.get('mean_chance_agree', float('nan')):.3f}） |")
        A(f"| 一致率 z>2 的对数 | {sa.get('pairs_z_gt2', 0)}/{sa['n_pairs']} |")
        A(f"| 窗口级「该赚多少」的跨 seed 相关 | {sa.get('mean_adv_corr_across_seeds', float('nan')):.3f} |")
        A(f"| ⚠️ 同一个臂逐窗口误差的跨 seed 相关 | {sa.get('mean_same_arm_err_corr', float('nan')):.4f} |")
        A(f"| ⚠️ 块级 MSE 随 seed 的波动 | {sa.get('mean_seed_mse_gap_pct', float('nan')):.2f}% |")
        A(f"| 最佳固定臂换 seed 后不变的比例 | {sa.get('best_fixed_arm_stable_pct', float('nan')):.0f}% |")
        A("")
        A("> 读这张表**必须**先看最后两行加 ⚠️ 的量：如果换 seed 之后同一个臂的逐窗口误差"
          "几乎一模一样（相关≈1、块级 MSE 几乎不动），说明该骨干在两个 seed 上收敛到了同一个解，"
          "此时「可复现」是平凡的，本审计对它没有区分力（DLinear 这类近凸模型尤其如此）。"
          "只有当同臂误差本身有可观 seed 波动、而 argmin 身份**仍然**一致时，"
          "才能断言 headroom 是真实结构。\n")
        A(f"- 结论：{sa.get('verdict', '')}\n")
    if sa.get("error"):
        A(f"### 2.0b 成对 seed 复制\n\n> ⛔ **本节本轮没有产出**：{sa['error']}\n")
    if not gat or (set(gat) <= {"error", "failed_step", "stale_json", "stale_json_mtime"}):
        A("没有产出。可能原因：phase-2 的 winerr 还没生成，或窗口特征表缺失。\n")
    # 这些 protocol 不是「门控器」，它们的 summary 里没有 gate_vs_best_fixed 这一族字段，
    # 各自在后面有专门的一节（§2.0 审计 / 证伪对照 / 特征组消融 / 在线选择 / 延迟扫描）。
    # 早期版本对它们套用同一套门控模板，于是报告里出现了整片 `+nan%`，把真正的结论淹掉了。
    DEDICATED = {"inpool_controls": "见「证伪对照」一节",
                 "inpool_ablation": "见「特征组消融」一节",
                 "oracle_audit": "见 §2.0",
                 "audit_strata": "见 §2.0a",
                 "seed_audit": "见 §2.0b",
                 "online": "见「换一种信号：延迟误差反馈的在线选择器」一节",
                 "online_sweep": "见「延迟 × 反馈窗口扫描」一节"}
    skipped: list[str] = []
    for proto, s in gat.items():
        if not isinstance(s, dict) or not s.get("n_groups"):
            continue
        if proto in DEDICATED:
            skipped.append(f"`{proto}`（{s['n_groups']} 组，{DEDICATED[proto]}）")
            continue
        # lodo 这类协议不产出全部指标（例如折内没有well-defined的「最佳固定臂」），
        # 缺失时打 `—` 并注明，绝不让裸 `nan` 出现在正文里冒充数值。

        def q(key: str, spec: str = "+.2f", suffix: str = "%") -> str:
            v = s.get(key, float("nan"))
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return "—"
            if fv != fv:
                return "—"
            return f"{fv:{spec}}{suffix}"

        A(f"### 协议 `{proto}`（{s['n_groups']} 个决策组）\n")
        A(f"- 结论：{s.get('verdict', '')}")
        A(f"- 门控 vs 不挂插件：平均 {q('mean_gain_gate_vs_none')}")
        A(f"- 最佳固定插件 vs 不挂插件：平均 {q('mean_gain_bestfixed_vs_none')}")
        A(f"- **门控 vs 最佳固定插件：平均 {q('mean_gain_gate_vs_best_fixed')}**"
          f"（{s.get('groups_gate_beats_best_fixed_pct', 0):.0f}% 的组为正，"
          f"{s.get('groups_significant_p05', 0)} 组 p<0.05）")
        A(f"- oracle 上界 vs 不挂插件：{q('mean_gain_oracle_vs_none')}"
          f"，门控吃到了其中 {q('mean_oracle_realized_pct', '.1f')}")
        A(f"- 逐窗口准确率 {q('mean_gate_acc', '.3f', '')} vs 多数类基线 "
          f"{q('mean_majority_acc', '.3f', '')}"
          f"（{q('gate_beats_majority_pct', '.0f')} 的组胜过基线）")
        A(f"- 平均挂插件比例 {q('mean_switch_rate', '.2f', '')}；"
          f"跨组符号检验 p={q('wilcoxon_p_across_groups', '.3g', '')}")
        A("")
    if skipped:
        A("> 另有 " + "、".join(skipped) + "，它们不是门控器、不适用上面这套指标，"
          "各自在专门的一节里报告。\n")
    gc = data.get("gating_clf") or {}
    if gc.get("error"):
        # 修复项 #3：失败时不渲染一张空表（更不读旧 json），只说清本轮没产出
        A(f"### 两种门控头的对比\n\n> ⛔ **本轮没有产出**：{gc['error']}\n")
    elif gc:
        A("### 两种门控头的对比（reg = 预测增益后设阈值；clf = 直接多分类选臂）\n")
        A("| 协议 | 头 | 组数 | gate−none | gate−best_fixed | oracle 上界 | 吃到 oracle | 挂插件比例 |")
        A("|---|---|---|---|---|---|---|---|")
        for proto in ("inpool", "lodo"):
            for head, src in (("reg", gat), ("clf", gc)):
                s = (src or {}).get(proto) or {}
                if not s.get("n_groups"):
                    continue
                A(f"| {proto} | {head} | {s['n_groups']} | "
                  f"{s.get('mean_gain_gate_vs_none', float('nan')):+.2f}% | "
                  f"{s.get('mean_gain_gate_vs_best_fixed', float('nan')):+.2f}% | "
                  f"{s.get('mean_gain_oracle_vs_none', float('nan')):+.2f}% | "
                  f"{_pct_or_dash(s.get('mean_oracle_realized_pct'))} | "
                  f"{s.get('mean_switch_rate', float('nan')):.2f} |")
        A("")
        A(f"clf 头的 inpool 结论：{((gc.get('inpool') or {}).get('verdict', '无'))}\n")
        A("> clf 头与扩展臂池只跑 inpool：lodo 需要在 ~80k 行池化数据上逐折训练（>25 min），"
          "而静态特征门控这条线已被 §2.0 的审计降级为负面对照，不值得为它牺牲报告的截止时间。\n")
    lrn = (gat or {}).get("inpool_learnable") or {}
    if lrn.get("n_groups"):
        A(f"### 只看「训练窗口够多」的组（{lrn.get('filter', '')}，{lrn['n_groups']} 组）\n")
        # 修复项 #4：正文不写死实验规模。ILI/ETTm1 的验证窗口数随覆盖度与切分变化，
        # 写死会和 §2 的「按验证窗口数分层」表自相矛盾；具体数字看那张表即可。
        A("验证窗口极少的数据集（如 ILI）上门控学不到东西是数据量的必然结果，"
          "把它和验证窗口多两个数量级的数据集（如 ETTm1）平均在一起会把结论稀释成噪声"
          "（逐层的窗口数见「按验证窗口数分层」一节）。论文主张只能建立在这一层上。\n")
        A(f"- 结论：{lrn.get('verdict', '')}")
        A(f"- 门控 vs 最佳固定插件：{lrn.get('mean_gain_gate_vs_best_fixed', float('nan')):+.2f}%"
          f"（{lrn.get('groups_gate_beats_best_fixed_pct', 0):.0f}% 的组为正，"
          f"{lrn.get('groups_significant_p05', 0)} 组 p<0.05）")
        A(f"- oracle 上界 {lrn.get('mean_gain_oracle_vs_none', float('nan')):+.2f}%，"
          f"吃到 {lrn.get('mean_oracle_realized_pct', float('nan')):.1f}%\n")
    strat = (gat.get("inpool") or {}).get("by_val_windows") or {}
    if strat:
        A("### 按验证窗口数分层：是「学不到」还是「没得学」\n")
        A("| 验证窗口数 | 组数 | gate−best_fixed | gate−none | oracle 上界 | 逐窗口准确率 vs 多数类 | 赢过 best_fixed 的组 |")
        A("|---|---|---|---|---|---|---|")
        for label, v in strat.items():
            A(f"| {label} | {v['n_groups']} | {v['mean_gain_gate_vs_best_fixed']:+.2f}% | "
              f"{v['mean_gain_gate_vs_none']:+.2f}% | {v['mean_gain_oracle_vs_none']:+.2f}% | "
              f"{v['mean_gate_acc']:.3f} / {v['mean_majority_acc']:.3f} | "
              f"{v['pct_beats_best_fixed']:.0f}% |")
        A("")
    ctl = (gat or {}).get("inpool_controls") or {}
    if ctl:
        A("### 证伪对照：收益到底来自窗口级信息，还是来自「换臂」本身\n")
        A(f"- 结论：{ctl.get('verdict', '')}")
        A(f"- 同频随机换臂 vs 最佳固定臂：{ctl.get('mean_gain_random_vs_best_fixed', float('nan')):+.2f}%"
          "（换臂频率与各臂占比都与真门控一致，只打乱『哪个窗口选哪个臂』）")
        A(f"- 打乱特征训练的门控 vs 最佳固定臂：{ctl.get('mean_gain_shuffled_vs_best_fixed', float('nan')):+.2f}%"
          "（等价于同容量模型拟合噪声能拿到多少，即过拟合基线）")
        A(f"- **真门控净优势：vs random {ctl.get('mean_gate_minus_random', float('nan')):+.2f}%"
          f"（{ctl.get('groups_gate_beats_random_pct', float('nan')):.0f}% 的组为正）、"
          f"vs shuffled {ctl.get('mean_gate_minus_shuffled', float('nan')):+.2f}%"
          f"（{ctl.get('groups_gate_beats_shuffled_pct', float('nan')):.0f}%）**\n")
    abl = (gat or {}).get("inpool_ablation") or {}
    if abl:
        A("### 特征组消融（leave-one-group-out，负数 = 去掉后变差 = 这组有用）\n")
        A(f"完整特征下 gate−best_fixed = {abl.get('mean_gain_full', float('nan')):+.2f}%，"
          f"最关键的一组是 **{abl.get('most_important_group')}**。\n")
        A("| 去掉哪组特征 | 含义 | 增益变化 (pp) |")
        A("|---|---|---|")
        names = {"w_pc": "可预测性/熵", "w_ns": "非平稳性（分布漂移）", "w_fr": "频域集中度",
                 "w_sd": "趋势与自相关", "w_ch": "通道间相关"}
        for k, v in sorted((abl.get("mean_delta_when_dropped") or {}).items(),
                           key=lambda kv: kv[1]):
            A(f"| {k} | {names.get(k, '')} | {v:+.3f} |")
        A("")
    ge = data.get("gating_ext") or {}
    gei = ge.get("inpool") or {}
    if gei.get("n_groups"):
        gi = gat.get("inpool") or {}
        A("### 扩大臂池（p2 的 4 臂 → 并入 p3 的 8 个 FreDF 变体，共 11 臂）\n")
        A(f"- 决策组 {gei['n_groups']} 个（只含非 TimesNet 骨干，因为 p3 不跑 TimesNet）")
        A(f"- oracle 上界：4 臂 {gi.get('mean_gain_oracle_vs_none', float('nan')):+.2f}% "
          f"→ 11 臂 {gei.get('mean_gain_oracle_vs_none', float('nan')):+.2f}%"
          "（上界随臂池变大而抬升，说明『该挂哪个插件』确实是个有内容的决策）")
        A(f"- 门控 vs 最佳固定臂：{gei.get('mean_gain_gate_vs_best_fixed', float('nan')):+.2f}%"
          f"，吃到 oracle 的 {gei.get('mean_oracle_realized_pct', float('nan')):.1f}%")
        A(f"- 结论：{gei.get('verdict', '')}\n")

    onl = (gat or {}).get("online") or {}
    onl_ext = (ge or {}).get("online") or {}
    if onl.get("n_groups"):
        A("### 换一种信号：延迟误差反馈的在线选择器\n")
        A("静态复杂度特征要在 val 上学、在 test 上用，跨了一次分布漂移。但部署时其实有"
          "更强的信号可用——**过去窗口上每个臂真实的误差**：预测发出 H 步后真值就到了。"
          "这条基线不需要任何 GPU，用已经落盘的逐窗口误差就能算。\n")
        A(f"**延迟的严格性**：窗口 t 的目标是 [t+L, t+L+H)，所以窗口 t 决策时只能用到"
          f"t−H 及更早窗口的真值，延迟**恰好 H 个窗口**。报告里同时给出「不延迟」版本作为"
          f"作弊上界，用来量化延迟本身的代价。反馈滑动窗口 = {onl.get('feedback_window')} 个窗口。\n")
        A("| 臂池 | 组数 | online−none | online−best_fixed | 不延迟−best_fixed | 延迟代价 | "
          "oracle 上界 | 吃到 oracle | 赢过 best_fixed | p<0.05 |")
        A("|---|---|---|---|---|---|---|---|---|---|")
        for label, s in (("p2 4 臂", onl), ("p2∪p3 11 臂", onl_ext)):
            if not s.get("n_groups"):
                continue
            A(f"| {label} | {s['n_groups']} | "
              f"{s.get('mean_gain_online_vs_none', float('nan')):+.2f}% | "
              f"**{s.get('mean_gain_online_vs_best_fixed', float('nan')):+.2f}%** | "
              f"{s.get('mean_gain_nodelay_vs_best_fixed', float('nan')):+.2f}% | "
              f"{s.get('mean_delay_cost_pp', float('nan')):.2f} pp | "
              f"{s.get('mean_gain_oracle_vs_none', float('nan')):+.2f}% | "
              f"{s.get('mean_oracle_realized_pct', float('nan')):.1f}% | "
              f"{s.get('groups_online_beats_best_fixed_pct', float('nan')):.0f}% | "
              f"{s.get('groups_significant_p05', 0)}/{s['n_groups']} |")
        A("")
        adapt = onl.get("mean_gain_online_vs_best_fixed_test")
        if adapt is not None:
            A("**收益来源分解（reviewer 一定会问的那个对照）**：把「在 test 上选出的最优常量臂」"
              "（best_fixed_test，不做任何时变切换）也算出来，就能把在线选择的收益拆成两半：\n")
            A("| 成分 | 数值 | 含义 |")
            A("|---|---|---|")
            A(f"| best_fixed_test − best_fixed_val | "
              f"{onl.get('mean_gain_bestfixed_test_vs_val', float('nan')):+.2f}% | "
              f"val→test 选臂失配的代价（{onl.get('groups_arm_id_mismatch_pct', float('nan')):.0f}% 的组"
              "在 val 上选出的臂并不是 test 最优臂）|")
            pos_pct = onl.get('groups_online_beats_best_fixed_test_pct', float('nan'))
            A(f"| online − best_fixed_test | {adapt:+.2f}% | "
              f"扣掉失配后**真正的时变自适应收益**（{pos_pct:.0f}% 的组为正）|")
            A("")
            if adapt <= 0:
                A("> ⚠️ 时变成分 ≤0：这条线只能写成「在线反馈能纠正 val→test 的插件选择失配」，"
                  "**不能**写成「窗口级自适应」。后者会被一个 best_fixed_test 对照直接推翻。\n")

        gi_bf = (gat.get("inpool") or {}).get("mean_gain_gate_vs_best_fixed", float("nan"))
        A(f"**和静态特征门控直接对比**：静态门控 vs best_fixed = {gi_bf:+.2f}%，"
          f"延迟反馈 vs best_fixed = {onl.get('mean_gain_online_vs_best_fixed', float('nan')):+.2f}%。")
        A(f"- 结论：{onl.get('verdict', '')}\n")

    sw = (gat or {}).get("online_sweep") or {}
    if sw.get("n_groups"):
        A("#### 延迟 × 反馈窗口扫描：盈亏平衡延迟在哪里\n")
        A("延迟用 H 的倍数表示（H=96 和 H=720 的「延迟 H 个窗口」严重程度完全不同）。"
          "**ratio=0 是作弊上界，ratio=1 是唯一合法的部署设定。**"
          "表格数值 = 相对最佳固定臂的增益 %。\n")
        if sw.get("aligned_eval_window"):
            A("> 所有格子都在**同一段公共测试窗口**上评测（跳过最长延迟的暖机期）。"
              "不这样做的话，延迟越长就有越多窗口因为「反馈还没到」而退回 best_fixed，"
              "指标被稀释着拉回 0，会得出「延迟 2H 比 1H 更好」的错误结论——"
              "这个坑本项目已经踩过一次。\n")
        wins = sw.get("windows") or []

        def wlabel(w: int) -> str:
            return "反馈窗口=全历史" if int(w) >= 10 ** 8 else f"反馈窗口={w}"

        A("| 延迟 (×H) | " + " | ".join(wlabel(w) for w in wins) + " | 可部署 |")
        A("|---" * (len(wins) + 2) + "|")
        for r, row in (sw.get("gain_by_delay_and_window") or {}).items():
            cells = []
            for w in wins:
                v = row.get(str(w))
                cells.append("—" if v is None else f"{v:+.2f}%")
            ok = "✅" if float(r) >= 1.0 else "作弊"
            A(f"| {float(r):g} | " + " | ".join(cells) + f" | {ok} |")
        A("")
        A(f"- **盈亏平衡延迟 = {sw.get('breakeven_delay_ratio', 0.0):g}×H**"
          f"（最大的还能赢过最佳固定臂的延迟）")
        A(f"- 合法设定（1×H）下：{sw.get('feasible_gain_vs_best_fixed', float('nan')):+.2f}%；"
          f"作弊上界（0 延迟）：{sw.get('cheat_gain_vs_best_fixed', float('nan')):+.2f}%")
        bw = sw.get("best_feedback_window")
        A(f"- 最佳反馈窗口：{wlabel(bw) if bw else '—'}"
          f"（在 {[('全历史' if int(w) >= 10 ** 8 else w) for w in wins]} 中平均最优）")
        A("- 注意：headline 的在线数字用的是**预先固定**的反馈窗口 200，没有在 test 上调窗口；"
          "本表是敏感性分析，若要把某个更优窗口写成方法的一部分，必须改成在 val 上选窗口。")
        A(f"- 结论：{sw.get('verdict', '')}\n")

    # --- FreDF ---
    A("## 3. FreDF 公平性（α 扫描 + √H 归一化）\n")
    if not fs:
        A((data.get("fredf") or {}).get("note") or "没有产出。\n")
    else:
        A(f"覆盖 {fs.get('n_blocks', 0)} 个 (骨干×数据集×horizon) 块、{fs.get('n_cells', 0)} 格。\n")
        # 修复项 #2 的渲染端：调参差距那三行是**配对块**口径，块数必须写在表前，
        # 否则读者会以为它们和上面的「n_blocks 个块」是同一批。
        npair = int(fs.get("n_blocks_paired_alpha05", 0) or 0)
        if npair:
            A(f"「固定 α=0.5 / 调 α / 作弊上界」这三行与**调参差距**都只在 {npair} 个"
              "**同时具备 α=0.5 参考与调 α 结果**的配对块上计算（α=0.5 只来自 p2/phase-1，"
              "未跑完的格子系统性集中在最贵的角落，不配对就会引入方向性偏差）。"
              f"全部块上的调 α 均值另见 `mean_gain_tuned_alpha_all_blocks`。\n")
        elif fs.get("n_blocks"):
            A("⚠️ 没有任何块同时具备 α=0.5 参考与调 α 结果，**调参差距本轮无法计算**"
              "（下表不会出现该行）。\n")
        A("| 指标 | 数值 |")
        A("|---|---|")
        # 修复项 #1：把「按验证集挑单一 α」（诚实）与「按测试集挑」（作弊）分成两行，
        # 且标签里必须带口径，读者不能只凭「最优单一 α」四个字判断它是否合法。
        nb_p = fs.get("plain_best_single_alpha_n_blocks")
        nb_s = fs.get("sqrth_best_single_alpha_n_blocks")
        for k, label in [
            ("mean_gain_fixed_alpha05", "固定 α=0.5 的平均增益（配对块）"),
            ("mean_gain_tuned_alpha", "按验证集逐格调 α 的平均增益（配对块）"),
            ("mean_gain_oracle_alpha", "按测试集逐格选 α（**作弊上界**，配对块）"),
            ("mean_gain_tuned_alpha_all_blocks", "按验证集逐格调 α（全部块，不参与差值）"),
            ("tuning_gap", "**调参差距**（配对块上的差）"),
            ("blocks_tuned_alpha_positive_pct", "调 α 后为正增益的块占比 (%)"),
            ("mean_gain_sqrth_tuned", "√H 归一化 + 调 α 的平均增益（配对块）"),
            ("mean_gain_sqrth_at_alpha05", "√H 归一化 + 固定 α=0.5（配对块）"),
            ("plain_best_single_alpha_mean_gain",
             "原始尺度：**按验证集**选出的单一 α 的测试增益"
             + (f"（{nb_p} 个全覆盖块）" if nb_p else "")),
            ("sqrth_best_single_alpha_mean_gain",
             "√H 归一化：**按验证集**选出的单一 α 的测试增益"
             + (f"（{nb_s} 个全覆盖块）" if nb_s else "")),
            ("plain_best_single_alpha_mean_gain_test_selected",
             "原始尺度：按**测试集**选单一 α（**作弊上界**，不可引用为方法结果）"),
            ("sqrth_best_single_alpha_mean_gain_test_selected",
             "√H 归一化：按**测试集**选单一 α（**作弊上界**，不可引用为方法结果）"),
            ("plain_mean_distinct_best_alpha_across_horizons", "原始尺度：最优 α 跨 horizon 的取值个数"),
            ("sqrth_mean_distinct_best_alpha_across_horizons", "√H 归一化：最优 α 跨 horizon 的取值个数"),
        ]:
            v = fs.get(k)
            if v is not None and np.isfinite(v):
                A(f"| {label} | {v:+.2f} |" if "占比" not in label and "个数" not in label
                  else f"| {label} | {v:.2f} |")
        A("")
        pa, sa_ = (fs.get("plain_best_single_alpha_by_val"),
                   fs.get("sqrth_best_single_alpha_by_val"))
        if pa is not None or sa_ is not None:
            A("被验证集选中的那个单一 α："
              + "、".join(f"{lab} α={v:g}" for lab, v in
                         (("原始尺度", pa), ("√H 归一化", sa_)) if v is not None)
              + "。选 α **只**用验证集 MSE；同表里带「作弊上界」字样的行仅用于说明"
                "「按测试集选 α 能多拿多少」，不得写成本文方法的结果。\n")

        if (data.get("fredf") or {}).get("detail_path"):
            A(f"逐格明细：`{data['fredf']['detail_path']}`\n")

    # --- phase-1 统计原文 ---
    A("## 4. phase-1 全量统计（原始输出）\n")
    for k in ("p1_stats", "p1_horizon", "p1_selector"):
        if logs.get(k):
            A(f"<details><summary>{k}</summary>\n\n```\n{logs[k]}\n```\n\n</details>\n")

    # --- 已知偏差：写论文前必须自己先说清楚，不然审稿人会替你说 ---
    A("## 5. 已知偏差与需要在论文里交代的事\n")
    A("1. **门控的训练分布带轻微乐观偏差**：逐窗口训练标签来自验证集，而验证集同时被用于"
      "早停和 best-epoch 选择，所以 val 上的逐窗口误差比"
      "「完全没被看过的数据」略优。这不影响 test 端的评估（test 从未参与任何选择），"
      "但会让 val→test 存在分布漂移，是 `inpool` 里门控可能学不到东西的一个候选原因。"
      "干净的做法是从训练集尾部再切一段 gate-train，作为审稿意见的预案。")
    # 修复项 #4：这一条原来写死「20 个 NaN 窗口（ETTm1/ETTm2 各 10 个）」。这个计数随
    # 覆盖度与数据集组合变化，写死会在下一轮覆盖度变化后变成假话。改成只说机制，
    # 具体数字让读者去看 window_features 的自检输出（`window_features` 执行日志）。
    A("2. **窗口特征存在少量 NaN 窗口**（出现在某个通道在该窗口内恒定、相关系数分母为 0 时；"
      "实际数量见执行日志 `window_features` 的自检行，会随覆盖度变化）。"
      "HistGradientBoosting 原生支持缺失值，不需要填补，"
      "但论文里要提一句，不能让人以为是 bug。")
    A("3. **α=0.5 的 plain FreDF 参考来自 p2/phase-1 而非 p3**（p3 的 plain 臂是 "
      "0.1/0.3/0.7/0.9）。已按单 seed 2021 对齐、并用『同格重复跑的 MSE 标准差』做确定性"
      "自检；若该标准差不≈0 就说明有未固定的随机源，必须先查清再引用这一列数字。")
    A("4. **未跑完的格子不是随机缺失**，而是系统性地集中在最贵的角落"
      "（TimesNet × 长 horizon × 大数据集）。所有跨骨干/跨 horizon 的平均值都要用"
      "「同一批可用块」重算，不能拿覆盖度不同的两组平均值直接比。\n")

    # --- 下一步 ---
    A("## 6. 建议的下一步\n")
    A(_next_steps(data))

    A("\n---\n<details><summary>各步骤执行日志（尾部）</summary>\n")
    for k, v in logs.items():
        A(f"\n**{k}**\n```\n{v[-1500:]}\n```\n")
    A("</details>\n")
    return "\n".join(L)


def _next_steps(data: dict[str, Any]) -> str:
    gat = (data.get("gating") or {}).get("inpool") or {}
    onl = (data.get("gating") or {}).get("online") or {}
    sw = (data.get("gating") or {}).get("online_sweep") or {}
    fs = (data.get("fredf") or {}).get("summary") or {}
    steps: list[str] = []
    aud = (data.get("gating") or {}).get("oracle_audit") or {}
    st = (data.get("gating") or {}).get("audit_strata") or {}
    n = gat.get("n_groups", 0)
    hl = aud.get("median_persist_half_life_windows")
    # 修复项 #4：原来写死「H（96~720 个窗口）」，但 ILI 的 H 实际是 24~60，
    # 覆盖度一变这句话就是错的。改成从分层审计的实测 H_min/H_max 现读，取不到就不写数字。
    hmin, hmax = st.get("H_min"), st.get("H_max")
    hrange = (f"（本轮实测 {hmin:.0f}~{hmax:.0f} 个窗口）"
              if all(isinstance(x, (int, float)) and np.isfinite(x) for x in (hmin, hmax))
              else "")
    if not n:
        steps.append("1. 先补 phase-2 的逐窗口误差：门控实验的输入还没齐，其他结论都无法定案。")
    elif (aud.get("n_groups") and aud.get("mean_excess_persist_lagH", 1.0) <= 0.05
          and aud.get("mean_excess_persist_lag1", 0.0) > 0.05):
        # 审计已经把"为什么所有选择器都失败"解释清楚了：相关长度 << 预测视界。
        # 这时继续调门控模型是纯浪费，应该直接转向把这个结论写成论文。
        steps.append(
            "1. **停止调门控，改写论文主张**。审计已经给出机制层面的解释："
            f"最优臂身份的相关半衰期只有 {hl:.1f} 个窗口，而合法延迟是 H{hrange}，"
            "差一到两个数量级。所以静态特征门控、延迟误差反馈、零延迟作弊版本全都赢不了"
            "最佳固定臂，不是方法不够好，而是问题在部署设定下是空的。"
            "论文骨架换成：①oracle 上界统计量对逐窗口重贴臂标签不变（命题+数值验证），"
            "故不能作为可学性证据；②最优臂身份的相关长度衰减曲线与半衰期；"
            "③三类选择器（静态特征/延迟反馈/零延迟作弊）的一致失败作为验证；"
            "④唯一可实现的部分是 val→test 选臂失配的修正。")
        steps.append(
            "2. 只补两个**零 GPU** 的收尾实验：①把衰减曲线按数据集/骨干分层，"
            "确认半衰期短是普遍现象而非某个数据集的特例；②把「相关半衰期 vs H」"
            "画成散点，作为主图。两个都只用已落盘的逐窗口误差。")
    else:
        g = gat.get("mean_gain_gate_vs_best_fixed", 0.0)
        orc = gat.get("mean_gain_oracle_vs_none", 0.0)
        if g > 0.5:
            steps.append("1. 门控有信号：把 TimesNet 与长 horizon 的空缺补齐，然后做特征组消融"
                         "（w_ns / w_fr / w_ch 分别去掉），确定是哪一类信号在起作用。")
            steps.append("2. 加两条必须有的对照：随机门控、以及只用 horizon+backbone 的"
                         "「无窗口特征」门控——用来证明收益真的来自窗口级信息。")
        elif orc < 1.0:
            steps.append("1. oracle 上界太低，说明当前插件池在窗口尺度上区分度不足。"
                         "换池子（加 SAN 完整版 / Dish-TS / FITS 之类）或换问题，不要继续加门控模型复杂度。")
        elif onl.get("mean_gain_online_vs_best_fixed", 0.0) > 0.5:
            # 静态特征学不到但延迟反馈能赢：论文主线应当换成「插件选择需要反馈信号」，
            # 这条路完全不需要新的 GPU 预算，静态门控降级成消融里的负面对照。
            steps.append(
                "1. **换主线**：静态复杂度特征门控赢不了最佳固定臂，但延迟误差反馈的在线选择器"
                f"能赢 {onl['mean_gain_online_vs_best_fixed']:+.2f}%。把论文主张改成「插件选择"
                "可行，但需要的是运行时反馈而非静态复杂度特征」，静态门控作为负面对照写进消融。"
                "下一步：补反馈窗口长度敏感性（50/200/1000）、切换成本约束、以及"
                "「反馈+特征」混合选择器。")
        elif onl.get("mean_delay_cost_pp", 0.0) > 1.0:
            b = sw.get("breakeven_delay_ratio")
            extra = (f"扫描给出的盈亏平衡延迟是 {b:g}×H，"
                     if isinstance(b, (int, float)) and b > 0 else "")
            steps.append(
                "1. 信号存在但被延迟吃掉：不延迟版本比最佳固定臂多拿"
                f" {onl['mean_delay_cost_pp']:.2f} pp，延迟 H 个窗口后收益消失。{extra}"
                "把「插件选择的可行性受反馈延迟约束」写成结论，图就是 delay×window 那张扫描表，"
                "这是一个不需要额外 GPU 的完整贡献。下一步补两件事："
                "① 用 h96 与 h720 分开画曲线（延迟严重程度差 7.5 倍）；"
                "② 试「用 t−H 之前的误差 + 静态特征」的混合选择器，看能否把延迟缺口补回来。")
        else:
            steps.append("1. 有空间但学不到：先查 val→test 的特征分布漂移，再考虑把门控目标从"
                         "「选臂」改成「预测相对增益并设阈值」，或引入近期误差反馈特征。")
    if fs.get("tuning_gap", 0) > 1.0:
        # 修复项 #2：这个阈值分支现在吃的是**配对块**上的差值，把配对块数一起写出来，
        # 否则「差距成立」这句话在覆盖度不全时无法自证。
        steps.append(f"{len(steps)+1}. FreDF 的调参差距成立（{fs['tuning_gap']:+.2f} 个百分点，"
                     f"在 {fs.get('n_blocks_paired_alpha05', 0)} 个配对块上），"
                     "可以据此写「插件的即插即用性被高估」这一节，附官方脚本行号证据。")
    steps.append(f"{len(steps)+1}. 把本报告的关键数字搬进论文骨架（Table 1 主表 / Table 2 门控 / "
                 "Table 3 α 扫描），并更新 README 的进度段。")
    return "\n".join(steps)


def main() -> int:
    ap = argparse.ArgumentParser(description="周末无人值守：自动汇总三组实验并给出结论")
    ap.add_argument("--config-p1", default="configs/matrix.yaml")
    ap.add_argument("--config-p2", default="configs/matrix_p2.yaml")
    ap.add_argument("--config-p3", default="configs/matrix_p3_fredf.yaml")
    ap.add_argument("--fast", action="store_true", help="跳过 src.report 的图表生成")
    args = ap.parse_args()
    build_report(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
