"""论文数字校验器自身的测试。

这个脚本是投稿前的最后一道闸门，所以它自己必须先是可信的。测试分三层：

1. **归一化与字面检查真的会失败**。一个恒真的检查比没有检查更危险，因为它会
   给人虚假的安全感。所以显式验证：改动一位数字就必须被抓到。
2. **异常被记录成失败而不是中断整轮**。校验 160 多条时，第 3 条抛异常就退出
   意味着后面 150 条的问题要分好几轮才发现。
3. **claim 清单本身自洽**：id 唯一、容差非负、`--only` 能选中。

最后附一个针对真实产物的集成测试：全部 claim 必须通过。产物不在时跳过，
这样别人 clone 仓库（不含 `monday_final/`）也能跑单测。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.verify_paper_numbers import (  # noqa: E402
    CLAIMS,
    Artefacts,
    Claim,
    check,
    load_tex,
    main,
    normalise_tex,
    render,
)


# --------------------------------------------------------------------------- #
# 归一化
# --------------------------------------------------------------------------- #
def test_normalise_strips_latex_noise_so_numbers_can_be_matched():
    """`$2{,}035$` 和 `2,035` 必须能对上，否则字面检查会全线误报。"""
    assert "2,035" in normalise_tex(r"a total of $2{,}035$ runs")
    assert "+10.15%" in normalise_tex(r"oracle gains $+10.15\%$ over")
    assert "-0.53" in normalise_tex(r"& $\mathbf{-0.53}$ &")


def test_normalise_does_not_rewrite_digits():
    """归一化只允许删噪声，不允许改数字——否则它能把错的洗成对的。"""
    out = normalise_tex(r"$-0.55\%$ and $+9.74$")
    assert "-0.55" in out
    assert "-0.53" not in out
    assert "+9.74" in out


def test_normalise_collapses_whitespace_across_linebreaks():
    """LaTeX 会在任意位置换行，跨行的 `56/ 128` 不能因此漏匹配。"""
    assert "56/128" in normalise_tex("beats it in $56/128$\n  blocks")
    assert normalise_tex("a\n\n  b") == "a b"


# --------------------------------------------------------------------------- #
# 字面检查确实会失败
# --------------------------------------------------------------------------- #
def _dummy(value: float) -> Claim:
    return Claim(cid="t", section="§0", expected=value, tol=1e-9,
                 source="test", fn=lambda a: value, printed=("+1.23",))


def test_literal_check_fails_when_paper_prints_a_different_number():
    """核心回归：产物算出 1.23，但论文写着 1.24 —— 必须 FAIL。"""
    [r] = check([_dummy(1.23)], art=None, tex="the gain is +1.24 percent")
    assert r.numeric_ok is True          # 数值本身没问题
    assert r.literal_ok is False         # 但论文里印错了
    assert r.ok is False
    assert r.missing_literals == ("+1.23",)


def test_literal_check_passes_when_paper_prints_the_number():
    [r] = check([_dummy(1.23)], art=None, tex="the gain is +1.23 percent")
    assert r.ok is True
    assert r.missing_literals == ()


def test_numeric_check_fails_outside_tolerance():
    """登记值和复算值不一致时，即使论文字面对得上也要 FAIL。"""
    c = Claim(cid="t", section="§0", expected=1.0, tol=0.01,
              source="test", fn=lambda a: 1.5)
    [r] = check([c], art=None, tex="")
    assert r.numeric_ok is False
    assert r.computed == 1.5


def test_numeric_check_respects_tolerance():
    c = Claim(cid="t", section="§0", expected=1.0, tol=0.01,
              source="test", fn=lambda a: 1.005)
    [r] = check([c], art=None, tex="")
    assert r.ok is True


# --------------------------------------------------------------------------- #
# 异常不中断
# --------------------------------------------------------------------------- #
def test_a_raising_claim_is_recorded_not_propagated():
    """一条 claim 读不到文件时，其余 claim 仍要跑完并给出结论。"""
    def boom(a):
        raise FileNotFoundError("no such artefact")

    ok = Claim(cid="ok", section="§0", expected=1.0, tol=0, source="t",
               fn=lambda a: 1.0)
    bad = Claim(cid="bad", section="§0", expected=1.0, tol=0, source="t", fn=boom)
    results = check([bad, ok], art=None, tex="")
    assert [r.claim.cid for r in results] == ["bad", "ok"]
    assert results[0].ok is False
    assert "FileNotFoundError" in results[0].error
    assert results[1].ok is True, "一条失败不应影响其他 claim"


def test_nan_only_matches_nan():
    """NaN 不能被当成「和任何东西都相等」。"""
    c = Claim(cid="t", section="§0", expected=1.0, tol=1e9,
              source="t", fn=lambda a: float("nan"))
    [r] = check([c], art=None, tex="")
    assert r.numeric_ok is False, "NaN 不能因为容差大就算通过"


# --------------------------------------------------------------------------- #
# claim 清单自洽
# --------------------------------------------------------------------------- #
def test_claim_ids_are_unique():
    ids = [c.cid for c in CLAIMS]
    assert len(ids) == len(set(ids)), f"重复 id: {[i for i in ids if ids.count(i) > 1]}"


def test_claims_have_sane_tolerances_and_sources():
    for c in CLAIMS:
        assert c.tol >= 0, c.cid
        assert c.source, f"{c.cid} 没写数据来源"
        assert callable(c.fn), c.cid


def test_registry_covers_every_major_section():
    """论文每个有数字的章节都必须至少有一条 claim 兜着。"""
    sections = {c.section for c in CLAIMS}
    for expected in ("§3", "§4.1", "§4.2", "§4.3", "§5", "§6",
                     "§7", "§8", "§8.3", "§9.1", "§9.2", "§9.3", "§10", "§12"):
        assert expected in sections, f"{expected} 没有任何数字校验"


def test_registry_is_large_enough_to_be_meaningful():
    assert len(CLAIMS) >= 150, "claim 太少，说明有整段结论没被钉住"


# --------------------------------------------------------------------------- #
# 渲染与 CLI
# --------------------------------------------------------------------------- #
def test_render_reports_failures_prominently():
    out = render(check([_dummy(1.23)], art=None, tex="nothing here"), verbose=False)
    assert "FAIL" in out
    assert "0/1 条通过" in out
    assert "1 条失败" in out


def test_render_marks_success():
    out = render(check([_dummy(1.23)], art=None, tex="+1.23"), verbose=True)
    assert "FAIL" not in out
    assert "1/1 条通过" in out


def test_cli_list_mode_does_not_touch_artefacts(capsys, tmp_path):
    """`--list` 必须在没有任何产物的空目录下也能工作。"""
    rc = main(["--root", str(tmp_path), "--list"])
    assert rc == 0
    assert "共" in capsys.readouterr().out


def test_cli_only_filters_claims(capsys, tmp_path):
    main(["--root", str(tmp_path), "--list", "--only", "splitgate"])
    out = capsys.readouterr().out
    assert "splitgate" in out
    assert "protocol.p1_done" not in out


def test_cli_rejects_an_unmatched_filter(tmp_path):
    assert main(["--root", str(tmp_path), "--list", "--only", "no_such_claim"]) == 2


def test_cli_returns_nonzero_when_artefacts_are_missing(tmp_path, capsys):
    """产物缺失必须是失败，不能静默当成通过。"""
    rc = main(["--root", str(tmp_path), "--skip-literals"])
    assert rc == 1
    assert "FAIL" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# 针对真实产物的集成测试
# --------------------------------------------------------------------------- #
_HAVE_ARTEFACTS = (ROOT / "monday_final" / "weekend_report.json").exists()
_HAVE_PAPER = (ROOT / "paper" / "main.tex").exists()
_needs_artefacts = pytest.mark.skipif(
    not _HAVE_ARTEFACTS, reason="monday_final/ 产物不在（clone 时未附带）")


@_needs_artefacts
def test_every_claim_matches_the_real_artefacts():
    """投稿闸门：论文里的每个数都必须能从落盘产物复算出来。"""
    results = check(CLAIMS, Artefacts(ROOT), tex="")
    bad = [(r.claim.cid, r.claim.expected, r.computed, r.error)
           for r in results if not r.numeric_ok]
    assert not bad, f"{len(bad)} 条与产物不符: {bad[:8]}"


@_needs_artefacts
@pytest.mark.skipif(not _HAVE_PAPER, reason="paper/main.tex 不在")
def test_every_number_is_actually_printed_in_the_paper():
    """反向闸门：产物变了但论文忘改，这条会红。"""
    tex = load_tex(ROOT / "paper")
    results = check(CLAIMS, Artefacts(ROOT), tex)
    bad = [(r.claim.cid, r.missing_literals) for r in results if not r.literal_ok]
    assert not bad, f"{len(bad)} 条在论文里找不到: {bad[:8]}"


@_needs_artefacts
def test_the_split_gate_regression_that_actually_happened():
    """§12 曾把 test 内部切分门控写成 -0.55，而表 13/§9.2 是 -0.53。

    这条测试专门钉住那次真实的漂移：既要求产物算出 -0.53，也要求论文里
    再也不出现 -0.55。
    """
    art = Artefacts(ROOT)
    assert art.split_gate("test", "gain_gate_vs_best_fixed") == pytest.approx(
        -0.5276, abs=5e-4)
    if _HAVE_PAPER:
        tex = load_tex(ROOT / "paper")
        assert "-0.53" in tex
        assert "-0.55" not in tex, "旧的错值又回来了"


@_needs_artefacts
def test_total_runs_is_the_sum_of_the_three_phases():
    """2035 这个总数是三期相加，不是配置数；防止有人改了某一期忘了改总数。"""
    art = Artefacts(ROOT)
    parts = [art.j("coverage.phase1.n_done"),
             art.j("coverage.phase2.n_done"),
             art.j("coverage.fredf.n_done")]
    assert parts == [844, 423, 768]
    assert sum(parts) == 2035


@_needs_artefacts
def test_alpha_cells_and_phase3_runs_are_distinct_quantities():
    """两个 768 含义不同：α-cell 数 vs phase-3 训练数。论文里区分过，这里锁住。"""
    art = Artefacts(ROOT)
    alpha_cells = float((art.p3.plain_n_alphas + art.p3.sqrth_n_alphas).sum())
    assert alpha_cells == 96 * (5 + 3) == 768
    assert art.j("coverage.fredf.n_done") == 768
    # 其中 96 个 plain α=0.5 复用 phase-1/2，phase-3 真正新训练的 α-cell 是 672
    assert alpha_cells - 96 == 672


@_needs_artefacts
def test_json_output_is_machine_readable(tmp_path):
    out = tmp_path / "verify.json"
    main(["--root", str(ROOT), "--json", str(out)])
    payload = json.loads(out.read_text())
    assert payload["n_total"] == len(CLAIMS)
    assert payload["n_failed"] == 0, "投稿前必须全绿"
    assert {"id", "expected", "computed", "ok"} <= set(payload["results"][0])


# --------------------------------------------------------------------------- #
# 投稿材料与论文本体的一致性
# --------------------------------------------------------------------------- #
# 这一组是被真实错误逼出来的：cover letter 里的标题是凭记忆写的，和 main.tex 的
# 真实标题完全不同（"The Selection Illusion..." vs "Per-window oracle headroom is
# not learnable..."）。这种错一旦投出去，编辑第一眼就会看到标题和稿件不符。
# 标题、邮箱、claim 数这三样在两处重复出现，就必须有测试钉住。
_PAPER = ROOT / "paper"
# 2026-09-15 重组：cover letter 是投稿材料而不是复现材料，因此留在
# `<repo>/internal/submission/`，不进 release 包。这三条检查在完整仓库里照常执行；
# 只拿到 release/（例如从投稿 zip 解出来）时优雅跳过，而不是报错。
_COVER = ROOT.parent / "internal" / "submission" / "cover_letter.md"
_needs_paper = pytest.mark.skipif(not _HAVE_PAPER, reason="paper/main.tex 不在")
_needs_cover = pytest.mark.skipif(not _COVER.exists(), reason="cover_letter.md 不在")


def _paper_title() -> str:
    """从 `main.tex` 抽出真实标题，去掉 LaTeX 换行与多余空白。"""
    import re
    src = (_PAPER / "main.tex").read_text()
    m = re.search(r"\\title\[mode = title\]\{(.+?)\}", src, re.S)
    assert m, "main.tex 里找不到 \\title"
    return re.sub(r"\s+", " ", m.group(1)).strip()


def _norm(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", text).strip().lower()


@_needs_paper
@_needs_cover
def test_cover_letter_title_matches_the_manuscript():
    """真实发生过的错误：cover letter 标题是凭记忆写的，和稿件不一致。"""
    cover = _COVER.read_text()
    assert _norm(_paper_title()) in _norm(cover), (
        f"cover letter 里的标题和 main.tex 不一致；main.tex 是:\n  {_paper_title()}")


@_needs_paper
@_needs_cover
def test_cover_letter_and_paper_agree_on_the_contact_email():
    cover = _COVER.read_text()
    src = (_PAPER / "main.tex").read_text()
    import re
    m = re.search(r"\\gdef\\correspemail\{(.+?)\}", src)
    assert m, "main.tex 里找不到 correspemail 宏"
    assert m.group(1) in cover, "cover letter 的邮箱和稿件不一致"


@_needs_cover
def test_cover_letter_claim_count_matches_the_registry():
    """cover letter 里写了「161 处数字全部机器校验」，registry 增长后必须同步。"""
    import re
    cover = _COVER.read_text()
    counts = [int(n) for n in re.findall(r"All (\d+) numerical claims", cover)]
    assert counts, "cover letter 里找不到 claim 数量的表述"
    for n in counts:
        assert n == len(CLAIMS), (
            f"cover letter 写了 {n} 条，registry 实际 {len(CLAIMS)} 条")


@_needs_paper
def test_manuscript_has_no_leftover_placeholders():
    """带占位符投出去是硬伤：匿名仓库 URL、TODO、XXX 一律不许留。"""
    src = (_PAPER / "main.tex").read_text()
    for bad in ("anonymised repository", "anonymous repository", "[TODO",
                "TODO:", "XXX", "\\todo{", "FIXME"):
        assert bad not in src, f"main.tex 里还留着占位符 {bad!r}"


@_needs_paper
def test_manuscript_declares_everything_elsevier_requires():
    """Elsevier 强制的几节缺一个就会被退回补件。"""
    src = (_PAPER / "main.tex").read_text()
    for section in ("Declaration of competing interest",
                    "Data availability",
                    "generative AI",
                    "Funding"):
        assert section in src, f"缺少 {section!r} 声明"


@_needs_paper
def test_credit_statement_appears_exactly_once():
    """曾经手写 CRediT 小节又调 \\printcredits，导致贡献声明重复出现两次。"""
    src = (_PAPER / "main.tex").read_text()
    assert src.count("\\printcredits") == 1
    assert "\\section*{CRediT" not in src, "又手写了 CRediT 小节，会和 \\printcredits 重复"


@_needs_paper
def test_highlights_respect_the_elsevier_length_limit():
    """Elsevier 要求 3–5 条、每条 ≤85 字符；超了会被系统退回。"""
    import re
    src = (_PAPER / "main.tex").read_text()
    block = re.search(r"\\begin\{highlights\}(.+?)\\end\{highlights\}", src, re.S)
    assert block, "找不到 highlights 环境"
    items = re.findall(r"\\item\s+(.+)", block.group(1))
    assert 3 <= len(items) <= 5, f"highlights 有 {len(items)} 条，要求 3–5 条"
    for it in items:
        plain = it.replace("\\%", "%").replace("\\emph", "").strip()
        assert len(plain) <= 85, f"这条 {len(plain)} 字符，超过 85: {plain}"
