# PlugGate — reproduction package

Repository: <https://github.com/oscarliu2019/plugin-selection-audit>

This repository is the public, self-contained reproduction package for the manuscript

> **Per-window oracle headroom is not learnable: a feedback-delay audit of
> plug-in selection for time series forecasting**
> ([`paper/main.pdf`](paper/main.pdf), 17 pages)

The paper is an **audit and a negative result**. It does not propose a new
plug-in or a new selector. It shows that (i) the per-window oracle headroom that
motivates adaptive plug-in selection is invariant to a random re-labeling of the
per-window arms, so it cannot serve as evidence that selection is learnable;
(ii) the underlying per-window structure is nonetheless real and reproducible
across training seeds; and (iii) it is unusable anyway, because the identity of
the best arm decorrelates with a median half-life of about **5%** of the forecast
horizon while deployment forces a decision lag of exactly **one** horizon.

Everything numeric in the paper is recomputed from the artifacts in this folder by
a script: `tools/verify_paper_numbers.py` re-derives **204 registered claims** and
fails on any mismatch, in either direction (artifact changed but prose did not, or
prose changed but artifact did not). There is a test that keeps the checker itself
honest (§2).

Scope of the experiments: 4 backbones (DLinear, PatchTST, TimesNet,
iTransformer) x 8 datasets (ETTh1, ETTh2, ETTm1, ETTm2, Electricity, Exchange,
ILI, Weather) x 4 horizons x 4 arms (`none`, `revin`, `san_lite`, `fredf`),
**2,035 training runs** in three phases on a single V100.

---

## 1. Install

```bash
git clone https://github.com/oscarliu2019/plugin-selection-audit.git
cd plugin-selection-audit
```

Python 3.11 is what was used. CPU is enough for *everything in this README except
retraining* (§4).

```bash
python -m venv .venv && source .venv/bin/activate

# analysis only (verification, tables, figures, paper build):
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu

# retraining on a GPU instead:
#   pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
#   (on a driver older than 525, use the cu118 wheel: torch==2.5.1+cu118)

pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used. Two pins are load-bearing:
`pandas<3` (the vendored TSLib data loader is incompatible with pandas 3) and
`torch==2.5.1` (later wheels drop the `sm_70` architecture of the V100).

All commands below are run **from this `release/` directory**.

---

## 2. Verify every number in the paper (~2 min, CPU, no data download)

```bash
python tools/verify_paper_numbers.py                     # must print 204/204
python tools/verify_paper_numbers.py --list              # every claim and its source
python tools/verify_paper_numbers.py -v --only splitgate # one group, verbose
python tools/verify_paper_numbers.py --json verify.json  # machine-readable
```

Each of the 204 registered claims carries a recompute function that reads
`monday_final/` or `artifacts/robustness/` and compares the recomputed value with
the registered one under an explicit tolerance. The checker then performs a
**second, independent check**: it searches the normalised text of
`paper/main.tex` and `paper/tables_robust/*.tex` for the literal value, so a
number that was updated in an artifact but not in the prose (or the reverse) also
fails.

Both directions are necessary. The first alone misses "the prose and the table
contradict each other"; the second alone misses "the paper and the artifact are
wrong together". This is not ceremony: §12 of an earlier draft reported the
within-test split gate as `-0.55` while Table 13 and §9.2 reported `-0.53`
(`-0.55` was the TimesNet-only value, copied into the wrong sentence). Three
manual read-throughs missed it; the checker caught it on the first run, and
`tests/test_verify_paper_numbers.py` now pins that specific regression.

The test suite (417 tests, ~90 s, CPU) covers the plug-in wrappers
(invertibility, no future leakage, zero added trainable parameters), the
statistics (Nemenyi critical-difference reference values, Holm monotonicity, the
power gained by the block definition), the scheduler (resumability, aggregation
hygiene), the window-level audits, and the verifier itself (including whether it
can be fooled):

```bash
python -m pytest tests/ -q      # 417 passed, 1 skipped (the skip needs a GPU)
python -m flake8 src tools tests conftest.py scripts
```

---

## 3. Regenerate the tables, figures and PDF from the shipped artifacts (CPU)

All four commands are pure post-processing: they read the result tables in
`monday_final/`, need no GPU and no dataset download.

```bash
# (a) the robustness / metric-lever / seed / lead-time / cost audits
#     regenerates artifacts/robustness/*.csv|json byte-for-byte
python -m src.robustness --results monday_final/p1_results.csv --out artifacts/robustness

# (b) the four LaTeX tables built from those audits (do not edit them by hand)
python tools/make_robustness_tables.py --robust-dir artifacts/robustness --out paper/tables_robust

# (c) Fig. 1, Fig. 2 and the 20-stratum table
python tools/make_audit_figs.py \
    --audit-csv monday_final/p2/selector_window_oracle_audit.csv \
    --out-dir monday_final/paper
cp monday_final/paper/fig_halflife_vs_horizon.pdf monday_final/paper/fig_decay_curves.pdf paper/figs/

# (d) Table 1 and Table 2 (the two aggregation conventions), recomputed
#     independently from the 844 raw phase-1 rows
python tools/recompute_main_table.py monday_final/p1_results.csv

# (e) the manuscript itself (bibliography -> four passes)
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

Expected: (a) and (b) reproduce the shipped files exactly; (c) reproduces
`audit_strata.csv` with an identical MD5 and the figures pixel-for-pixel (the
`.pdf` bytes differ only in the embedded creation timestamp); (e) gives 17 pages
with 0 undefined citations and 0 undefined references. The build prints one
`Overfull \hbox (117pt)` on the title page — that comes from the e-mail icon in
Elsevier's own CAS template and is present in the official sample too.

`monday_final/weekend_report.{md,json}` is the aggregate report over all three
phases; it is produced by `tools/weekend_report.py`, which reads the live
`results/` tree of a training run and therefore only re-runs after §4.

---

## 4. Re-run the experiments from scratch (needs a GPU)

```bash
# 4.0 datasets (~123 MB, scripted, row and channel counts are verified)
python scripts/download_data.py
python scripts/download_data.py --verify-only

# 4.1 cheap sanity gate before burning GPU hours
python -m pytest tests/ -q

# 4.2 complexity features of the training split only (CPU, ~90 s)
python -m src.features --out artifacts/features.csv

# 4.3 measure ms/iter, then look at the schedule and the budget
python -m src.scheduler --probe --probe-steps 40 --device cuda
python -m src.scheduler --dry-run

# 4.4 phase 1: the main 844-cell matrix (~123 GPU-hours measured on one V100,
#     ~2.3 days wall clock at --workers 3). Resumable: rerun to continue.
python -m src.scheduler --run --device cuda --workers 3
python -m src.scheduler --status
python -m src.scheduler --aggregate            # -> results/results.csv

# 4.5 phase 2: the four-arm pool re-run that persists per-window errors.
#     --save-window-mse is what makes every window-level audit possible.
python scripts/make_phase2_config.py
python -m src.scheduler --config configs/matrix_p2.yaml --run --device cuda \
       --workers 3 --save-window-mse

# 4.6 phase 3: the FreDF alpha sweep (768 cells, ~27 GPU-hours)
python scripts/make_fredf_config.py
python -m src.scheduler --config configs/matrix_p3_fredf.yaml --run --device cuda --workers 3

# 4.7 the seed-paired replication group
python scripts/make_p2seed_config.py
python -m src.scheduler --config configs/matrix_p2seed.yaml --run --device cuda \
       --workers 1 --save-window-mse --backbones DLinear --datasets ETTh1 ETTh2 Exchange ILI

# 4.8 the window-level audits (CPU, minutes)
python -m src.features_window --config configs/matrix_p2.yaml     # window feature table
python -m src.selector_window --config configs/matrix_p2.yaml --protocol inpool \
       --mode reg --oracle-audit --online --controls --ablate
python -m src.selector_window --config configs/matrix_p2.yaml --protocol none --seed-audit \
       --winerr results/p2/winerr results/p2seed/winerr --tag seed
python -m src.selector_window --config configs/matrix_p2.yaml --protocol none \
       --split-audit --split-train-frac 0.7 --coverage-gap \
       --audit-csv artifacts/p2/selector_window_oracle_audit.csv --tag splitfix --out-dir artifacts/p2

# 4.9 aggregate everything into one report
python tools/weekend_report.py
```

Note on the raw per-window error arrays (`results/p2/winerr/*.npy`, written by
4.5): they are **not** redistributed here. Step 4.8 regenerates every derived
artifact from them, and those derived artifacts *are* shipped in
`monday_final/p2/`.

`--save-window-mse` on the validation split must go through the deterministic
loader that also writes a `.order.json` marker. TSLib's `data_factory` only
disables shuffling for the `test` split, so validation-window traces collected
without that marker are in a random permutation. This was a real, published-draft
bug in this project; `tests/test_val_order_provenance.py` and
`tests/test_winerr_persist_guard.py` now make it impossible to persist an
unmarked validation trace.

---

## 5. Where every number in the paper comes from

| Paper | Number | Artifact / command |
|---|---|---|
| §3 | 2,035 runs = 844 + 423 + 768 | `monday_final/weekend_report.json` -> `coverage.*` |
| §4.1, Tab. 1–2 | two aggregation conventions, per-backbone sign flips | `python tools/recompute_main_table.py monday_final/p1_results.csv` |
| §4.2 | MSE/MAE verdict flips, rank correlation | `artifacts/robustness/metric_robustness.csv`, `metric_agreement.csv` |
| §4.3 | mean vs. median conflict, left-tail risk, worst block | `metric_robustness.csv` (`skew`, `worst_*`, `frac_worse_than_*`) |
| §5, Tab. 3 | FreDF tuning gap 5.48 pp, 768 alpha cells | `weekend_report.json` -> `fredf`; per cell `monday_final/p3/fredf_alpha.csv` |
| §6 | seed noise, compute cost | `artifacts/robustness/seed_stability.csv`, `cost_benefit_by_method.csv` |
| §7 | lead-time (legal axis) headroom +0.24% | `artifacts/robustness/robustness_summary.json` -> `leadtime` |
| §8.1, Prop. 1 | oracle re-labeling invariance 127/127, lag agreement | `monday_final/p2/selector_window_oracle_audit.csv` |
| §8.2, Fig. 1–2 | half-life 9.0 windows = 5.1% of H, 20/20 strata | same, plus `monday_final/paper/audit_strata.csv` |
| §8.3 | seed-paired replication (79% reproducible) | `weekend_report.json` -> `seed_audit` |
| §9.1 | gating: in-pool / classifier / LODO / controls | `weekend_report.json` -> `gating.inpool*`, `gating.lodo` |
| §9.2 | within-split diagnostic: val −1.18% (81 groups), test −0.53% (111) | `monday_final/paper/splitfix/selector_window_split_{val,test}_splitfix.csv` |
| §9.3 | online, delay sweep, break-even at 0.25xH | `weekend_report.json` -> `gating.online`, `gating.online_sweep` |
| §10 | +1.36% / −2.32% decomposition | `gating.online` -> `best_fixed_test` fields |
| §12, Tab. 13 | coverage gap 127 vs 96, gap homogeneity | `monday_final/paper/splitfix/selector_window_coverage_gap_splitfix.csv` |

`tools/verify_paper_numbers.py --list` prints the same mapping at claim
granularity.

---

## 6. Layout

```
release/
├── paper/                     manuscript sources; main.pdf is committed
│   ├── main.tex  refs.bib  cas-*.cls|sty|bst   Elsevier CAS template, vendored
│   ├── figs/                  Fig. 1 and Fig. 2 (generated, do not edit)
│   └── tables_robust/         4 generated LaTeX tables (do not edit)
├── configs/                   declarative experiment matrices
│   ├── matrix.yaml            phase 1: backbones, datasets, horizons, arms,
│   │                          seeds, cost model, skip rules, protocol deviations
│   ├── matrix_p2.yaml         phase 2: four-arm pool with per-window persistence
│   ├── matrix_p3_fredf.yaml   phase 3: FreDF alpha sweep
│   └── matrix_p2seed.yaml     seed-paired replication group
├── src/
│   ├── config.py              matrix loading, cell enumeration, atomic writes
│   ├── features.py            complexity / non-stationarity / spectral features
│   ├── features_window.py     the same features per input window
│   ├── plugins/               zero-intrusion wrappers: none, revin, san_lite,
│   │                          fredf, fredf_sqrth  (no new trainable parameters)
│   ├── runner.py              one cell: train, streaming metrics, per-window MSE
│   ├── train_cell.py          single-cell CLI (subprocess target of the scheduler)
│   ├── scheduler.py           cost model, ordering, resume, atomic aggregation
│   ├── stats.py               Wilcoxon + Holm, Friedman + Nemenyi, win/tie/loss
│   ├── selector.py            block-level selector (LODO) and decision gain
│   ├── selector_window.py     window-level gating, oracle audit, delayed-feedback
│   │                          online selection, seed audit, split diagnostics
│   ├── robustness.py          metric / estimator / seed / lead-time / cost audits
│   ├── horizon.py             horizon-aware checkpoint selection criteria
│   ├── report.py              result tables -> LaTeX
│   ├── fft_compat.py          forces fp32 FFT under AMP (cuFFT fp16 restriction)
│   └── synth.py               synthetic results, for testing the analysis code only
├── tools/
│   ├── verify_paper_numbers.py    the 204-claim checker (start here)
│   ├── recompute_main_table.py    Tab. 1–2 from the raw phase-1 rows
│   ├── make_audit_figs.py         Fig. 1, Fig. 2, 20-stratum table
│   ├── make_robustness_tables.py  paper/tables_robust/*.tex
│   └── weekend_report.py          aggregate report over all three phases
├── scripts/                   dataset download and config derivation
├── tests/                     417 tests
├── monday_final/              the authoritative results (see below)
├── artifacts/                 features.csv and the robustness audit outputs
└── third_party/tslib/         vendored Time-Series-Library, unmodified
```

`monday_final/` is the only authoritative result directory:

| Path | Content |
|---|---|
| `p1_results.csv` | 844 phase-1 runs, one row each |
| `p2_results.csv` | 423 phase-2 runs, one row each |
| `p3/fredf_alpha.csv` | phase-3 alpha sweep, aggregated to 96 blocks |
| `p2/window_features.parquet` | 82,492 windows x 29 columns (36,871 val + 45,621 test) |
| `p2/selector_window_oracle_audit.csv` | per-group decay curves, half-lives, re-labeling invariance |
| `p2/selector_window_{inpool,inpool_clf,lodo,controls,ablation}.csv` | the gating protocols and their falsification controls |
| `p2/selector_window_online{,_ext,_sweep,_sweep_ext}.csv` | delayed-feedback selection and the delay x feedback sweep |
| `p2/selector_window_seed_audit_seed.csv` | seed-paired replication, 48 pairs |
| `paper/audit_strata.csv` | the 20 strata (by backbone / dataset / horizon) |
| `paper/splitfix/*.csv` | within-split gate diagnostic and the coverage gap |
| `weekend_report.{md,json}` | the aggregate report; `*.json` is what the verifier reads |

A note on language: this README and the manuscript are in English; the docstrings
inside `src/` are in Chinese, because they were written as working notes while the
experiments ran. Every module is summarised in English in the tree above, and
`tools/verify_paper_numbers.py --list` is language-neutral.

---

## 7. Protocol facts a reproducer needs to know

1. **Baselines are re-trained here, not copied from papers.** Upstream TSLib now
   sets `drop_last=False` for the test loader, which makes the published PatchTST
   and iTransformer tables numerically incomparable. Every table in the paper
   states this. Registered in `configs/matrix.yaml` under
   `meta.protocol.drop_last_test`.
2. **A statistical block is (dataset x horizon x backbone)**, not a dataset. With
   the dataset-level convention (k=4, N=8 datasets) the Nemenyi critical
   difference is 1.66 -- wider than the entire observed rank range, so no plug-in
   comparison can ever be significant; with this convention (k=4, N=128) it is
   0.41. `tests/test_stats.py` pins the resulting sqrt(16)=4 power ratio.
3. **`san_lite` is a training-free surrogate of SAN, not SAN.** The paper says so
   explicitly. A training-free extrapolator of future slice statistics is the
   part of SAN that can be made architecture-agnostic; the negative result about
   ridge extrapolation is reported as such.
4. **`revin` is a provable near-no-op on 3 of the 4 backbones.** PatchTST,
   iTransformer and TimesNet normalise instances inside `forecast()` in this
   TSLib revision and expose no switch. Those cells are pruned *a priori* by
   `skip_rules.revin_on_internal_norm_backbones`, except for a retained
   verification subset ({ETTh1, Weather} x {96, 720} on all three backbones,
   12 cells) that demonstrates the ~0 gain empirically: median gain +0.00%,
   Wilcoxon p=0.23, and MSE unchanged to four decimals in 7 of the 8
   PatchTST/iTransformer cells. On DLinear, which has no internal normalisation,
   the same arm has a median gain of +0.98% over 64 cells. Nothing crashed --
   all 844 phase-1 rows have `status == ok` -- so this pruning, not a failure, is
   why `revin` has 44 blocks where the other arms have 128, and why the main table
   is reported per backbone.
5. **Seed policy.** One seed (2021) over the whole matrix; three seeds
   (2021/2022/2023) on the cheap tier (ETTh1, ETTh2, Exchange, ILI); one seed on
   the heavy tier (Electricity). Stated in the paper.
6. **Coverage is asymmetric and reported as such.** The test-side audits cover
   127 decision groups; the gating protocols cover 96, because 31 TimesNet groups
   have no `.order.json` marker on their validation traces. Table 13 shows the
   missing groups are distributed like the covered ones, and that their oracle
   headroom is *larger*, i.e. closing the gap would add samples without flipping
   any verdict.
7. **Under AMP, FFTs are forced to fp32** (`src/fft_compat.py`): cuFFT only
   supports half precision for power-of-two signal lengths, and
   `seq_len + pred_len` is 192 or 816 here. Registered as
   `meta.deviations.fft_in_fp32_under_amp`. TSLib itself is never patched.

---

## 8. Environment actually used

| | |
|---|---|
| GPU | one Tesla V100-SXM2-32GB (`sm_70`), driver 470, CUDA 11.4 |
| PyTorch | 2.5.1+cu118 on the GPU host, 2.5.1+cpu for analysis |
| Peak memory | 3.56 GB for the largest cell |
| Concurrency | 3 cells per GPU (2.06x throughput; saturates at 4) |
| Measured training time | 185 GPU-hours (phase 1, sum of `train_seconds` over 844 runs) + 104 (phase 2, 423 runs) + ~27 scheduled (phase 3, 768 runs); wall clock is roughly half of the sum because of the 3-way concurrency |
| Numerical check | ETTh1/H=96 over 3 seeds agrees with the published TSLib numbers within 4% for all four backbones |

---

## License and attribution

- This package: MIT, see [`LICENSE`](LICENSE).
- `third_party/tslib/` is an unmodified copy of the
  [Time-Series-Library](https://github.com/thuml/Time-Series-Library) (MIT). All
  plug-ins are applied as external wrappers; no upstream file is edited. See
  [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- Datasets are the standard public long-term-forecasting benchmarks and are not
  redistributed; `scripts/download_data.py` fetches them from the official
  mirror.

## Citing

> M. Liu and C. Chen. *Per-window oracle headroom is not learnable: a
> feedback-delay audit of plug-in selection for time series forecasting.* Under
> review, 2026.

If you use this audit, please cite the paper above. If you use only the checklist
of §11 of the paper, citing the paper is still the right thing to do -- the
checklist is the part we most want reused.
