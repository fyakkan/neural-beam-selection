# Consolidated Results

All experiments use the official pre-recorded data (15,000 train / 500 test UE
locations), 70 beam pairs (10 Tx × 7 Rx), identical splits/seeds. RSRP in dB;
blocked beams floored to −120 dB. Reproduce: `baseline/run_baseline.m`,
`novel/run_gated.m`, and `novel/exp_*.m`.

## 1. Benchmark reproduction (Task 1) — `baseline/run_baseline.m`
Regression NN: 14 downsampled RSRP → 70-element RSRP profile; top-K sweep.

| Metric | NN | KNN | Optimal |
|---|---|---|---|
| 90% top-K at | **K = 13** | — | — |
| overhead reduction @K13 | **81.4%** | — | — |
| top-K acc @K=13 | **90.0%** | 37.8% | — |
| top-K acc @K=30 | 97.4% | — | — |
| avg RSRP @K=13 (dB) | **−23.21** | −25.29 | −22.97 |

NN reaches 90% sweeping <¼ of beams, within 0.24 dB of exhaustive, beating
KNN/Statistical/Random. Figures: `baseline_topk_accuracy.png`, `baseline_avg_rsrp.png`.

## 2. Proposed method (gated residual fusion) — `novel/run_gated.m`
out = tanh( base(RSRP) + gate·corr(position) ). Both RSRP-only and fusion are
trained impairment-aware (blockage of 0–8 of 14 input beams + light noise).

**Blockage robustness — top-K=13 accuracy (mean ± std over 10 random blockage draws):**

| # blocked (of 14) | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| RSRP-only | 89.2±0.0 | 84.6±1.1 | 78.1±2.2 | 69.8±1.5 | 56.0±1.6 | 35.9±2.4 | 22.3±0.9 |
| **Gated Fusion** | 89.4±0.0 | 83.5±1.0 | 78.7±1.1 | 72.1±1.7 | 57.9±2.0 | 39.7±1.7 | 24.9±1.4 |
| gain (pts) | +0.2 | −1.1 | +0.6 | +2.3 | +2.0 | +3.8 | +2.6 |

(The ±std above is **blockage-draw** variance at the canonical seed=1; **seed/initialization**
variance is characterized separately in §2a below.)

**Link quality under blockage — avg RSRP (dB) of the SELECTED beam (K=13, mean over 10 draws):**

| # blocked (of 14) | 0 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| RSRP-only | −23.23 | −24.11 | −25.08 | −27.20 | −28.61 |
| **Gated Fusion** | −23.21 | −23.84 | −24.52 | −25.95 | −27.45 |
| gain (dB) | +0.01 | +0.27 | +0.56 | +1.26 | +1.16 |

The top-K accuracy gain translates into a better link (higher selected RSRP), and the margin
GROWS with blockage severity (+1.26 dB at nB=10). Exhaustive optimum −22.97 dB. This is paper
Table IV (`tab:blockrsrp`). Numbers from `results/metrics/novel_metrics.json` (`rsrpRSRP`/`rsrpFusion`).

Position-only KNN ≈ 37.8% (flat, blockage-immune). Headline: **no clean-data
cost (+0.2), consistent +2–4 pt gain once beams are blocked** — a no-regret
robustness improvement. **Significance:** gains at nB≤4 (+0.2/−1.1/+0.6) are within
one std (noise); gains at nB≥5 (+2.0 to +3.8 for nB∈[6,12]) exceed the per-draw std
(real effect). Figures (with error bars / std bands): `novel_blockage_accuracy.png`,
`novel_blockage_rsrp.png`, `novel_blockage_topk.png`, `novel_gate_behavior.png`.
Per-cell std also in `results/metrics/novel_metrics.json`.

## 2a. Seed-variance of the i.i.d. blockage gain (paper Table III) — `novel/exp_seed_variance.m`
The §2 table above is one trained model pair at seed=1, with ±std over blockage DRAWS.
To show the gain survives weight-initialization noise, BOTH models were retrained across
**5 random seeds** (each per-seed acc averaged over the same 10 i.i.d. draws). Reported as
**mean ± std ACROSS SEEDS** — this is the paper's Table III. Fig `novel_blockage_accuracy.png`
now uses these across-seed error bars (`novel/make_seedvar_fig.m`). Raw per-seed data:
`results/metrics/seed_variance.{mat,json}`.

| # blocked (of 14) | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| RSRP-only | 88.7±0.8 | 84.2±0.6 | 77.7±0.5 | 69.3±0.6 | 56.0±0.4 | 36.2±0.4 | 22.6±0.3 |
| **Gated fusion** | 89.0±0.4 | 83.2±0.3 | 78.5±0.5 | 71.8±0.3 | 58.3±0.4 | 39.5±0.3 | 25.2±0.3 |
| gain (pts) | +0.3±0.6 | −1.0±0.7 | +0.8±0.7 | +2.5±0.7 | +2.3±0.8 | +3.2±0.6 | +2.6±0.3 |

- **Clean parity holds across seeds** (gain +0.3±0.6); fusion is also the *more stable* model
  (std 0.4 vs 0.8). No clean-data cost survives the seed average.
- **Blockage gains survive seeds:** at nB∈[6,12] every gain (+2.3 to +3.2) exceeds **2× its
  across-seed std** → real, not an init artifact. Low-blockage deltas (nB≤4) are small/inconsistent.
- Honest headline: **+2 to +3.2 pts (i.i.d., across 5 seeds)**, vs the single-seed §2 view (+2 to +3.8).
  per-seed clean RSRP-only acc = [89.2 89.0 87.4 89.4 88.4]; clean fusion = [89.4 88.6 88.6 89.2 89.0].

## 3. Architecture ablation (gate necessity) — `novel/exp_ablation_arch.m`
acc@K=13 vs # blocked beams; all fusion variants trained with identical aug.

Mean ± std over 10 blockage draws at the canonical seed=1 (paper **Table V**).
All four variants are scored on **one shared set of 10 draws** by the single
evaluator `run_gated.m`, so the RSRP-only and Gated cells here match §2 exactly.
(Paper Table III instead aggregates the RSRP-only/Gated rows **across 5 seeds**,
so its cells differ slightly from these single-seed numbers — see §2a.)

| # blocked | RSRP-only | Concat | Residual (no gate) | **Gated** |
|---|---|---|---|---|
| 0 (clean) | 89.2±0.0 | 84.4±0.0 | 88.0±0.0 | **89.4±0.0** |
| 6 | 69.8±1.5 | 69.7±1.7 | 71.7±1.8 | 72.1±1.7 |
| 8 | 56.0±1.6 | 59.8±1.5 | 58.0±2.1 | 57.9±2.0 |
| 10 | 35.9±2.4 | **45.9±1.2** | 39.7±1.8 | 39.7±1.7 |
| 12 | 22.3±0.9 | 29.1±1.2 | 25.0±1.4 | 24.9±1.4 |

- **Gate justified:** it preserves clean parity (89.4 vs no-gate 88.0) at equal
  blockage robustness. The learned gate is ≈0.78 (near-constant) — a learned
  down-weighting of the position correction, NOT an adaptive open/close.
- **Tradeoff:** ungated **concatenation** yields the largest heavy-blockage gains
  (+10 @10 blocked) but costs ~4.8 pts on clean data. The gated residual is the
  best no-regret operating point.

## 4. Negative results on clean RSRP (motivate the blockage framing)
On clean RSRP the single-modality baseline is near the achievable ceiling;
the following all FAILED to beat it (≤±1 pt, within noise). Scripts in
`novel/explorations/`.

| Idea | Outcome vs baseline (clean) |
|---|---|
| Position fusion (naive concat) | tie / slightly worse |
| "Smart" fixed beam sampling (var/freq/mean) | worse (uniform every-5th is best) |
| 2D beam-grid CNN inpainting | worse at operating point (K@90=16 vs 13) |
| Uncertainty-aware adaptive-K | negligible (≤+0.6 pts at matched mean-K) |

Takeaway: extra modality/structure is redundant when RSRP is clean; the value
appears only when the RSRP modality is degraded (blockage).

## 5. Baseline ablations (Task 6) — `novel/exp_ablation_baseline.m`
RSRP-only, clean.

**Width (4 layers):** K@90 = 14/13/13/12/12 for width 32/48/96/192/384
(acc@K13 89.6→91.8). 96 is a good accuracy/size balance.
**Depth (width 96):** 2–4 layers ≈ 90% @K13; 6–8 layers slightly worse
(K@90 14–15) — mild overfitting. 4 hidden layers justified.
**UE density (train fraction):** acc@K13 = 81.0 / 87.8 / 90.6 / 90.4 / 90.6 for
5% / 10% / 25% / 50% / 100% of training UEs. Saturates by ~25% (≈3,375 UEs); the
full 15k set is in the saturated regime.

## 6. Effect of K and modality
Covered by the Task-1 figures (NN vs KNN vs Statistical vs Random across K=1..70)
and the §2 blockage Top-K curve. NN dominates all single-signal benchmarks across
K on clean data; under blockage the gated fusion dominates RSRP-only across K
(`novel_blockage_topk.png`).

## 7. Correlated (contiguous) blockage — additional robustness study — `novel/exp_correlated_blockage.m`
Same i.i.d.-trained nets, but the test-time blockage drops a CONTIGUOUS run of nB
sampled beams (single obstruction) instead of i.i.d.-random. acc@K=13, mean±std,
10 draws. The gated-fusion advantage **grows** vs the i.i.d. case.

| # blocked | 0 | 2 | 4 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|---|---|
| RSRP-only | 89.2±0.0 | 84.0±1.1 | 78.7±1.1 | 70.4±1.4 | 57.9±1.2 | 34.3±1.0 | 23.7±1.6 |
| **Gated fusion** | 89.4±0.0 | 83.0±1.3 | 80.6±1.8 | 77.0±1.5 | 65.2±1.6 | 41.3±1.6 | 26.2±1.3 |
| gain (pts) | +0.2 | −0.9 | +1.8 | **+6.5** | **+7.3** | **+7.0** | +2.5 |

vs i.i.d. gains (§2): +2.3/+2.0/+3.8 at nB=6/8/10. A contiguous gap removes a whole
local region of the RSRP profile (no nearby samples to interpolate), so the global
position prior helps more. Figure: `novel_contiguous_blockage.png`. Presented as an
additional study (paper §VI-E); the i.i.d. result remains the headline.
