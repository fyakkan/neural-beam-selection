# Blockage-Robust Neural Beam Selection for 5G NR mmWave

Reproduction of the MathWorks *Neural Network for Beam Selection* (regression)
benchmark, and a novel **gated position-aided fusion** method that makes neural
beam selection robust to mmWave **beam blockage** at no clean-data cost.

The task: instead of an exhaustive sweep over all **70 beam pairs** (10 Tx × 7 Rx)
at initial access, a neural network predicts the full per-beam RSRP profile from a
small set of **14 downsampled RSRP measurements**, and only the top-**K** predicted
beams are swept. Scenario: UMa, FR2 @ 30 GHz (3GPP TR 38.901 / TR 38.843).

## Key results

**Benchmark (Task 1):** the regression network reaches **90% top-K accuracy at
K=13** (≈**81.4%** beam-sweep overhead reduction), with average RSRP within
**0.24 dB** of exhaustive search — beating KNN/Statistical/Random.

**Proposed method:** a gated residual network fuses the 14 RSRP measurements with
the UE 3D position: `out = tanh(base(RSRP) + gate·corr(position))`. It is a
**no-regret** design — it matches the RSRP-only model on clean data and improves
top-K accuracy by **+2 to +4 points under beam blockage** (when 6–12 of the 14
measured beams are lost).

| # beams blocked (of 14) | 0 | 6 | 8 | 10 | 12 |
|---|---|---|---|---|---|
| RSRP-only (top-K=13, %) | 89.2 | 69.8 | 56.0 | 35.9 | 22.3 |
| **Gated fusion (%)** | 89.4 | 72.1 | 57.9 | 39.7 | 24.9 |

Full numbers, ablations, and the clean-data analysis are in
[`results/RESULTS.md`](results/RESULTS.md). The paper is in
[`paper/`](paper/main.pdf).

## Repository layout

```
baseline/            regression benchmark (Task 1)
  loadBeamData.m        load + preprocess official data (configurable downsampling/impairment)
  buildRegressionNet.m  4-hidden-layer MLP (tanh output)
  evalTopK.m            Top-K accuracy + average RSRP for all methods
  run_baseline.m        train, evaluate, plot, log metrics   (entry point)
  data/                 official pre-recorded data (nnBS_*.mat)
  helpers/              official TR 38.901 channel helpers (unmodified)
novel/               proposed method + ablations
  applyImpairment.m     RSRP noise / quantization / beam-blockage model
  buildFusionNet.m      concatenation fusion (ablation)
  buildGatedFusionNet.m gated residual fusion (proposed; useGate toggles the gate)
  run_gated.m           train + evaluate the proposed method   (entry point)
  exp_ablation_arch.m   architecture ablation (gate necessity)
  exp_ablation_baseline.m  width / depth / UE-density ablations
  exp_impairments.m     noise / quantization / blockage sweeps
  explorations/         negative-result experiments (clean-data: CNN, sampling, adaptive-K, fusion)
results/figures/     exported figures (PNG)
results/metrics/     metrics (JSON + .mat)
results/RESULTS.md   consolidated results and tables
paper/               IEEE conference paper (main.tex, ref.bib, figures/, main.pdf)
```

## Requirements

MATLAB **R2025b** with the Deep Learning, 5G, Communications, Phased Array System,
Statistics and Machine Learning, DSP System, and Signal Processing Toolboxes.
No GPU required (networks are small; CPU training is ~25 s baseline, ~3 min fusion).

For the paper: any LaTeX distribution with `IEEEtran` (or upload `paper/` to Overleaf).

## Reproduce

```matlab
% Benchmark (Task 1)
cd baseline
run_baseline                       % train, evaluate, save figures + metrics
run_baseline('doTraining', false)  % re-evaluate the saved net

% Proposed method (gated fusion, blockage robustness)
cd ../novel
run_gated                          % train RSRP-only + gated fusion, sweep blockage, plot

% Ablations
exp_ablation_arch                  % gate necessity (RSRP-only / concat / residual / gated)
exp_ablation_baseline              % width, depth, UE density
```

Headless / terminal:

```bash
cd baseline && /Applications/MATLAB_R2025b.app/bin/matlab -batch "run_baseline"
cd ../novel && /Applications/MATLAB_R2025b.app/bin/matlab -batch "run_gated"
```

## Note on the source example

The example is opened with
`openExample('deeplearning_shared/NeuralNetworkBeamSelectionExample')`. The
*online* documentation is the **regression** version reproduced here, but the copy
that ships with the locally installed R2025b is an older position→beam
*classification* variant. This repository therefore implements a faithful
reconstruction of the regression benchmark from the official pre-recorded data
(which contains both UE positions and the full 7×10×N RSRP matrices) and the
documented architecture — no channel re-simulation is needed. The bundled data
uses 15,000 training / 500 test UE locations.
