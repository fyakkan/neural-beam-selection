# Literature Survey — ML for mmWave Beam Management (2021–2025)

Focus: machine-learning beam selection/prediction that reduces beam-sweep
overhead, with emphasis on **side-information / multi-modal** approaches —
the context for our gated position-aided fusion method. Quantitative results
are as reported by each paper. (Exact authors/venues/years to be confirmed in
`ref.bib` before paper submission; entries marked ⚠ need verification.)

## Comparison table

| # | Work (year) | Input modality | Approach | Dataset | Key quantitative result |
|---|---|---|---|---|---|
| 1 | Heng & Andrews, GLOBECOM 2019 | UE position | NN classifier of beam pair | Ray-tracing | Baseline ML beam alignment; basis of the MathWorks example |
| 2 | Klautau et al., ITA 2018 | LiDAR + position | DL beam selection | Raymobtime | Established ML-for-beam-selection w/ out-of-band data |
| 3 | Sim et al., IEEE Access 2020 | Sub-6 GHz channel | DL beam selection (NR/6G) | Prototype + sim | ~**79.3%** beam-sweep overhead reduction |
| 4 | Alrabeiah & Alkhateeb, IEEE TWC 2020 | Sub-6 GHz channel | DL beam + blockage prediction | DeepMIMO | Predicts beams w/ ~no training overhead; **>90%** blockage pred. |
| 5 | Heng & Andrews, IEEE TWC 2022 | Position + orientation | DL 3D beam selection | Ray-tracing (arXiv 2110.06859) | Location/orientation-aided beam set selection |
| 6 | Alkhateeb et al., IEEE Comm. Mag. 2023 | Multi-modal (RF/LiDAR/cam/radar/GPS) | DeepSense 6G dataset + tasks | Real-world | First large-scale real multi-modal sensing+comm dataset |
| 7 | Jiang & Alkhateeb, IEEE WCL 2023 ⚠ | LiDAR | DL future beam prediction | DeepSense 6G (arXiv 2203.05548) | **~95%** optimal-beam, **~90%** overhead reduction (V2I) |
| 8 | Demirhan & Alkhateeb, WCNC 2022 | Radar | DL beam prediction | DeepSense 6G | Real-world radar-aided beam prediction |
| 9 | Morais et al. (Alkhateeb), 2023–24 ⚠ | GPS position | DL position-aided beam pred. | DeepSense 6G | Real-world position-aided; up to **~95%** training-overhead cut |
| 10 | Wu et al. (Alkhateeb), ICC 2022 | In-band mmWave power | DL moving-blockage prediction | Real mmWave | Blockage predicted with **>85%** accuracy (proactive handoff) |
| 11 | Alkhateeb et al., 2018 (arXiv 1807.02723) | Sub-6 / beam sequence | ML blockage pred. + proactive handoff | DeepMIMO | Foundational reliable-mmWave-via-ML |
| 12 | Polese et al., "DeepBeam" (arXiv 2012.14350) | Waveform/RF fingerprints | DL coordination-free beam mgmt | Over-the-air | Up to **7×** lower latency vs NR initial sweep |
| 13 | Survey, IEEE COMST 2024 ⚠ | — | Beam mgmt for mmWave/THz toward 6G | — | Taxonomy of beam-management techniques |
| 14 | Survey, IEEE Wireless Comm. 2023 ⚠ | — | DL for mmWave beam mgmt: SoTA & challenges | — | Opportunities/challenges overview |
| 15 | Survey, Sensors (MDPI) 2023, 23(9):4359 | — | AI-aided beamforming/beam mgmt 5G/6G | — | Literature survey |
| 16 | 3GPP TR 38.843 | — | Study on AI/ML for NR air interface | — | Standardization framework (incl. beam management) |

## Synthesis / gap our work addresses

- **Two dominant single-modality families.** (a) *RSRP/channel-based* prediction
  (rows 1,3,4,12 and the MathWorks regression benchmark) — highly accurate when
  measurements are clean; (b) *side-information-based* (position/LiDAR/radar/
  vision/sub-6 GHz; rows 2,5,6,7,8,9) — robust to RF degradation but generally
  lower peak accuracy.
- **Multi-modal fusion** (row 6 DeepSense and challenge entrants) typically fuses
  rich external sensors (cameras/LiDAR/radar), which need extra hardware and are
  studied mostly on V2X/vehicular data.
- **Gap.** For the canonical SSB-RSRP beam-selection setting (MathWorks
  benchmark), no prior work fuses the *sparse in-band RSRP* with the *already-
  available UE position* and studies the fusion's value **as a function of RSRP
  reliability**. Our finding: on clean RSRP the position modality is redundant
  (single-modality is near-optimal), but under **beam blockage / lost
  measurements** a gated residual position path restores accuracy at **no clean-
  data cost** — a no-regret robustness contribution distinct from the accuracy-
  on-clean-data framing of prior fusion work.

## Notes for paper
- Cite the benchmark's own refs (1,2,3) as background; position the novelty
  against the position/multi-modal line (5,6,7,9) and blockage line (4,10,11).
- Verify ⚠ entries (exact authors/year/venue) before finalizing `ref.bib`.
- Quantitative numbers above are taken from abstracts/search snippets — confirm
  each against the source PDF when citing a specific figure.
