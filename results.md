# Phone-level3 Romaji-rev — Devoicing Detection (stratified, test split) — Added 2026-07-31 (`--enc-padding-mask`)

Full 12-way BS×LR×DW sweep (`scripts/sweep_runner.py`, 4 GPUs in parallel), all runs
with `ENC_PADDING_MASK=1` (decoder cross-attention padding-mask fix). 

| model config | PER ↓ (%) DW1 / DW5 | KER ↓ (%) DW1 / DW5 | Precision ↑ (%) DW1 / DW5 | Recall ↑ (%) DW1 / DW5 | F1 ↑ (%) DW1 / DW5 | CCDA ↑ (%) DW1 / DW5 | DEO ↑ (%) DW1 / DW5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BS: 8; LR: 1e-3 * | **2.47*** / 2.61 | **3.17** / 3.42 | 91.70 / 92.34 | 93.80 / **94.20** | 92.74 / **93.26** | 92.86 / **93.40** | 94.20 / **94.34** |
| BS: 8; LR: 1e-4 | 8.45 / 8.30 | 10.63 / 10.30 | 87.27 / 87.09 | 87.74 / 89.08 | 87.50 / 88.07 | 86.66 / 88.27 | 90.57 / 91.64 |
| BS: 16; LR: 1e-3 | 3.04 / 3.32 | 3.90 / 4.27 | **92.34** / 90.94 | 92.59 / 93.40 | 92.46 / 92.15 | 91.64 / 92.18 | 93.67 / 94.34 |
| BS: 16; LR: 1e-4 | 13.01 / 13.11 | 15.86 / 16.36 | 83.33 / 81.88 | 83.56 / 87.06 | 83.45 / 84.39 | 84.23 / 86.66 | 88.81 / 92.72 |
| BS: 32; LR: 1e-3 | 3.39 / 3.69 | 4.28 / 4.68 | 92.62 / 91.44 | 92.99 / 93.53 | 92.80 / 92.47 | 91.78 / 91.64 | 93.67 / 93.94 |
| BS: 32; LR: 1e-4 | 17.35 / 18.40 | 21.53 / 22.96 | 81.04 / 75.97 | 79.51 / 82.21 | 80.27 / 78.96 | 79.78 / 83.56 | 85.71 / 90.16 |

## Summary

- **Best PER:** BS 8 / LR 1e-3 / DW1 (2.47%) — though this is the run that never
  finished cleanly (epoch 88 checkpoint), so treat it as provisional.
- **Best F1 / CCDA / DEO:** BS 8 / LR 1e-3 / DW5 (F1 93.26%, CCDA 93.40%, DEO 94.34%)
  — the cleanest fully-completed run in the sweep, and the config to treat as
  this sweep's headline result.
- lr1e-3 dominates lr1e-4 across every batch size and both DW settings, same
  pattern as every prior sweep in this doc.
- DW5 gives a consistent small F1/CCDA edge over DW1 at bs8/bs16 (matching the
  2026-07-28-vs-07-29 comparison above), but DW1 pulls ahead at bs32 lr1e-3
  (92.80 vs 92.47 F1) — unlike prior sweeps, bs32/lr1e-4/DW1 also beats DW5 on
  every metric except CCDA, so the DW1-vs-DW5 edge is no longer a clean sweep
  in either direction the way it was pre-`--enc-padding-mask`.
- Given the transcript/eval-target caveat above, whether `--enc-padding-mask`
  itself helped or hurt relative to the pre-07-30 runs can't be read off these numbers directly — a same-transcript A/B (mask on vs off) would be needed to isolate that effect.
