# Phone-level3 Romaji-rev — Devoicing Detection (stratified, test split) — Updated 2026-08-02

Full 12-way BS×LR×DW sweep (`scripts/sweep_runner.py`, 4 GPUs in parallel), all runs
with `ENC_PADDING_MASK=1` (decoder cross-attention padding-mask fix).

| model config | PER ↓ (%) DW1 / DW5 | KER ↓ (%) DW1 / DW5 | Precision ↑ (%) DW1 / DW5 | Recall ↑ (%) DW1 / DW5 | F1 ↑ (%) DW1 / DW5 | CCDA ↑ (%) DW1 / DW5 | DEO ↑ (%) DW1 / DW5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| BS: 8; LR: 1e-3 * | 2.65 / **2.61** | 3.42 / 3.42 | 91.79 / **92.34** | 93.40 / **94.20** | 92.59 / **93.26** | 92.45 / **93.40** | 94.34 / 94.34 |
| BS: 8; LR: 1e-4 | 8.45 / 8.30 | 10.63 / 10.30 | 87.27 / 87.09 | 87.74 / 89.08 | 87.50 / 88.07 | 86.66 / 88.27 | 90.57 / 91.64 |
| BS: 16; LR: 1e-3 | 3.04 / 3.32 | 3.90 / 4.27 | **92.34** / 90.94 | 92.59 / 93.40 | 92.46 / 92.15 | 91.64 / 92.18 | 93.67 / 94.34 |
| BS: 16; LR: 1e-4 | 13.01 / 13.11 | 15.86 / 16.36 | 83.33 / 81.88 | 83.56 / 87.06 | 83.45 / 84.39 | 84.23 / 86.66 | 88.81 / 92.72 |
| BS: 32; LR: 1e-3 | 3.39 / 3.69 | 4.28 / 4.68 | 92.62 / 91.44 | 92.99 / 93.53 | 92.80 / 92.47 | 91.78 / 91.64 | 93.67 / 93.94 |
| BS: 32; LR: 1e-4 | 17.35 / 18.40 | 21.53 / 22.96 | 81.04 / 75.97 | 79.51 / 82.21 | 80.27 / 78.96 | 79.78 / 83.56 | 85.71 / 90.16 |

\* BS8/LR1e-3/DW1 was previously reported as PER 2.47% (epoch-88 checkpoint, training
killed before completion — see `doc/archive/results_20260802.md`). It was fully re-run
2026-08-02 (epoch-85 checkpoint, 100/100 epochs) and now scores PER 2.65% / CCDA 92.45%,
both worse than the earlier provisional number and worse than DW5 in the same row. That
re-run also had `ALLOW_TF32=1` on, which the rest of the sweep didn't use, so it isn't a
perfectly clean like-for-like re-run — but it's the best completed number available for
this cell, and the config to treat as authoritative going forward. Full writeup of the
2026-08-02 TF32 re-run experiment (including a DW5 rerun under the same flag that was
*not* used to replace the table, since the original DW5 number was already a clean
completed run) is in `doc/archive/results_20260802.md`.

## Summary

- **Best overall: BS 8 / LR 1e-3 / DW5** — wins every metric in its row outright or
  ties for it: PER 2.61% (best in the whole sweep), KER 3.42% (tied), F1 93.26%,
  CCDA 93.40%, DEO 94.34% (tied). DW1 in the same row looked like it had the best PER
  (2.47%) in earlier versions of this doc, but that number came from an incomplete
  run; the completed re-run puts DW1 at PER 2.65% / CCDA 92.45%, both behind DW5, so
  this is no longer a split result between two configs — DW5 is the clean winner.
- lr1e-3 dominates lr1e-4 across every batch size and both DW settings, same
  pattern as every prior sweep in this doc.
- DW5 gives a consistent small F1/CCDA edge over DW1 at bs8/bs16 (matching the
  2026-07-28-vs-07-29 comparison above), but DW1 pulls ahead at bs32 lr1e-3
  (92.80 vs 92.47 F1) — unlike prior sweeps, bs32/lr1e-4/DW1 also beats DW5 on
  every metric except CCDA, so the DW1-vs-DW5 edge is no longer a clean sweep
  in either direction the way it was pre-`--enc-padding-mask`.
- Given the transcript/eval-target caveat above, whether `--enc-padding-mask`
  itself helped or hurt relative to the pre-07-30 runs can't be read off these numbers directly — a same-transcript A/B (mask on vs off) would be needed to isolate that effect.
- Whether `ALLOW_TF32=1` itself helped or hurt the re-run DW1 number is likewise
  unconfirmed from a single run — the same flag made the 2026-08-02 DW5 rerun
  noticeably *worse* (PER 3.27% vs. this table's 2.61%), so treat TF32 as an open
  variable, not a settled improvement. See `doc/archive/results_20260802.md`.

## Aug 2 batch: DW1 vs DW5 (same-day comparison)

Both runs below are BS8/LR1e-3, `ENC_PADDING_MASK=1`, `ALLOW_TF32=1`, best checkpoint at
epoch 85 — the only pair in this doc that isolates DW1 vs DW5 under identical flags on
the same day. DW5 ran DataParallel across all 4 GPUs (unpinned); DW1 was pinned to a
single GPU.

| run (start) | config | PER ↓ (%) | KER ↓ (%) | Precision ↑ (%) | Recall ↑ (%) | F1 ↑ (%) | CCDA ↑ (%) | DEO ↑ (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-08-02 04:40 | DW1 | **2.65** | **3.42** | **91.79** | 93.40 | 92.59 | 92.45 | 94.34 |
| 2026-08-02 04:27 | DW5 | 3.27 | 4.19 | 90.27 | **95.01** | 92.58 | **93.53** | **96.23** |

- DW1 wins cleanly on PER (by 0.62pp), KER (0.77pp), and Precision; DW5 wins on Recall,
  CCDA, and DEO; F1 is a dead heat (92.59 vs 92.58).
- This is the opposite split from the main sweep table above, where the headline DW5
  row beats DW1 on every metric — because that headline DW5 number (PER 2.61%) comes
  from the clean 07-31 run, not this noisier Aug-2 rerun (PER 3.27%). Read this
  comparison as "DW1 tolerated the TF32/DataParallel conditions better than DW5 did
  that morning," not as evidence that DW1 beats DW5 in general — it doesn't overturn
  the "Best overall: DW5" conclusion above, which rests on the cleaner runs.
