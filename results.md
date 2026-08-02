# Phoneme Level — Vowel Devoicing Detection (stratified, test split) — Updated 2026-08-02

| model config | PER ↓ (%) DW1 / DW5 | KER ↓ (%) DW1 / DW5 | Precision ↑ (%) DW1 / DW5 | Recall ↑ (%) DW1 / DW5 | F1 ↑ (%) DW1 / DW5 | CCDA ↑ (%) DW1 / DW5 | DEO ↑ (%) DW1 / DW5 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **BS: 8; LR: 1e-3** | **2.65** / 3.27 | **3.42** / 4.19 | 91.79 / 90.27 | 93.40 / **95.01** | 92.59 / 92.58 | 92.45 / **93.53** | 94.34 / **96.23** |
| BS: 8; LR: 1e-4 | 8.45 / 8.30 | 10.63 / 10.30 | 87.27 / 87.09 | 87.74 / 89.08 | 87.50 / 88.07 | 86.66 / 88.27 | 90.57 / 91.64 |
| BS: 16; LR: 1e-3 | 3.04 / 3.32 | 3.90 / 4.27 | 92.34 / 90.94 | 92.59 / 93.40 | 92.46 / 92.15 | 91.64 / 92.18 | 93.67 / 94.34 |
| BS: 16; LR: 1e-4 | 13.01 / 13.11 | 15.86 / 16.36 | 83.33 / 81.88 | 83.56 / 87.06 | 83.45 / 84.39 | 84.23 / 86.66 | 88.81 / 92.72 |
| BS: 32; LR: 1e-3 | 3.39 / 3.69 | 4.28 / 4.68 | **92.62** / 91.44 | 92.99 / 93.53 | **92.80** / 92.47 | 91.78 / 91.64 | 93.67 / 93.94 |
| BS: 32; LR: 1e-4 | 17.35 / 18.40 | 21.53 / 22.96 | 81.04 / 75.97 | 79.51 / 82.21 | 80.27 / 78.96 | 79.78 / 83.56 | 85.71 / 90.16 |

## Summary

- Best model config: BS 8 / LR 1e-3 — achieves the lowest (best) PER and KER among global metrics; also leads on vowel devoicing metrics CCDA and DEO
- BS 32 / LR 1e-3 marginally outperforms BS 8 / LR 1e-3, specifically on Precision and F1-score
- The batch size effect shows smaller batch sizes improve metrics overall, meaning more frequent gradient updates yield a regularization benefit
- Learning rate has the dominant effect, significantly improving all metrics across configurations; this suggests the lower-LR runs may still converge given a longer training schedule (more epochs)
- The devoicing weight (DW) introduces a trade-off: it improves model sensitivity (recall) to vowel devoicing — reflected in CCDA, DEO, and Recall — but degrades PER, KER, and Precision
 
